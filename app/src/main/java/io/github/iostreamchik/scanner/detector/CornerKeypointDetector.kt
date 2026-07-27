package io.github.iostreamchik.scanner.detector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import io.github.iostreamchik.scanner.computeAngle
import io.github.iostreamchik.scanner.opencv.IMatBundle
import io.github.iostreamchik.scanner.opencv.PipelineParams
import io.github.iostreamchik.scanner.sortQuadPoints
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class CornerKeypointDetector(
    private val context: Context,
    private val matBundle: IMatBundle,
    private val modelPath: String = "onnx/lcnet050_p_multi_decoder_l3_d64_256_fp32.onnx",
) : IDocumentDetector {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()

    var minScore: Float = 0.3f
        set(value) {
            field = value.coerceIn(0.05f, 0.95f)
        }

    var maxAngleDeviation: Double = 30.0
        set(value) {
            field = value.coerceIn(5.0, 45.0)
        }

    private val _detectionParams = MutableStateFlow(DetectionParameters())
    override val detectionParams = _detectionParams.asStateFlow()

    private var session: OrtSession? = null
    private var inputName: String? = null
    private var isInit = false

    private var cachedCoords: FloatArray? = null
    private var cachedScore: Float = 0f
    private var cachedInputSize = INPUT_SIZE
    private var cachedCornerBitmap: Bitmap? = null

    private fun initModel() {
        try {
            sessionOptions.addXnnpack(emptyMap())
            sessionOptions.setIntraOpNumThreads(1)
            sessionOptions.setMemoryPatternOptimization(true)
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)

            val modelBytes = context.assets.open(modelPath).use { it.readBytes() }
            session = env.createSession(modelBytes, sessionOptions)
            inputName = session!!.inputNames.iterator().next()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load ONNX model", e)
        } finally {
            sessionOptions.close()
        }
    }

    override fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat {
        if (!isInit) {
            initModel()
            isInit = true
        }

        cachedCoords = null
        cachedScore = 0f
        cachedCornerBitmap?.recycle()
        cachedCornerBitmap = null

        val sess = session ?: return matBundle.getMorph()

        val resized = Mat()
        Imgproc.resize(rawMat, resized, Size(INPUT_SIZE.toDouble(), INPUT_SIZE.toDouble()))

        val rgb = Mat()
        Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_RGBA2RGB)
        resized.release()

        Core.normalize(rgb, rgb, 0.0, 1.0, Core.NORM_MINMAX, CvType.CV_32FC3)

        val channels = mutableListOf<Mat>()
        Core.split(rgb, channels)

        val nchwData = FloatArray(INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS)
        val channelSize = INPUT_SIZE * INPUT_SIZE
        for (c in 0 until INPUT_CHANNELS) {
            val channelData = FloatArray(channelSize)
            channels[c].get(0, 0, channelData)
            System.arraycopy(channelData, 0, nchwData, c * channelSize, channelSize)
        }
        channels.forEach { it.release() }
        rgb.release()

        val inputShape = longArrayOf(1, INPUT_CHANNELS.toLong(), INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        val inputTensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(nchwData), inputShape)

        val output: OrtSession.Result = try {
            sess.run(mapOf(inputName!! to inputTensor))
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            inputTensor.close()
            return matBundle.getMorph()
        } finally {
            inputTensor.close()
        }

        val coordsOutput = output.get(0) as OnnxTensor
        val coordsData = FloatArray(8)
        coordsOutput.floatBuffer.get(coordsData)
        coordsOutput.close()

        val scoreOutput = output.get(1) as OnnxTensor
        val scoreLogit = FloatArray(1)
        scoreOutput.floatBuffer.get(scoreLogit)
        val score = sigmoid(scoreLogit[0])
        scoreOutput.close()

        val coords = FloatArray(8)
        for (i in 0 until 8) {
            coords[i] = coordsData[i].coerceIn(0.0f, 1.0f)
        }

        output.close()

        cachedCoords = coords
        cachedScore = score

        _detectionParams.value = _detectionParams.value.copy(
            cornerScore = "%.3f".format(score),
            cornerError = ""
        )

        val cornerBitmap = buildCornerVisualization(scaledWidth, scaledHeight, coords)
        cachedCornerBitmap = cornerBitmap

        return matBundle.getMorph()
    }

    override fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        rotation: Int,
        params: PipelineParams
    ): MatOfPoint? {
        val coords = cachedCoords ?: return null

        if (cachedScore < minScore) {
            _detectionParams.value = _detectionParams.value.copy(
                cornerError = "score ${"%.3f".format(cachedScore)} < min ${"%.3f".format(minScore)}"
            )
            return null
        }

        val scaleX = originalWidth.toDouble() / cachedInputSize
        val scaleY = originalHeight.toDouble() / cachedInputSize

        val corners = mutableListOf<Point>()
        for (i in 0 until 4) {
            val nx = coords[i * 2]
            val ny = coords[i * 2 + 1]
            corners.add(Point(nx * cachedInputSize * scaleX, ny * cachedInputSize * scaleY))
        }

        if (!validateCornerGeometry(corners)) {
            _detectionParams.value = _detectionParams.value.copy(
                cornerError = "geometry failed"
            )
            return null
        }

        val sorted = sortQuadPoints(corners)
        if (sorted.size != 4) return null

        return MatOfPoint(*sorted.toTypedArray())
    }

    override fun validateQuadSize(
        quad: MatOfPoint,
        originalWidth: Int,
        originalHeight: Int
    ): Boolean {
        val pts = quad.toArray()
        val minX = pts.minOf { it.x }
        val maxX = pts.maxOf { it.x }
        val minY = pts.minOf { it.y }
        val maxY = pts.maxOf { it.y }
        val quadArea = (maxX - minX) * (maxY - minY)
        val frameArea = originalWidth * originalHeight
        return quadArea <= frameArea * 0.95
    }

    override fun captureIntermediateSnapshots(
        rotation: Int
    ): IntermediateSnapshots {
        val cornerBitmap = cachedCornerBitmap
        return if (cornerBitmap != null) {
            IntermediateSnapshots(
                corners = cornerBitmap.copy(Bitmap.Config.ARGB_8888, false)
            )
        } else {
            IntermediateSnapshots()
        }
    }

    private fun validateCornerGeometry(corners: List<Point>): Boolean {
        if (corners.size != 4) return false

        var maxDeviation = 0.0
        for (i in 0..3) {
            val angle = computeAngle(
                corners[(i + 1) % 4],
                corners[(i + 3) % 4],
                corners[i]
            )
            maxDeviation = max(maxDeviation, abs(90.0 - angle))
        }
        if (maxDeviation > maxAngleDeviation) return false

        val xs = corners.map { it.x }
        val ys = corners.map { it.y }
        val width = xs.maxOrNull()!! - xs.minOrNull()!!
        val height = ys.maxOrNull()!! - ys.minOrNull()!!
        val aspectRatio = min(width, height) / max(width, height)
        if (aspectRatio < 0.25) return false

        return true
    }

    private fun buildCornerVisualization(
        width: Int,
        height: Int,
        coords: FloatArray
    ): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.BLACK)

        val scaleX = width.toDouble() / cachedInputSize
        val scaleY = height.toDouble() / cachedInputSize

        val colors = intArrayOf(
            Color.RED,
            Color.GREEN,
            Color.BLUE,
            Color.YELLOW
        )

        val dotPaint = Paint().apply {
            style = Paint.Style.FILL
            strokeWidth = 3f
        }
        val linePaint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 2f
            color = Color.WHITE
        }

        val px = FloatArray(4)
        val py = FloatArray(4)
        for (i in 0 until 4) {
            px[i] = (coords[i * 2] * cachedInputSize * scaleX).toFloat()
            py[i] = (coords[i * 2 + 1] * cachedInputSize * scaleY).toFloat()
        }

        for (i in 0 until 4) {
            dotPaint.color = colors[i]
            canvas.drawCircle(px[i], py[i], 8f, dotPaint)
        }

        for (i in 0..3) {
            canvas.drawLine(
                px[i], py[i],
                px[(i + 1) % 4], py[(i + 1) % 4],
                linePaint
            )
        }

        return bitmap
    }

    fun release() {
        session?.close()
        session = null
        cachedCoords = null
        cachedScore = 0f
        cachedCornerBitmap?.recycle()
        cachedCornerBitmap = null
        try {
            env.close()
        } catch (_: Exception) {
        }
    }

    companion object {
        const val INPUT_SIZE = 256
        const val INPUT_CHANNELS = 3

        private const val TAG = "CornerKeypointDetector"

        fun sigmoid(x: Float): Float {
            return 1.0f / (1.0f + kotlin.math.exp(-x))
        }
    }
}
