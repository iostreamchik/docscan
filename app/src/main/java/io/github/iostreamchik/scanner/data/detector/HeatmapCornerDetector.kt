package io.github.iostreamchik.scanner.data.detector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import io.github.iostreamchik.scanner.data.utils.computeMaxAngleDeviation
import io.github.iostreamchik.scanner.data.utils.sortQuadPoints
import io.github.iostreamchik.scanner.data.utils.toBitmap
import io.github.iostreamchik.scanner.data.opencv.IMatBundle
import io.github.iostreamchik.scanner.entity.DetectionParameters
import io.github.iostreamchik.scanner.entity.IntermediateBitmaps
import io.github.iostreamchik.scanner.entity.PipelineParams
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import androidx.core.graphics.createBitmap
import org.opencv.geometry.Geometry
import kotlin.math.max
import kotlin.math.min

class HeatmapCornerDetector(
    private val context: Context,
    private val env: OrtEnvironment,
    private val matBundle: IMatBundle,
    private val modelPath: String = "onnx/lcnet100_h_e_bifpn_256_fp32.onnx",
) : IDocumentDetector {

    private val sessionManager = OnnxSessionManager(context, env, modelPath)

    var heatmapThreshold: Float = 0.3f
        set(value) {
            field = value.coerceIn(0.05f, 0.7f)
        }

    var minCornerArea: Float = 0.0001f

    var maxAngleDeviation: Double = 45.0
        set(value) {
            field = value.coerceIn(5.0, 60.0)
        }

    private val _detectionParams = MutableStateFlow(DetectionParameters())
    override val detectionParams = _detectionParams.asStateFlow()

    private var cachedCorners: List<Point>? = null
    private var cachedRawMat: Mat? = null
    private var cachedCornerBitmap: Bitmap? = null

    private var originalWidth = 0
    private var originalHeight = 0

    private val channelData = FloatArray(INPUT_SIZE * INPUT_SIZE)
    private val hierarchy = Mat()

    override suspend fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat {
        sessionManager.init(TAG)

        cachedCorners = null
        cachedCornerBitmap?.recycle()
        cachedCornerBitmap = null
        cachedRawMat?.release()
        cachedRawMat = null

        cachedRawMat = rawMat.clone()
        originalWidth = rawMat.cols()
        originalHeight = rawMat.rows()

        val sess = sessionManager.getSession() ?: return matBundle.getMorph()

        val resized = matBundle.getBlurred()
        Imgproc.resize(rawMat, resized, Size(INPUT_SIZE.toDouble(), INPUT_SIZE.toDouble()))

        val rgb = matBundle.getGray()
        Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_RGBA2RGB)

        rgb.convertTo(rgb, CvType.CV_32FC3, 1.0 / 255.0)

        val inputTensor = sessionManager.prepareInputTensor(rgb, INPUT_SIZE, INPUT_CHANNELS)

        val output: OrtSession.Result
        try {
            output = sess.run(mapOf(sessionManager.inputName!! to inputTensor))
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            inputTensor.close()
            return matBundle.getMorph()
        }

        val outputTensor = output.get(0) as OnnxTensor
        val outputShape = outputTensor.getInfo().shape
        val heatmapH = outputShape[2].toInt()
        val heatmapW = outputShape[3].toInt()
        val totalElements = outputShape.fold(1L) { acc, dim -> acc * dim }.toInt()

        Log.d(TAG, "Heatmap output: shape=${outputShape.contentToString()}, h=${heatmapH}x${heatmapW}")

        val outputData = FloatArray(totalElements)
        outputTensor.floatBuffer.get(outputData)
        outputTensor.close()

        output.close()
        inputTensor.close()

        val corners = mutableListOf<Point?>()
        for (channel in 0 until HEATMAP_CHANNELS) {
            val corner = extractCornerFromHeatmap(outputData, channel, heatmapH, heatmapW)
            corners.add(corner)
        }

        val validCorners = corners.filterNotNull()
        cachedCorners = if (validCorners.size == 4) validCorners else null

        _detectionParams.value = _detectionParams.value.copy(
            detectorName = "Heatmap Corner",
            heatmapThreshold = "%.2f".format(heatmapThreshold),
            cornerError = if (cachedCorners == null) "only ${validCorners.size}/4 corners" else ""
        )

        if (cachedCorners != null) {
            val cornerBitmap = buildCornerVisualization(cachedCorners!!)
            cachedCornerBitmap = cornerBitmap
        }

        return matBundle.getMorph()
    }

    private fun extractCornerFromHeatmap(
        flatData: FloatArray,
        channel: Int,
        heatmapH: Int,
        heatmapW: Int
    ): Point? {
        val channelSize = heatmapH * heatmapW
        val channelOffset = channel * channelSize
        System.arraycopy(flatData, channelOffset, channelData, 0, channelSize)

        val heatmap = matBundle.getTemp()
        heatmap.create(heatmapH, heatmapW, CvType.CV_32FC1)
        heatmap.put(0, 0, channelData)

        val binary = matBundle.getEnhanced()
        binary.create(heatmapH, heatmapW, CvType.CV_8UC1)
        heatmap.convertTo(binary, CvType.CV_8UC1, 255.0)

        val threshValue = heatmapThreshold * 255.0
        Imgproc.threshold(binary, binary, threshValue, 255.0, Imgproc.THRESH_BINARY)

        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

        var bestContour: MatOfPoint? = null
        var bestArea = 0.0
        val minArea = (minCornerArea * heatmapH * heatmapW).coerceAtLeast(4.0f)

        for (contour in contours) {
            val area = Geometry.contourArea(contour)
            if (area > minArea && area > bestArea) {
                bestArea = area
                bestContour = contour
            }
        }

        val corner = bestContour?.let { contour ->
            val moments = Geometry.moments(contour, false)
            if (moments.m00 > 0) {
                val scaleX = originalWidth.toDouble() / heatmapW
                val scaleY = originalHeight.toDouble() / heatmapH
                Point(moments.m10 / moments.m00 * scaleX, moments.m01 / moments.m00 * scaleY)
            } else {
                null
            }
        }

        contours.forEach { it.release() }
        bestContour?.release()

        return corner
    }

    override suspend fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        rotation: Int,
        params: PipelineParams
    ): MatOfPoint? {
        val corners = cachedCorners ?: return null

        if (corners.size < 4) {
            _detectionParams.value = _detectionParams.value.copy(
                cornerError = "only ${corners.size}/4 corners"
            )
            return null
        }

        val sorted = sortQuadPoints(corners)
        if (sorted.size != 4) return null

        if (!validateCornerGeometry(sorted)) {
            _detectionParams.value = _detectionParams.value.copy(
                cornerError = "geometry failed"
            )
            return null
        }

        return MatOfPoint(*sorted.toTypedArray())
    }

    private fun validateCornerGeometry(corners: List<Point>): Boolean {
        if (corners.size != 4) return false

        if (computeMaxAngleDeviation(corners) > maxAngleDeviation) return false

        val xs = corners.map { it.x }
        val ys = corners.map { it.y }
        val width = xs.maxOrNull()!! - xs.minOrNull()!!
        val height = ys.maxOrNull()!! - ys.minOrNull()!!
        val aspectRatio = min(width, height) / max(width, height)
        if (aspectRatio < 0.15) return false

        return true
    }

    override fun captureIntermediateSnapshots(
        rotation: Int
    ): IntermediateBitmaps {
        val cornerBitmap = cachedCornerBitmap
        return if (cornerBitmap != null) {
            IntermediateBitmaps(
                corners = cornerBitmap.copy(Bitmap.Config.ARGB_8888, false)
            )
        } else {
            IntermediateBitmaps()
        }
    }

    private fun buildCornerVisualization(corners: List<Point>): Bitmap {
        val baseBitmap = cachedRawMat?.toBitmap()
            ?: createBitmap(originalWidth, originalHeight)

        val width = baseBitmap.width
        val height = baseBitmap.height

        val bitmap = if (baseBitmap.config == Bitmap.Config.ARGB_8888) {
            baseBitmap
        } else {
            val converted = createBitmap(width, height)
            Canvas(converted).drawBitmap(baseBitmap, 0f, 0f, null)
            if (baseBitmap != converted) baseBitmap.recycle()
            converted
        }

        val canvas = Canvas(bitmap)

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
            strokeWidth = 3f
            color = Color.WHITE
        }

        val px = FloatArray(4)
        val py = FloatArray(4)
        for (i in 0 until 4) {
            px[i] = corners[i].x.toFloat()
            py[i] = corners[i].y.toFloat()
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

    override fun release() {
        sessionManager.close()
        hierarchy.release()
        cachedCorners = null
        cachedCornerBitmap?.recycle()
        cachedCornerBitmap = null
        cachedRawMat?.release()
        cachedRawMat = null
        matBundle.releaseAll()
    }

    companion object {
        const val INPUT_SIZE = 256
        const val INPUT_CHANNELS = 3
        const val HEATMAP_CHANNELS = 4

        private const val TAG = "HeatmapCornerDetector"
    }
}
