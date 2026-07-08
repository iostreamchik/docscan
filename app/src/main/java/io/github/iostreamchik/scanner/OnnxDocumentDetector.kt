package io.github.iostreamchik.scanner

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import io.github.iostreamchik.scanner.opencv.IMatBundle
import io.github.iostreamchik.scanner.opencv.PipelineParams
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.sqrt

class OnnxDocumentDetector(
    private val matBundle: IMatBundle,
    private val modelPath: String = "onnx/deeplabv3_mbv3_docseg.onnx"
) : IDocumentDetector {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()

    private val _detectionParams = MutableStateFlow(DetectionParameters())
    override val detectionParams = _detectionParams.asStateFlow()

    private var session: OrtSession? = null
    private var inputName: String? = null
    private var modelLoaded = false
    private var storedContext: Context? = null

    private val inputBuffer = FloatArray(INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS)
    private val outputBuffer = FloatArray(INPUT_SIZE * INPUT_SIZE * OUTPUT_CHANNELS)

    private var isInit = false
    private var isDeepLabV3 = false

    fun initModel(context: Context) {
        storedContext = context
        if (isInit) return
        isInit = true
        isDeepLabV3 = modelPath.contains("deeplabv3")
        try {
            sessionOptions.setIntraOpNumThreads(1)
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)

            val inputStream: InputStream = context.assets.open(modelPath)
            val modelBytes = inputStream.readBytes()
            inputStream.close()

            session = env.createSession(modelBytes, sessionOptions)
            inputName = session!!.inputNames.iterator().next()
            modelLoaded = true
            Log.d(TAG, "Model loaded: $modelPath, input=$inputName, size=${INPUT_SIZE}x$INPUT_SIZE, type=${if (isDeepLabV3) "DeepLabV3" else "BiRefNet"}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load ONNX model", e)
            modelLoaded = false
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
        if (isInit.not()) return Mat()

        if (session == null && storedContext != null) {
            initModel(storedContext!!)
        }
        val sess = session ?: return matBundle.getMorph()

        Log.d(TAG, "preprocess: rawMat=${rawMat.rows()}x${rawMat.cols()}, type=${rawMat.type()}, channels=${rawMat.channels()}")

        // Resize maintaining aspect ratio, then pad to INPUT_SIZE x INPUT_SIZE
        val scale = INPUT_SIZE.toDouble() / maxOf(rawMat.cols(), rawMat.rows())
        val resizedW = (rawMat.cols() * scale).toInt()
        val resizedH = (rawMat.rows() * scale).toInt()

        val resized = Mat()
        Imgproc.resize(rawMat, resized, Size(resizedW.toDouble(), resizedH.toDouble()))

        val rgb = Mat()
        Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_RGBA2RGB)
        resized.release()

        // Pad RGB to square with black borders
        val rgbPadded = Mat(INPUT_SIZE, INPUT_SIZE, CvType.CV_8UC3)
        rgbPadded.setTo(org.opencv.core.Scalar(0.0, 0.0, 0.0))
        val roi = rgbPadded.submat(0, rgb.rows(), 0, rgb.cols())
        rgb.copyTo(roi)
        roi.release()
        rgb.release()

        Log.d(TAG, "preprocess: padded to ${INPUT_SIZE}x${INPUT_SIZE} from ${resizedW}x${resizedH}")

        // Fill input buffer in NCHW layout
        if (isDeepLabV3) {
            for (c in 0 until INPUT_CHANNELS) {
                val channelOffset = c * INPUT_SIZE * INPUT_SIZE
                for (y in 0 until INPUT_SIZE) {
                    for (x in 0 until INPUT_SIZE) {
                        val pixel = rgbPadded.get(y, x)
                        inputBuffer[channelOffset + y * INPUT_SIZE + x] = (pixel[c] / 255.0).toFloat()
                    }
                }
            }
        } else {
            for (c in 0 until INPUT_CHANNELS) {
                val channelOffset = c * INPUT_SIZE * INPUT_SIZE
                for (y in 0 until INPUT_SIZE) {
                    for (x in 0 until INPUT_SIZE) {
                        val pixel = rgbPadded.get(y, x)
                        inputBuffer[channelOffset + y * INPUT_SIZE + x] =
                            (((pixel[c] / 255.0) - IMAGE_MEAN[c]) / IMAGE_STD[c]).toFloat()
                    }
                }
            }
        }

        val directBuffer = ByteBuffer.allocateDirect(inputBuffer.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        directBuffer.put(inputBuffer)
        directBuffer.flip()

        val inputTensor = OnnxTensor.createTensor(
            env,
            directBuffer,
            longArrayOf(1, INPUT_CHANNELS.toLong(), INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )

        Log.d(TAG, "preprocess: running model inference (input=${inputName})")
        val output: OrtSession.Result = try {
            sess.run(mapOf(inputName to inputTensor))
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            inputTensor.close()
            rgbPadded.release()
            return matBundle.getMorph()
        } finally {
            inputTensor.close()
        }
        Log.d(TAG, "preprocess: inference complete")

        output.use {
            val outputValue = output.get(0).value
            if (outputValue is OnnxTensor) {
                val floatBuf = outputValue.floatBuffer
                floatBuf.rewind()
                floatBuf.get(outputBuffer)
            } else if (outputValue is Array<out Any?>) {
                @Suppress("UNCHECKED_CAST")
                val arr = outputValue as Array<Array<Array<FloatArray>>>
                var flatIdx = 0
                for (c in arr) {
                    for (z in c) {
                        for (row in z) {
                            for (v in row) {
                                if (flatIdx < outputBuffer.size) {
                                    outputBuffer[flatIdx++] = v
                                }
                            }
                        }
                    }
                }
            } else {
                Log.e(TAG, "Unexpected output type: ${outputValue.javaClass.name}")
                rgbPadded.release()
                return matBundle.getMorph()
            }

            var outMin = Float.MAX_VALUE; var outMax = Float.MIN_VALUE
            var outMean = 0f
            for (i in outputBuffer.indices) {
                outMean += outputBuffer[i]
                if (outputBuffer[i] < outMin) outMin = outputBuffer[i]
                if (outputBuffer[i] > outMax) outMax = outputBuffer[i]
            }
            outMean /= outputBuffer.size
            Log.d(TAG, "preprocess: output tensor stats: min=${"%.4f".format(outMin)}, max=${"%.4f".format(outMax)}, mean=${"%.4f".format(outMean)}, shape=[1,${OUTPUT_CHANNELS},${INPUT_SIZE},${INPUT_SIZE}]")

            var below0 = 0; var below05 = 0; var above05 = 0; var above08 = 0
            for (v in outputBuffer) {
                if (v < 0f) below0++
                if (v < 0.5f) below05++
                if (v > 0.5f) above05++
                if (v > 0.8f) above08++
            }
            Log.d(TAG, "preprocess: output distribution: <0=$below0, <0.5=$below05, >0.5=$above05, >0.8=$above08, total=${outputBuffer.size}")
        }

        val fullMask = Mat(INPUT_SIZE, INPUT_SIZE, CvType.CV_8UC1)
        var maskNonzero = 0

        if (isDeepLabV3) {
            for (y in 0 until INPUT_SIZE) {
                for (x in 0 until INPUT_SIZE) {
                    val idx = y * INPUT_SIZE + x
                    val bgLogit = outputBuffer[idx]
                    val fgLogit = outputBuffer[INPUT_SIZE * INPUT_SIZE + idx]
                    val documentProb = 1f / (1f + kotlin.math.exp(-(fgLogit - bgLogit)))
                    val value = if (documentProb > 0.5f) 255 else 0
                    fullMask.put(y, x, value.toDouble())
                    if (value > 0) maskNonzero++
                }
            }
        } else {
            val totalPixels = INPUT_SIZE * INPUT_SIZE
            for (i in 0 until totalPixels) {
                val value = if (outputBuffer[i] > 0.5f) 255 else 0
                fullMask.put(i / INPUT_SIZE, i % INPUT_SIZE, value.toDouble())
                if (value > 0) maskNonzero++
            }
        }
        Log.d(TAG, "preprocess: fullMask nonzero=$maskNonzero / ${fullMask.total()} (${"%.1f".format(100.0 * maskNonzero / fullMask.total())}%) at threshold=0.5")

        // Crop mask to the unpadded region, then resize to target dimensions
        val croppedMask = fullMask.submat(0, resizedH, 0, resizedW).clone()
        fullMask.release()
        rgbPadded.release()

        val morph = matBundle.getMorph()
        Imgproc.resize(croppedMask, morph, Size(scaledWidth.toDouble(), scaledHeight.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
        croppedMask.release()

        val nonzeroCount = Core.countNonZero(morph)
        val morphTotal = morph.total()
        Log.d(TAG, "preprocess: resized mask nonzero=$nonzeroCount / $morphTotal (${"%.1f".format(100.0 * nonzeroCount / morphTotal)}%) -> ${scaledWidth}x${scaledHeight}")

        _detectionParams.value = _detectionParams.value.copy(
            brightness = "ONNX",
            cannyHigh = "N/A",
            cannyLow = "N/A"
        )

        Log.d(TAG, "preprocess done: mask nonzero=$nonzeroCount / $morphTotal, scaled to ${scaledWidth}x${scaledHeight}")
        return morph
    }

    override fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        params: PipelineParams
    ): MatOfPoint? {
        Log.d(TAG, "=== detectQuad START: morph=${morphImage.rows()}x${morphImage.cols()}, scaled=${scaledWidth}x${scaledHeight} ===")

        Log.d(TAG, "  morphImage: type=${morphImage.type()}, channels=${morphImage.channels()}, total=${morphImage.total()}, depth=${morphImage.depth()}")
        val morphNonZero = Core.countNonZero(morphImage)
        Log.d(TAG, "  morphImage nonzero pixels: $morphNonZero / ${morphImage.total()}")

        var hasNonZero = false
        val sampleN = kotlin.math.min(5000, morphImage.total().toInt())
        for (i in 0 until sampleN) {
            val r = i / morphImage.cols()
            val c = i % morphImage.cols()
            if (r < morphImage.rows() && c < morphImage.cols()) {
                val v = morphImage.get(r, c)
                if (v != null && v[0] > 0.0) {
                    hasNonZero = true
                    break
                }
            }
        }
        Log.d(TAG, "  morphImage hasNonZero (sampled): $hasNonZero")

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = matBundle.getHierarchy()
        Log.d(TAG, "  findContours: calling with RETR_LIST + CHAIN_APPROX_SIMPLE, hierarchy=${hierarchy.rows()}x${hierarchy.cols()}, type=${hierarchy.type()}")
        Imgproc.findContours(
            morphImage,
            contours,
            hierarchy,
            Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        Log.d(TAG, "  findContours: found ${contours.size} contours")

        if (contours.isEmpty() && morphImage.total() > 0) {
            Log.d(TAG, "  [DEBUG] morphImage pixel samples (top-left 10x10):")
            for (r in 0 until kotlin.math.min(10, morphImage.rows())) {
                val rowStr = mutableListOf<String>()
                for (c in 0 until kotlin.math.min(10, morphImage.cols())) {
                    val v = morphImage.get(r, c)
                    rowStr.add(if (v != null) "${"%.0f".format(v[0])}" else "?")
                }
                Log.d(TAG, "    row $r: ${rowStr.joinToString(" ")}")
            }
        }

        val frameArea = scaledWidth * scaledHeight
        val minArea = frameArea * params.minAreaFraction
        Log.d(TAG, "  frameArea=$frameArea, minArea=$minArea, minAreaFraction=${params.minAreaFraction}")

        val areaStats = contours.map { abs(Geometry.contourArea(it)) }.sortedDescending().take(5)
        Log.d(TAG, "  Top contour areas: ${areaStats.joinToString(", ") { "%.0f".format(it) }}")

        val candidates = mutableListOf<MatOfPoint>()
        val approx = matBundle.getApprox()
        var skippedArea = 0
        var skippedPoints = 0
        var skippedNot4 = 0
        var skippedNotRect = 0
        var skippedSolidity = 0

        for (contour in contours) {
            val area = abs(Geometry.contourArea(contour))
            if (area < minArea) {
                skippedArea++
                continue
            }

            if (contour.total() < 10) {
                skippedPoints++
                continue
            }

            val hull = matBundle.getHull()
            Geometry.convexHull(contour, hull)
            val hullCount = hull.rows() * hull.cols()
            val hullData = IntArray(hullCount * hull.channels())
            hull.get(0, 0, hullData)
            val hullPoints = matBundle.getHullPoints()
            hullPoints.release()
            hullPoints.create(0, 1, CvType.CV_32FC2)
            val hullPointList = mutableListOf<Point>()
            for (i in hullData.indices step 2) {
                if (i + 1 < hullData.size) {
                    hullPointList.add(Point(hullData[i].toDouble(), hullData[i + 1].toDouble()))
                }
            }
            hullPoints.fromList(hullPointList)

            val hullPtCount = hullPointList.size
            val peri = Geometry.arcLength(hullPoints, true)
            var foundQuad = false
            val epsilons = listOf(0.015, 0.025, 0.04, 0.06, 0.10)

            for (tol in epsilons) {
                val epsilon = tol * peri
                Geometry.approxPolyDP(hullPoints, approx, epsilon, true)
                if (approx.total() == 4L) {
                    Log.d(TAG, "    approxPolyDP: tol=$tol, peri=${"%.0f".format(peri)}, hullPts=$hullPtCount -> QUAD at epsilon=${"%.2f".format(epsilon)}")
                    foundQuad = true
                    break
                }
            }

            if (!foundQuad) {
                skippedNot4++
                continue
            }

            if (!isRectangle(approx)) {
                skippedNotRect++
                continue
            }

            val rect = Geometry.boundingRect(approx)
            val rectArea = rect.width * rect.height
            val solidity = area / rectArea.toDouble()
            if (solidity < 0.3) {
                skippedSolidity++
                continue
            }

            val scaleX = originalWidth.toDouble() / scaledWidth
            val scaleY = originalHeight.toDouble() / scaledHeight
            val scaledPoints = approx.toArray().map { Point(it.x * scaleX, it.y * scaleY) }
            val quad = MatOfPoint(*scaledPoints.toTypedArray())

            Log.d(TAG, "  CANDIDATE: area=$area, solidity=${"%.2f".format(solidity)}, rect=${rect.width}x${rect.height}")
            candidates.add(quad)
        }

        Log.d(TAG, "  detectQuad summary: contours=${contours.size}, candidates=${candidates.size}, skippedArea=$skippedArea, skippedPoints=$skippedPoints, skippedNot4=$skippedNot4, skippedNotRect=$skippedNotRect, skippedSolidity=$skippedSolidity")

        val best = candidates.maxByOrNull { abs(Geometry.contourArea(it)) }
        val result = best?.let { MatOfPoint(*it.toArray()) }
        candidates.forEach { it.release() }
        Log.d(TAG, "  detectQuad END: result=${if (result != null) "found (${result.total()} pts)" else "null"}")
        return result
    }

    override fun validateQuadSize(
        quad: MatOfPoint,
        originalWidth: Int,
        originalHeight: Int
    ): Boolean {
        val rect = Geometry.boundingRect(quad)
        val quadArea = rect.width * rect.height
        val frameArea = originalWidth * originalHeight
        return quadArea <= frameArea * 0.95
    }

    fun release() {
        session?.close()
        session = null
        modelLoaded = false
        try {
            env.close()
        } catch (_: Exception) {
            // Ignore cleanup errors
        }
    }

    companion object {
        const val INPUT_SIZE = 384
        const val INPUT_CHANNELS = 3
        const val OUTPUT_CHANNELS = 2

        private val IMAGE_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGE_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        private const val TAG = "OnnxDetector"

        fun isRectangle(approx: MatOfPoint2f): Boolean {
            val pts = approx.toArray()
            var maxDeviation = 0.0
            for (i in 0..3) {
                val angle = computeAngle(
                    pts[(i + 1) % 4],
                    pts[(i + 3) % 4],
                    pts[i]
                )
                maxDeviation = max(maxDeviation, abs(90.0 - angle))
            }
            return maxDeviation < 25
        }

        fun computeAngle(p1: Point, p2: Point, center: Point): Double {
            val dx1 = p1.x - center.x
            val dy1 = p1.y - center.y
            val dx2 = p2.x - center.x
            val dy2 = p2.y - center.y
            val dot = dx1 * dx2 + dy1 * dy2
            val norm1 = sqrt(dx1 * dx1 + dy1 * dy1)
            val norm2 = sqrt(dx2 * dx2 + dy2 * dy2)
            return acos(dot / (norm1 * norm2)) * 180.0 / Math.PI
        }
    }
}
