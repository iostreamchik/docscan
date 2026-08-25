package io.github.iostreamchik.scanner.data.detector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import io.github.iostreamchik.scanner.entity.IntermediateBitmaps
import io.github.iostreamchik.scanner.entity.DetectionParameters
import io.github.iostreamchik.scanner.entity.PipelineParams
import io.github.iostreamchik.scanner.data.utils.validateQuadRectangularity
import io.github.iostreamchik.scanner.data.utils.isRectangle
import io.github.iostreamchik.scanner.data.opencv.IMatBundle
import io.github.iostreamchik.scanner.data.utils.fixRotation
import io.github.iostreamchik.scanner.data.utils.scoreContourWithParams
import io.github.iostreamchik.scanner.data.utils.sortQuadPoints
import io.github.iostreamchik.scanner.data.utils.toBitmap
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class SegmentationDetector(
    context: Context,
    env: OrtEnvironment,
    private val matBundle: IMatBundle,
    modelPath: String = "onnx/deeplabv3_mbv3_docseg.onnx",
    private val useCustomNormalization: Boolean = true
) : IDocumentDetector {

    private val sessionManager = OnnxSessionManager(context, env, modelPath)

    var maskThreshold: Float = 0.5f
        set(value) {
            field = value.coerceIn(0.1f, 0.7f)
            _detectionParams.value = _detectionParams.value.copy(
                maskThreshold = "%.2f".format(field)
            )
        }

    private val _detectionParams = MutableStateFlow(
        DetectionParameters(detectorName = AsyncDetectorSource.SEGMENTATION.detectionParamsName)
    )
    override val detectionParams = _detectionParams.asStateFlow()

    override suspend fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat {
        sessionManager.init(TAG)

        val raw = matBundle.getRawMat()
        rawMat.copyTo(raw)

        val sess = sessionManager.getSession() ?: return matBundle.getMorph()
        val inputName = sess.inputNames.firstOrNull() ?: return matBundle.getMorph()

        val scale = INPUT_SIZE.toDouble() / max(rawMat.cols(), rawMat.rows())
        val resizedW = (rawMat.cols() * scale).toInt()
        val resizedH = (rawMat.rows() * scale).toInt()

        val resized = Mat()
        Imgproc.resize(rawMat, resized, Size(resizedW.toDouble(), resizedH.toDouble()))

        val rgb = Mat()
        Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_RGBA2RGB)
        resized.release()

        val rgbPadded = Mat(INPUT_SIZE, INPUT_SIZE, CvType.CV_8UC3)
        rgbPadded.setTo(Scalar(128.0, 128.0, 128.0))
        val roi = rgbPadded.submat(0, rgb.rows(), 0, rgb.cols())
        rgb.copyTo(roi)
        roi.release()
        rgb.release()

        Core.normalize(rgbPadded, rgbPadded, 0.0, 1.0, Core.NORM_MINMAX, CvType.CV_32FC3)

        val meanR = if (useCustomNormalization) IMAGE_MEAN[0].toDouble() else 0.0
        val meanG = if (useCustomNormalization) IMAGE_MEAN[1].toDouble() else 0.0
        val meanB = if (useCustomNormalization) IMAGE_MEAN[2].toDouble() else 0.0
        val stdR = if (useCustomNormalization) 1.0 / IMAGE_STD[0] else 1.0
        val stdG = if (useCustomNormalization) 1.0 / IMAGE_STD[1] else 1.0
        val stdB = if (useCustomNormalization) 1.0 / IMAGE_STD[2] else 1.0

        val mean = Scalar(meanR, meanG, meanB)
        val std = Scalar(stdR, stdG, stdB)
        Core.subtract(rgbPadded, mean, rgbPadded)
        Core.multiply(rgbPadded, std, rgbPadded)

        val inputTensor = sessionManager.prepareInputTensor(rgbPadded, INPUT_SIZE)

        val output: OrtSession.Result
        try {
            output = sess.run(mapOf(inputName to inputTensor))
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            inputTensor.close()
            rgbPadded.release()
            return matBundle.getMorph()
        }

        val outputTensor = output.get(0) as OnnxTensor
        val shape = outputTensor.getInfo().shape
        val isNchw = shape[0] == 2L || shape.size == 4

        val bgMat = Mat(INPUT_SIZE, INPUT_SIZE, CvType.CV_32FC1)
        val fgMat = Mat(INPUT_SIZE, INPUT_SIZE, CvType.CV_32FC1)

        val totalSize = INPUT_SIZE * INPUT_SIZE
        val flatData = FloatArray(totalSize * 2)
        outputTensor.floatBuffer.get(flatData)
        if (isNchw) {
            bgMat.put(0, 0, flatData.copyOfRange(0, totalSize))
            fgMat.put(0, 0, flatData.copyOfRange(totalSize, totalSize * 2))
        } else {
            val bgChan = FloatArray(totalSize)
            val fgChan = FloatArray(totalSize)
            for (i in 0 until totalSize) {
                bgChan[i] = flatData[i * 2]
                fgChan[i] = flatData[i * 2 + 1]
            }
            bgMat.put(0, 0, bgChan)
            fgMat.put(0, 0, fgChan)
        }
        outputTensor.close()
        output.close()
        inputTensor.close()

        val logitDiff = Mat(INPUT_SIZE, INPUT_SIZE, CvType.CV_32FC1)
        Core.subtract(fgMat, bgMat, logitDiff)
        bgMat.release()
        fgMat.release()

        val negLogit = Mat(INPUT_SIZE, INPUT_SIZE, CvType.CV_32FC1)
        Core.multiply(logitDiff, Scalar(-1.0), negLogit)
        logitDiff.release()

        Core.exp(negLogit, negLogit)
        Core.add(negLogit, Scalar(1.0), negLogit)

        val ones = Mat.ones(INPUT_SIZE, INPUT_SIZE, CvType.CV_32FC1)
        val probMap = Mat(INPUT_SIZE, INPUT_SIZE, CvType.CV_32FC1)
        Core.divide(ones, negLogit, probMap)
        ones.release()
        negLogit.release()

        if (resizedH < INPUT_SIZE) probMap.submat(resizedH, INPUT_SIZE, 0, INPUT_SIZE)
            .setTo(Scalar(0.0))
        if (resizedW < INPUT_SIZE) probMap.submat(0, resizedH, resizedW, INPUT_SIZE)
            .setTo(Scalar(0.0))

        Imgproc.GaussianBlur(probMap, probMap, Size(5.0, 5.0), 1.5)

        val prob8 = Mat(INPUT_SIZE, INPUT_SIZE, CvType.CV_8UC1)
        probMap.convertTo(prob8, CvType.CV_8UC1, 255.0)
        probMap.release()

        val contentProb = prob8.submat(0, resizedH, 0, resizedW)
        val otsuMask = Mat(resizedH, resizedW, CvType.CV_8UC1)
        val otsuVal = Imgproc.threshold(
            contentProb,
            otsuMask,
            0.0,
            255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )

        val fullMask = if (Core.countNonZero(otsuMask) > INPUT_SIZE * INPUT_SIZE * 0.003) {
            val m = Mat.zeros(INPUT_SIZE, INPUT_SIZE, CvType.CV_8UC1)
            otsuMask.copyTo(m.submat(0, resizedH, 0, resizedW))
            otsuMask.release()
            m
        } else {
            otsuMask.release()
            val m = Mat(INPUT_SIZE, INPUT_SIZE, CvType.CV_8UC1)
            Imgproc.threshold(prob8, m, maskThreshold * 255.0, 255.0, Imgproc.THRESH_BINARY)
            m
        }
        prob8.release()

        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val nLabels = Imgproc.connectedComponentsWithStats(
            fullMask,
            labels,
            stats,
            centroids,
            8,
            CvType.CV_32S
        )

        var docArea = 0
        var largestLabel = 0
        var largestArea = 0
        for (label in 1 until nLabels) {
            val area = stats.get(label, Imgproc.CC_STAT_AREA)[0].toInt()
            if (area > docArea) docArea = area
            if (area > largestArea) {
                largestArea = area
                largestLabel = label
            }
        }

        val areaScale = (scaledWidth * scaledHeight).toDouble() / (INPUT_SIZE * INPUT_SIZE)
        val docAreaScaled = (docArea * areaScale).toInt()
        val docLinear = sqrt(docAreaScaled.toDouble()).coerceIn(50.0, 800.0)
        val kernelCloseK = ((docLinear / 6.0).toInt().coerceIn(5, 41)).takeIf { it % 2 == 1 }
            ?: ((docLinear / 6.0).toInt().coerceIn(5, 41) - 1)
        val kernelOpenK = ((docLinear / 12.0).toInt().coerceIn(3, 21)).takeIf { it % 2 == 1 }
            ?: ((docLinear / 12.0).toInt().coerceIn(3, 21) - 1)

        val kernelClose = Mat(kernelCloseK, kernelCloseK, CvType.CV_8UC1, Scalar.all(1.0))
        val kernelOpen = Mat(kernelOpenK, kernelOpenK, CvType.CV_8UC1, Scalar.all(1.0))

        Imgproc.morphologyEx(fullMask, fullMask, Imgproc.MORPH_CLOSE, kernelClose)
        Imgproc.morphologyEx(fullMask, fullMask, Imgproc.MORPH_OPEN, kernelOpen)
        Imgproc.morphologyEx(fullMask, fullMask, Imgproc.MORPH_CLOSE, kernelOpen)
        kernelClose.release()
        kernelOpen.release()

        val minBlobArea = (INPUT_SIZE * INPUT_SIZE * 0.0005).toInt()
        val cleanedMask = Mat.zeros(fullMask.size(), CvType.CV_8UC1)
        if (largestArea >= minBlobArea) {
            val compMask = Mat()
            Core.compare(labels, Scalar(largestLabel.toDouble()), compMask, Core.CMP_EQ)
            compMask.copyTo(cleanedMask)
            compMask.release()
        }
        labels.release()
        stats.release()
        centroids.release()
        fullMask.release()

        val croppedMask = cleanedMask.submat(0, resizedH, 0, resizedW).clone()
        cleanedMask.release()
        rgbPadded.release()

        val morph = matBundle.getMorph()
        Imgproc.resize(
            croppedMask, morph,
            Size(scaledWidth.toDouble(), scaledHeight.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST
        )
        croppedMask.release()

        morph.copyTo(matBundle.getSegmentationMask())

        _detectionParams.value = _detectionParams.value.copy(
            brightness = "N/A",
            cannyHigh = "N/A",
            cannyLow = "N/A"
        )

        return morph
    }

    private fun computeMaskConfidence(mask: Mat): Float {
        val totalPixels = mask.rows() * mask.cols()
        val foregroundPixels = Core.countNonZero(mask)
        return foregroundPixels.toFloat() / totalPixels
    }

    private fun isQuadRectangular(pts: Array<Point>): Boolean =
        validateQuadRectangularity(pts.toList(), 15.0)

    private fun getAspectRatio(pts: Array<Point>): Double {
        val xs = pts.map { it.x }
        val ys = pts.map { it.y }
        val width = xs.maxOrNull()!! - xs.minOrNull()!!
        val height = ys.maxOrNull()!! - ys.minOrNull()!!
        return min(width, height) / max(width, height)
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
        val workingMask = matBundle.getSegmentationMask()
        if (workingMask.empty()) return null

        if (computeMaskConfidence(workingMask) < 0.03) return null

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = matBundle.getHierarchy()
        Imgproc.findContours(
            workingMask,
            contours,
            hierarchy,
            Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        val frameArea = scaledWidth * scaledHeight
        val minArea = frameArea * params.minAreaFraction

        val candidates = mutableListOf<MatOfPoint>()
        val approx = matBundle.getApprox()
        val hullPoints = matBundle.getHullPoints()
        hullPoints.release()
        hullPoints.create(0, 1, CvType.CV_32FC2)

        for (contour in contours) {
            val area = abs(Geometry.contourArea(contour))
            if (area < minArea) continue
            if (contour.total() < 10) continue

            val hull = matBundle.getHull()
            Geometry.convexHull(contour, hull)
            val hullCount = hull.rows() * hull.cols()
            val hullData = IntArray(hullCount * hull.channels())
            hull.get(0, 0, hullData)
            val hullPointList = mutableListOf<Point>()
            for (i in hullData.indices step 2) {
                if (i + 1 < hullData.size) {
                    hullPointList.add(Point(hullData[i].toDouble(), hullData[i + 1].toDouble()))
                }
            }
            hullPoints.fromList(hullPointList)

            val peri = Geometry.arcLength(hullPoints, true)
            var foundQuad = false

            val epsilon = 0.02 * peri
            Geometry.approxPolyDP(hullPoints, approx, epsilon, true)
            if (approx.total() == 4L && isRectangle(approx)) {
                foundQuad = true
            }

            if (!foundQuad) {
                val pts = contour.toArray()
                var tlIdx = 0
                var brIdx = 0
                var trIdx = 0
                var blIdx = 0
                var tlSum = Long.MAX_VALUE
                var brSum = Long.MIN_VALUE
                var trDiff = Long.MAX_VALUE
                var blDiff = Long.MIN_VALUE

                for (i in pts.indices) {
                    val x = pts[i].x.toLong()
                    val y = pts[i].y.toLong()
                    val sum = x + y
                    val diff = y - x
                    if (sum < tlSum) {
                        tlSum = sum; tlIdx = i
                    }
                    if (sum > brSum) {
                        brSum = sum; brIdx = i
                    }
                    if (diff < trDiff) {
                        trDiff = diff; trIdx = i
                    }
                    if (diff > blDiff) {
                        blDiff = diff; blIdx = i
                    }
                }

                val fallbackPts = arrayOf(pts[tlIdx], pts[trIdx], pts[brIdx], pts[blIdx])
                val fallbackApprox = matBundle.getApprox()
                fallbackApprox.fromArray(*fallbackPts)
                val rectCheck = Geometry.boundingRect(fallbackApprox)
                val rectAreaCheck = rectCheck.width * rectCheck.height
                val solidityCheck = area / rectAreaCheck.toDouble()

                if (solidityCheck >= 0.5 && isQuadRectangular(fallbackPts) && getAspectRatio(fallbackPts) >= 0.35) {
                    val scaleX = originalWidth.toDouble() / scaledWidth
                    val scaleY = originalHeight.toDouble() / scaledHeight
                    val scaledFallback = fallbackPts.map { Point(it.x * scaleX, it.y * scaleY) }
                    val quadFallback = MatOfPoint(*scaledFallback.toTypedArray())
                    candidates.add(quadFallback)
                }
                continue
            }

            if (!isRectangle(approx)) continue

            val rect = Geometry.boundingRect(approx)
            val rectArea = rect.width * rect.height
            val solidity = area / rectArea.toDouble()
            if (solidity < 0.55) continue

            val scaleX = originalWidth.toDouble() / scaledWidth
            val scaleY = originalHeight.toDouble() / scaledHeight
            val scaledPoints = approx.toArray().map { Point(it.x * scaleX, it.y * scaleY) }
            val quad = MatOfPoint(*scaledPoints.toTypedArray())
            candidates.add(quad)
        }

        val best = candidates.maxByOrNull {
            scoreContourWithParams(it, originalWidth, originalHeight, params)
        }
        val result = best?.let { quad ->
            val sorted = sortQuadPoints(quad.toArray().toList())
            if (sorted.size == 4) MatOfPoint(*sorted.toTypedArray()) else quad
        }
        candidates.forEach { it.release() }
        return result
    }

    override fun captureIntermediateSnapshots(
        rotation: Int
    ): IntermediateBitmaps {
        val maskMat = matBundle.getSegmentationMask()
        val rawMat = matBundle.getRawMat()
        return if (!maskMat.empty() && !rawMat.empty()) {
            IntermediateBitmaps(
                mask = buildMaskOverlay(rawMat, maskMat, rotation)
            )
        } else {
            IntermediateBitmaps()
        }
    }

    override fun capturePostDetectionSnapshots(
        rotation: Int
    ): IntermediateBitmaps {
        val maskMat = matBundle.getSegmentationMask()
        val rawMat = matBundle.getRawMat()
        return if (!maskMat.empty() && !rawMat.empty()) {
            IntermediateBitmaps(
                mask = buildMaskOverlay(rawMat, maskMat, rotation)
            )
        } else {
            IntermediateBitmaps()
        }
    }

    private fun buildMaskOverlay(
        rawMat: Mat,
        maskMat: Mat,
        rotation: Int
    ): Bitmap {
        val rotatedRawMat = rawMat.fixRotation(rotation)
        val rotatedRaw = rotatedRawMat.toBitmap()
        rotatedRawMat.release()
        val width = rotatedRaw.width
        val height = rotatedRaw.height

        val rotatedMask = maskMat.fixRotation(rotation)
        val resizedMask = Mat()
        Imgproc.resize(
            rotatedMask, resizedMask,
            Size(width.toDouble(), height.toDouble()),
            0.0, 0.0, Imgproc.INTER_NEAREST
        )

        val overlay = rotatedRaw.copy(Bitmap.Config.ARGB_8888, true)
        val pixels = IntArray(width * height)
        overlay.getPixels(pixels, 0, width, 0, 0, width, height)

        val maskData = ByteArray(width * height)
        resizedMask.get(0, 0, maskData)

        for (i in pixels.indices) {
            val maskAlpha = maskData[i].toInt() and 0xFF
            if (maskAlpha < 128) {
                val r = Color.red(pixels[i])
                val g = Color.green(pixels[i])
                val b = Color.blue(pixels[i])
                pixels[i] = Color.argb(255, (r * 0.3f).toInt(), (g * 0.3f).toInt(), (b * 0.3f).toInt())
            }
        }

        overlay.setPixels(pixels, 0, width, 0, 0, width, height)

        resizedMask.release()
        rotatedMask.release()

        return overlay
    }

    override fun release() {
        sessionManager.close(TAG)
        matBundle.releaseAll()
    }

    companion object {
        const val INPUT_SIZE = 384

        private val IMAGE_MEAN = floatArrayOf(0.4611f, 0.4359f, 0.3905f)
        private val IMAGE_STD = floatArrayOf(0.2193f, 0.2150f, 0.2109f)

        private const val TAG = "OnnxDetector"
    }
}