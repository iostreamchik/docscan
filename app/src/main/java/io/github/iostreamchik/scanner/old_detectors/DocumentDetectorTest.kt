package io.github.iostreamchik.scanner.old_detectors

import android.util.Log
import io.github.iostreamchik.scanner.detector.IDocumentDetector
import io.github.iostreamchik.scanner.opencv.IMatBundle
import io.github.iostreamchik.scanner.opencv.PipelineParams
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.sqrt

class DocumentDetectorTest(
    private val matBundle: IMatBundle
) : IDocumentDetector {

    private val _detectionParams = MutableStateFlow(DetectionParameters())
    override val detectionParams = _detectionParams.asStateFlow()

    override fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat {
        val smallMat = Mat()
        Imgproc.resize(rawMat, smallMat, Size(scaledWidth.toDouble(), scaledHeight.toDouble()))
        Imgproc.cvtColor(smallMat, matBundle.getGray(), Imgproc.COLOR_RGBA2GRAY)
        smallMat.release()

        Core.meanStdDev(matBundle.getGray(), matBundle.getMean(), matBundle.getStd())
        val avgBrightness = matBundle.getMean().toArray()[0]
        _detectionParams.value = _detectionParams.value.copy(
            brightness = "%.1f".format(avgBrightness)
        )

        val blurKsize = params.medianBlurKsize.coerceAtLeast(3)
        Imgproc.medianBlur(matBundle.getGray(), matBundle.getBlurred(), blurKsize)

        val brightness = avgBrightness.coerceIn(20.0, 200.0)
        val dimBoost = if (brightness < 80.0) {
            100.0 / (brightness + 10.0)
        } else {
            0.0
        }
        val brightBoost = if (brightness > 130.0) {
            (brightness - 130.0) / 30.0
        } else {
            0.0
        }
        val claheClipLimit = (0.3 + dimBoost + brightBoost).coerceIn(1.0, 3.0)
        _detectionParams.value = _detectionParams.value.copy(
            claheClipLimit = claheClipLimit.toString()
        )
        val tileSize = params.claheTileSize.coerceAtLeast(8).toDouble()
        val clahe = Imgproc.createCLAHE(claheClipLimit, Size(tileSize, tileSize))
        clahe.apply(matBundle.getBlurred(), matBundle.getEnhanced())

        val morphCloseKsize = params.morphCloseSize.coerceAtLeast(3).toDouble()
        val kernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT,
            Size(morphCloseKsize, morphCloseKsize)
        )
        Imgproc.morphologyEx(matBundle.getEnhanced(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, kernel)
        kernel.release()

        Imgproc.GaussianBlur(matBundle.getMorph(), matBundle.getTemp(), Size(3.0, 3.0), 0.8)

        val otsu = Imgproc.threshold(
            matBundle.getTemp(),
            matBundle.getEnhanced(),
            0.0,
            255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )

        val ratio = 0.3
        val high = otsu//.coerceIn(10.0, 80.0)
        val low = (high * ratio)//.coerceIn(5.0, 40.0)

        _detectionParams.value = _detectionParams.value.copy(
            cannyHigh = high.toInt().toString(),
            cannyLow = low.toInt().toString()
        )

        Imgproc.Canny(matBundle.getTemp(), matBundle.getEdges(), low, high)

        var closeKsize = params.strongCloseSize.coerceIn(3, 5)
        if (closeKsize % 2 == 0) closeKsize++
        val kernel2 = Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT,
            Size(closeKsize.toDouble(), closeKsize.toDouble())
        )
        Imgproc.morphologyEx(matBundle.getEdges(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, kernel2)
        kernel2.release()

        val dirKsize = params.directionalKernelSize.coerceIn(3, 5)
        val hKernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT,
            Size(dirKsize.toDouble(), 1.0)
        )
        Imgproc.morphologyEx(
            matBundle.getMorph(),
            matBundle.getHorizontalClose(),
            Imgproc.MORPH_CLOSE,
            hKernel
        )
        hKernel.release()

        val vKernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT,
            Size(1.0, dirKsize.toDouble())
        )
        Imgproc.morphologyEx(
            matBundle.getHorizontalClose(),
            matBundle.getVerticalClose(),
            Imgproc.MORPH_CLOSE,
            vKernel
        )
        vKernel.release()

        matBundle.getVerticalClose().copyTo(matBundle.getMorph())

        val nonzeroCount = Core.countNonZero(matBundle.getMorph())
        Log.d("DocScanTest", "brightness: ${"%.1f".format(avgBrightness)}, clahe: ${"%.2f".format(claheClipLimit)}, otsu: [$low-$high], nonzero: $nonzeroCount / ${matBundle.getMorph().total()}")

        return matBundle.getMorph()
    }

    override fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        params: PipelineParams
    ): MatOfPoint? {
        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(
            morphImage,
            contours,
            matBundle.getHierarchy(),
            Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        val frameArea = scaledWidth * scaledHeight
        val minArea = frameArea * params.minAreaFraction
        val candidates = mutableListOf<Pair<MatOfPoint, Double>>()
        var skippedArea = 0
        var skippedPoints = 0
        var skippedNot4 = 0
        var skippedNotRect = 0
        var skippedSolidity = 0

        val areaStats = contours.map { abs(Geometry.contourArea(it)) }.sortedDescending().take(5)
        Log.d("DocScanTest", "contours: ${contours.size}, minArea: $minArea, topAreas: ${areaStats.joinToString(", ") { "%.0f".format(it) }}")

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

            val pts2f = MatOfPoint2f(*contour.toArray().map { Point(it.x, it.y) }.toTypedArray())
            val peri = Geometry.arcLength(pts2f, true)
            val approx = MatOfPoint2f()
            val epsilons = listOf(0.015, 0.025, 0.04, 0.06, 0.10)
            var foundQuad = false

            for (tol in epsilons) {
                Geometry.approxPolyDP(pts2f, approx, tol * peri, true)
                if (approx.total() == 4L) {
                    foundQuad = true
                    break
                }
            }

            if (!foundQuad) {
                skippedNot4++
                pts2f.release()
                approx.release()
                continue
            }

            if (!isRectangle(approx)) {
                skippedNotRect++
                pts2f.release()
                approx.release()
                continue
            }

            val scaleX = originalWidth.toDouble() / scaledWidth
            val scaleY = originalHeight.toDouble() / scaledHeight
            val scaledPoints = approx.toArray().map { Point(it.x * scaleX, it.y * scaleY) }
            val quad = MatOfPoint(*scaledPoints.toTypedArray())

            val rect = Geometry.boundingRect(quad)
            val scaledArea = area * (scaleX * scaleY)
            val solidity = scaledArea / (rect.width * rect.height).toDouble()

            if (solidity < 0.3) {
                quad.release()
                skippedSolidity++
                pts2f.release()
                approx.release()
                continue
            }

            val score = (area * params.scoreAreaWeight) +
                    (1.0 - abs((area / frameArea) - 0.5) * 2.0) * params.scoreCenterWeight +
                    solidity * params.scoreAreaRatioWeight

            Log.d("DocScanTest", "candidate: area=${"%.0f".format(area)}, solidity=${"%.2f".format(solidity)}, score=${"%.3f".format(score)}")
            candidates.add(Pair(quad, score))

            pts2f.release()
            approx.release()
        }

        Log.d("DocScanTest", "summary: candidates=${candidates.size}, skippedArea=$skippedArea, skippedPts=$skippedPoints, skippedNot4=$skippedNot4, skippedRect=$skippedNotRect, skippedSolidity=$skippedSolidity")

        val best = candidates.maxByOrNull { it.second }
        val result = best?.let { MatOfPoint(*it.first.toArray()) }
        candidates.forEach { if (it !== best) it.first.release() }
        best?.first?.release()

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

    companion object {
        fun isRectangle(approx: MatOfPoint2f): Boolean {
            val pts = approx.toArray()
            var maxDeviation = 0.0

            for (i in 0..3) {
                val angle = computeAngle(
                    pts[(i + 1) % 4],
                    pts[(i + 3) % 4],
                    pts[i]
                )
                maxDeviation = max(maxDeviation, abs(90 - angle))
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