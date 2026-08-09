package io.github.iostreamchik.scanner.data.opencv

import io.github.iostreamchik.scanner.entity.PipelineParams
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import kotlin.math.abs

data class PreprocessingConfig(
    val dimBoostDivisor: Double = 40.0,
    val brightBoostDivisor: Double = 60.0,
    val brightnessFormat: String = "%.0f",
    val claheFormat: String = "%.1f",
    val cannyFormat: String = "%.0f",
)

typealias DetectionParamsCallback = (
    brightness: Double,
    claheClipLimit: Double,
    cannyHigh: Double,
    cannyLow: Double,
) -> Unit

typealias QuadSelector = (List<MatOfPoint>) -> MatOfPoint?

object OpenCVAdapter {

    fun resizeToGray(source: Mat, width: Int, height: Int, gray: Mat) {
        val smallMat = Mat()
        Imgproc.resize(source, smallMat, Size(width.toDouble(), height.toDouble()))
        Imgproc.cvtColor(smallMat, gray, Imgproc.COLOR_RGBA2GRAY)
        smallMat.release()
    }

    fun getAverageBrightness(image: Mat, bundle: IMatBundle): Double {
        Core.meanStdDev(image, bundle.getMean(), bundle.getStd())
        return bundle.getMean().toArray()[0]
    }

    fun getStdDev(image: Mat, bundle: IMatBundle): Double {
        Core.meanStdDev(image, bundle.getMean(), bundle.getStd())
        return bundle.getStd().toArray()[0]
    }

    fun applyClahe(source: Mat, dest: Mat, clipLimit: Double, tileSize: Double) {
        val clahe = Imgproc.createCLAHE(clipLimit, Size(tileSize, tileSize))
        clahe.apply(source, dest)
    }

    fun createRectKernel(size: Size, kernel: Mat) {
        Imgproc.getStructuringElement(Imgproc.MORPH_RECT, size).also { created ->
            kernel.release()
            created.copyTo(kernel)
            created.release()
        }
    }

    fun morphClose(source: Mat, dest: Mat, kernel: Mat) {
        Imgproc.morphologyEx(source, dest, Imgproc.MORPH_CLOSE, kernel)
    }

    fun findContours(image: Mat, hierarchy: Mat): List<MatOfPoint> {
        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(
            image,
            contours,
            hierarchy,
            Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        return contours
    }

    fun isRectangle(approx: MatOfPoint2f, toleranceDegrees: Double = 15.0): Boolean {
        return io.github.iostreamchik.scanner.data.utils.isRectangle(approx, toleranceDegrees)
    }

    fun preprocessClassical(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams,
        config: PreprocessingConfig,
        bundle: IMatBundle,
        onParams: DetectionParamsCallback
    ): Mat {
        resizeToGray(rawMat, scaledWidth, scaledHeight, bundle.getGray())

        val blurKsize = params.medianBlurKsize.coerceAtLeast(3)
        Imgproc.medianBlur(bundle.getGray(), bundle.getBlurred(), blurKsize)

        val avgBrightness = getAverageBrightness(bundle.getBlurred(), bundle)
        val brightness = avgBrightness.coerceIn(20.0, 200.0)
        val dimBoost = if (brightness < 80.0) {
            config.dimBoostDivisor / (brightness + 10.0)
        } else {
            0.0
        }
        val brightBoost = if (brightness > 130.0) {
            (brightness - 130.0) / config.brightBoostDivisor
        } else {
            0.0
        }
        val claheClipLimit = (0.5 + dimBoost + brightBoost).coerceIn(1.0, 1.5)

        val tileSize = params.claheTileSize.coerceAtLeast(8).toDouble()
        applyClahe(bundle.getBlurred(), bundle.getEnhanced(), claheClipLimit, tileSize)

        val enhancedContrast = getStdDev(bundle.getEnhanced(), bundle)
        val skipMorphClose = enhancedContrast < 25.0

        if (skipMorphClose) {
            bundle.getEnhanced().copyTo(bundle.getMorph())
        } else {
            val morphCloseKsize = params.morphCloseSize.coerceAtLeast(3).toDouble()
            createRectKernel(Size(morphCloseKsize, morphCloseKsize), bundle.getKernel())
            morphClose(bundle.getEnhanced(), bundle.getMorph(), bundle.getKernel())
        }

        val morphSource = if (skipMorphClose) bundle.getEnhanced() else bundle.getMorph()
        Imgproc.GaussianBlur(morphSource, bundle.getTemp(), Size(5.0, 5.0), 2.0)

        val otsu = Imgproc.threshold(
            bundle.getTemp(),
            bundle.getEdges(),
            0.0, 255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )

        val cannyHigh = otsu
        val cannyLow = cannyHigh * 0.2

        onParams(brightness, claheClipLimit, cannyHigh, cannyLow)

        Imgproc.Canny(bundle.getTemp(), bundle.getEdges(), cannyLow, cannyHigh)

        return bundle.getEdges()
    }

    fun findBestQuad(
        morphImage: Mat,
        bundle: IMatBundle,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        minAreaFraction: Double,
        approxEpsilon: Double,
        rectangleTolerance: Double = 15.0,
        selector: QuadSelector
    ): MatOfPoint? {
        val contours = findContours(morphImage, bundle.getHierarchy())

        val frameArea = scaledWidth * scaledHeight
        val minArea = frameArea * minAreaFraction
        val candidates = mutableListOf<MatOfPoint>()
        val approx = bundle.getApprox()

        for (contour in contours) {
            val area = abs(Geometry.contourArea(contour))
            if (area < minArea || contour.total() < 10) continue

            val pts2f = MatOfPoint2f(*contour.toArray().map { Point(it.x, it.y) }.toTypedArray())
            try {
                val peri = Geometry.arcLength(pts2f, true)
                Geometry.approxPolyDP(pts2f, approx, approxEpsilon * peri, true)

                if (approx.total() != 4L || !isRectangle(approx, rectangleTolerance)) continue

                val scaleX = originalWidth.toDouble() / scaledWidth
                val scaleY = originalHeight.toDouble() / scaledHeight
                val scaledPoints = approx.toArray().map { Point(it.x * scaleX, it.y * scaleY) }
                val quad = MatOfPoint(*scaledPoints.toTypedArray())

                val scaledArea = area * (scaleX * scaleY)
                val rect = Geometry.boundingRect(quad)
                val solidity = scaledArea / (rect.width * rect.height).toDouble()

                if (solidity < 0.5) {
                    quad.release()
                    continue
                }

                candidates.add(quad)
            } finally {
                pts2f.release()
            }
        }

        val best = selector(candidates)
        val result = best?.let { MatOfPoint(*it.toArray()) }
        candidates.forEach { it.release() }

        return result
    }
}
