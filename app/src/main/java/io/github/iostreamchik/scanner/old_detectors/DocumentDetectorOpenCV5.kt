package io.github.iostreamchik.scanner.old_detectors

import android.util.Log
import io.github.iostreamchik.scanner.detector.DetectionParameters
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

class DocumentDetectorOpenCV5(
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
        Log.d("DocScan5", "=== preprocess START: ${scaledWidth}x${scaledHeight} ===")

        val smallMat = Mat()
        Imgproc.resize(rawMat, smallMat, Size(scaledWidth.toDouble(), scaledHeight.toDouble()))
        Imgproc.cvtColor(smallMat, matBundle.getGray(), Imgproc.COLOR_RGBA2GRAY)
        smallMat.release()

        val blurKsize = params.medianBlurKsize.coerceAtLeast(3)
        Imgproc.medianBlur(matBundle.getGray(), matBundle.getBlurred(), blurKsize)

        // --- CLAHE (brightness-adaptive) ---
        Core.meanStdDev(matBundle.getBlurred(), matBundle.getMean(), matBundle.getStd())
        val avgBrightness = matBundle.getMean().toArray()[0]
        val brightness = avgBrightness.coerceIn(20.0, 200.0)
        val dimBoost = if (brightness < 80.0) {
            8.0 / (brightness + 10.0)
        } else {
            0.0
        }
        val brightBoost = if (brightness > 130.0) {
            (brightness - 130.0) / 100.0
        } else {
            0.0
        }
        val claheClipLimit = (0.5 + dimBoost + brightBoost).coerceIn(1.0, 1.5)
        _detectionParams.value = _detectionParams.value.copy(
            brightness = "%.1f".format(avgBrightness),
            claheClipLimit = "%.2f".format(claheClipLimit)
        )
        Log.d("DocScan5", "  CLAHE: clipLimit=${"%.2f".format(claheClipLimit)}")
        val tileSize = params.claheTileSize.coerceAtLeast(8).toDouble()
        val clahe = Imgproc.createCLAHE(claheClipLimit, Size(tileSize, tileSize))
        clahe.apply(matBundle.getBlurred(), matBundle.getEnhanced())

        // --- Morph Close (contrast-gated skip) ---
        Core.meanStdDev(matBundle.getEnhanced(), matBundle.getMean(), matBundle.getStd())
        val enhancedContrast = matBundle.getStd().toArray()[0]
        val skipMorphClose = enhancedContrast < 25.0

        Log.d("DocScan5", "  Morph Close: kernel=${params.morphCloseSize}, contrast=${"%.1f".format(enhancedContrast)}, skip=$skipMorphClose")

        if (skipMorphClose) {
            matBundle.getEnhanced().copyTo(matBundle.getMorph())
        } else {
            val morphCloseKsize = params.morphCloseSize.coerceAtLeast(3).toDouble()
            Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT,
                Size(morphCloseKsize, morphCloseKsize)
            ).also { kernel ->
                matBundle.getKernel().release()
                kernel.copyTo(matBundle.getKernel())
            }
            Imgproc.morphologyEx(matBundle.getEnhanced(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, matBundle.getKernel())
        }

        val morphSource = if (skipMorphClose) matBundle.getEnhanced() else matBundle.getMorph()
        Imgproc.GaussianBlur(morphSource, matBundle.getTemp(), Size(5.0, 5.0), 2.0)

        val otsu = Imgproc.threshold(
            matBundle.getTemp(),
            matBundle.getEdges(),
            0.0, 255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )

        val cannyHigh = otsu
        val cannyLow = cannyHigh * 0.2

        _detectionParams.value = _detectionParams.value.copy(
            cannyHigh = cannyHigh.toInt().toString(),
            cannyLow = cannyLow.toInt().toString()
        )
        Log.d("DocScan5", "  Canny: high=${"%.1f".format(cannyHigh)}, low=${"%.1f".format(cannyLow)}")

        Imgproc.Canny(matBundle.getTemp(), matBundle.getEdges(), cannyLow, cannyHigh)

        // Baseline: copy Canny edges to morph slot so the pipeline always has edge data.
        // Post-processing steps (Strong Close, Directional Suppression) build on top.
        matBundle.getEdges().copyTo(matBundle.getMorph())

        // --- Strong Closing (post-Canny) ---
        var closeKsize = params.strongCloseSize.coerceIn(3, 5)
        if (closeKsize % 2 == 0) closeKsize++
        Log.d("DocScan5", "  Strong Close: kernel=$closeKsize")
        Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT,
            Size(closeKsize.toDouble(), closeKsize.toDouble())
        ).also { kernel ->
            matBundle.getKernel2().release()
            kernel.copyTo(matBundle.getKernel2())
        }
        Imgproc.morphologyEx(matBundle.getEdges(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, matBundle.getKernel2())

        // --- Directional Suppression (H+V MORPH_CLOSE) ---
        val dirKsize = params.directionalKernelSize.coerceIn(1, 5)
        Log.d("DocScan5", "  Directional Suppression: kernel=$dirKsize")
        Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT,
            Size(dirKsize.toDouble(), 1.0)
        ).also { kernel ->
            matBundle.getHorizontalKernel().release()
            kernel.copyTo(matBundle.getHorizontalKernel())
        }
        Imgproc.morphologyEx(
            matBundle.getMorph(),
            matBundle.getHorizontalClose(),
            Imgproc.MORPH_CLOSE,
            matBundle.getHorizontalKernel()
        )

        Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT,
            Size(1.0, dirKsize.toDouble())
        ).also { kernel ->
            matBundle.getVerticalKernel().release()
            kernel.copyTo(matBundle.getVerticalKernel())
        }
        Imgproc.morphologyEx(
            matBundle.getHorizontalClose(),
            matBundle.getVerticalClose(),
            Imgproc.MORPH_CLOSE,
            matBundle.getVerticalKernel()
        )

        matBundle.getVerticalClose().copyTo(matBundle.getMorph())

        Log.d("DocScan5", "=== preprocess END ===")
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
        Log.d("DocScan5", "=== detectQuad START ===")

        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(
            morphImage,
            contours,
            matBundle.getHierarchy(),
            Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        Log.d("DocScan5", "  found ${contours.size} contours")

        val frameArea = scaledWidth * scaledHeight
        val minArea = frameArea * params.minAreaFraction
        val candidates = mutableListOf<MatOfPoint>()
        val approx = matBundle.getApprox()

        for (contour in contours) {
            val area = abs(Geometry.contourArea(contour))
            if (area < minArea) continue
            if (contour.total() < 10) continue

            val pts2f = MatOfPoint2f(*contour.toArray().map { Point(it.x, it.y) }.toTypedArray())
            val peri = Geometry.arcLength(pts2f, true)
            Geometry.approxPolyDP(pts2f, approx, 0.015 * peri, true)

            if (approx.total() == 4L && isRectangle(approx)) {
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
            }
        }

        val best = candidates.maxByOrNull { abs(Geometry.contourArea(it)) }
        val result = best?.let { MatOfPoint(*it.toArray()) }
        candidates.forEach { it.release() }

        Log.d("DocScan5", "  candidates=${candidates.size}, result=${if (result != null) "found" else "null"}")
        Log.d("DocScan5", "=== detectQuad END ===")
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
        fun computeAutoClaheClipLimit(avgBrightness: Double): Double {
            val brightness = avgBrightness.coerceIn(20.0, 200.0)
            val dimBoost = if (brightness < 80.0) {
                40.0 / (brightness + 10.0)
            } else {
                0.0
            }
            val brightBoost = if (brightness > 130.0) {
                (brightness - 130.0) / 60.0
            } else {
                0.0
            }
            return (0.5 + dimBoost + brightBoost).coerceIn(1.0, 1.5)
        }

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
            return maxDeviation < 15
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