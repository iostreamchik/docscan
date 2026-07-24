package io.github.iostreamchik.scanner.detector

import android.graphics.Bitmap
import android.util.Log
import io.github.iostreamchik.scanner.fixRotation
import io.github.iostreamchik.scanner.opencv.IMatBundle
import io.github.iostreamchik.scanner.opencv.OpenCVAdapter
import io.github.iostreamchik.scanner.opencv.PipelineParams
import io.github.iostreamchik.scanner.toBitmap
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import kotlin.math.abs

class DocumentDetectorOpenCV5(
    internal val matBundle: IMatBundle
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

        OpenCVAdapter.resizeToGray(rawMat, scaledWidth, scaledHeight, matBundle.getGray())

        val blurKsize = params.medianBlurKsize.coerceAtLeast(3)
        Imgproc.medianBlur(matBundle.getGray(), matBundle.getBlurred(), blurKsize)

        val avgBrightness = OpenCVAdapter.getAverageBrightness(matBundle.getBlurred(), matBundle)
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
        OpenCVAdapter.applyClahe(
            matBundle.getBlurred(),
            matBundle.getEnhanced(),
            claheClipLimit,
            tileSize
        )

        val enhancedContrast = OpenCVAdapter.getStdDev(matBundle.getEnhanced(), matBundle)
        val skipMorphClose = enhancedContrast < 25.0

        Log.d(
            "DocScan5",
            "  Morph Close: kernel=${params.morphCloseSize}, contrast=${
                "%.1f".format(enhancedContrast)
            }, skip=$skipMorphClose"
        )

        if (skipMorphClose) {
            matBundle.getEnhanced().copyTo(matBundle.getMorph())
        } else {
            val morphCloseKsize = params.morphCloseSize.coerceAtLeast(3).toDouble()
            OpenCVAdapter.createRectKernel(
                Size(morphCloseKsize, morphCloseKsize),
                matBundle.getKernel()
            )
            OpenCVAdapter.morphClose(
                matBundle.getEnhanced(),
                matBundle.getMorph(),
                matBundle.getKernel()
            )
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
        Log.d(
            "DocScan5",
            "  Canny: high=${"%.1f".format(cannyHigh)}, low=${"%.1f".format(cannyLow)}"
        )

        Imgproc.Canny(matBundle.getTemp(), matBundle.getEdges(), cannyLow, cannyHigh)

        matBundle.getEdges().copyTo(matBundle.getMorph())

        var closeKsize = params.strongCloseSize.coerceIn(3, 5)
        if (closeKsize % 2 == 0) closeKsize++
        Log.d("DocScan5", "  Strong Close: kernel=$closeKsize")
        OpenCVAdapter.createRectKernel(
            Size(closeKsize.toDouble(), closeKsize.toDouble()),
            matBundle.getKernel2()
        )
        OpenCVAdapter.morphClose(matBundle.getEdges(), matBundle.getMorph(), matBundle.getKernel2())

        val dirKsize = params.directionalKernelSize.coerceIn(1, 5)
        Log.d("DocScan5", "  Directional Suppression: kernel=$dirKsize")
        OpenCVAdapter.createRectKernel(
            Size(dirKsize.toDouble(), 1.0),
            matBundle.getHorizontalKernel()
        )
        Imgproc.morphologyEx(
            matBundle.getMorph(),
            matBundle.getHorizontalClose(),
            Imgproc.MORPH_CLOSE,
            matBundle.getHorizontalKernel()
        )

        OpenCVAdapter.createRectKernel(
            Size(1.0, dirKsize.toDouble()),
            matBundle.getVerticalKernel()
        )
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

        val contours = OpenCVAdapter.findContours(morphImage, matBundle.getHierarchy())
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

            if (approx.total() == 4L && OpenCVAdapter.isRectangle(approx)) {
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

        Log.d(
            "DocScan5",
            "  candidates=${candidates.size}, result=${if (result != null) "found" else "null"}"
        )
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

    override fun captureIntermediateSnapshots(
        rotation: Int
    ): IntermediateSnapshots {
        val toBitmap = { mat: Mat ->
            mat.fixRotation(rotation).toBitmap()
                .copy(Bitmap.Config.ARGB_8888, false)
        }
        return IntermediateSnapshots(
            blur = toBitmap(matBundle.getBlurred()),
            clahe = toBitmap(matBundle.getEnhanced()),
            morph = toBitmap(matBundle.getMorph()),
            edges = toBitmap(matBundle.getEdges())
        )
    }
}