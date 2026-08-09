package io.github.iostreamchik.scanner.data.detector

import io.github.iostreamchik.scanner.entity.DetectionParameters
import io.github.iostreamchik.scanner.entity.IntermediateBitmaps
import io.github.iostreamchik.scanner.entity.PipelineParams
import io.github.iostreamchik.scanner.data.opencv.IMatBundle
import io.github.iostreamchik.scanner.data.opencv.OpenCVAdapter
import io.github.iostreamchik.scanner.data.utils.scoreContourWithParams
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import kotlin.math.abs

class DocumentDetectorMinimal(
    internal val matBundle: IMatBundle
) : IDocumentDetector {

    private val _detectionParams = MutableStateFlow(DetectionParameters())
    override val detectionParams: StateFlow<DetectionParameters> = _detectionParams.asStateFlow()

    override fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat {
        OpenCVAdapter.resizeToGray(rawMat, scaledWidth, scaledHeight, matBundle.getGray())

        val blurKsize = params.medianBlurKsize.coerceAtLeast(3)
        Imgproc.medianBlur(matBundle.getGray(), matBundle.getBlurred(), blurKsize)

        val avgBrightness = OpenCVAdapter.getAverageBrightness(matBundle.getBlurred(), matBundle)
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
        val claheClipLimit = (0.5 + dimBoost + brightBoost).coerceIn(1.0, 1.5)

        val tileSize = params.claheTileSize.coerceAtLeast(8).toDouble()
        OpenCVAdapter.applyClahe(matBundle.getBlurred(), matBundle.getEnhanced(), claheClipLimit, tileSize)

        val enhancedContrast = OpenCVAdapter.getStdDev(matBundle.getEnhanced(), matBundle)
        val skipMorphClose = enhancedContrast < 25.0

        if (skipMorphClose) {
            matBundle.getEnhanced().copyTo(matBundle.getMorph())
        } else {
            val morphCloseKsize = params.morphCloseSize.coerceAtLeast(3).toDouble()
            OpenCVAdapter.createRectKernel(Size(morphCloseKsize, morphCloseKsize), matBundle.getKernel())
            OpenCVAdapter.morphClose(matBundle.getEnhanced(), matBundle.getMorph(), matBundle.getKernel())
        }

        val morphSource = if (skipMorphClose) matBundle.getEnhanced() else matBundle.getMorph()

        Imgproc.GaussianBlur(morphSource, matBundle.getTemp(), Size(5.0, 5.0), 2.0)

        val otsu = Imgproc.threshold(
            matBundle.getTemp(),
            matBundle.getEdges(),
            0.0, 255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )

        val high = otsu
        val low = (high * 0.2)

        _detectionParams.value = _detectionParams.value.copy(
            claheClipLimit = String.format("%.1f", claheClipLimit),
            cannyHigh = String.format("%.0f", high),
            cannyLow = String.format("%.0f", low),
            brightness = String.format("%.0f", brightness)
        )

        Imgproc.Canny(matBundle.getTemp(), matBundle.getEdges(), low, high)

        OpenCVAdapter.createRectKernel(Size(7.0, 7.0), matBundle.getKernel())
        OpenCVAdapter.morphClose(matBundle.getEdges(), matBundle.getMorph(), matBundle.getKernel())

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
        val contours = OpenCVAdapter.findContours(morphImage, matBundle.getHierarchy())

        val frameArea = scaledWidth * scaledHeight
        val minArea = frameArea * params.minAreaFraction
        val candidates = mutableListOf<MatOfPoint>()
        val approx = matBundle.getApprox()

        for (contour in contours) {
            val area = abs(Geometry.contourArea(contour))
            if (area < minArea || contour.total() < 10) continue

            val pts2f = MatOfPoint2f(*contour.toArray().map { Point(it.x, it.y) }.toTypedArray())
            val peri = Geometry.arcLength(pts2f, true)
            Geometry.approxPolyDP(
                pts2f,
                approx,
                params.approxPolyDPTolerance.toDouble() * peri,
                true
            )

            if (approx.total() != 4L || !OpenCVAdapter.isRectangle(approx, 20.0)) {
                pts2f.release()
                continue
            }

            val scaleX = originalWidth.toDouble() / scaledWidth
            val scaleY = originalHeight.toDouble() / scaledHeight
            val scaledPoints = approx.toArray().map { Point(it.x * scaleX, it.y * scaleY) }
            val quad = MatOfPoint(*scaledPoints.toTypedArray())

            val rect = Geometry.boundingRect(quad)
            val scaledArea = area * (scaleX * scaleY)
            val solidity = scaledArea / (rect.width * rect.height).toDouble()

            if (solidity < 0.5) {
                quad.release()
                pts2f.release()
                continue
            }

            candidates.add(quad)
            pts2f.release()
        }

        val best = candidates.maxByOrNull {
            scoreContourWithParams(it, originalWidth, originalHeight, params)
        }
        val result = best?.let { MatOfPoint(*it.toArray()) }
        candidates.forEach { it.release() }

        return result
    }

    override fun captureIntermediateSnapshots(rotation: Int): IntermediateBitmaps =
        matBundle.captureClassicalSnapshots(rotation)

    override fun release() {
        matBundle.releaseAll()
    }
}