package io.github.iostreamchik.scanner.detector

import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import io.github.iostreamchik.scanner.opencv.IMatBundle
import io.github.iostreamchik.scanner.opencv.PipelineParams
import io.github.iostreamchik.scanner.scoreContourWithParams
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.sqrt

class DocumentDetectorMinimal_back(
    private val matBundle: IMatBundle
) : IDocumentDetector {

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

        Imgproc.GaussianBlur(matBundle.getGray(), matBundle.getTemp(), Size(3.0, 3.0), 0.8)

        val otsu = Imgproc.threshold(
            matBundle.getTemp(),
            matBundle.getEdges(),
            0.0, 255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )

        val high = otsu.coerceIn(40.0, 80.0)
        val low = (high * 0.25).coerceIn(5.0, 40.0)

        Imgproc.Canny(matBundle.getTemp(), matBundle.getEdges(), low, high)
        matBundle.getEdges().copyTo(matBundle.getMorph())

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
        val candidates = mutableListOf<MatOfPoint>()
        val approx = matBundle.getApprox()

        for (contour in contours) {
            val area = abs(Geometry.contourArea(contour))
            if (area < minArea || contour.total() < 10) continue

            val pts2f = MatOfPoint2f(*contour.toArray().map { Point(it.x, it.y) }.toTypedArray())
            val peri = Geometry.arcLength(pts2f, true)
            Geometry.approxPolyDP(pts2f, approx, params.approxPolyDPTolerance.toDouble() * peri, true)

            if (approx.total() != 4L || !isRectangle(approx)) {
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
                maxDeviation = maxOf(maxDeviation, abs(90 - angle))
            }

            return maxDeviation < 20
        }

        fun computeAngle(p1: Point, p2: Point, center: Point): Double {
            val dx1 = p1.x - center.x
            val dy1 = p1.y - center.y
            val dx2 = p2.x - center.x
            val dy2 = p2.y - center.y
            val dot = dx1 * dx2 + dy1 * dy2
            val norm1 = sqrt(dx1 * dx1 + dy1 * dy1)
            val norm2 = sqrt(dx2 * dx2 + dy2 * dy2)
            return acos(dot / (norm1 * norm2)) * 180.0 / PI
        }
    }
}