package io.github.iostreamchik.scanner

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import io.github.iostreamchik.scanner.opencv.IMatBundle
import io.github.iostreamchik.scanner.opencv.PipelineParams
import java.lang.Math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Shared document detection logic — contour finding, hull computation,
 * quad validation, and scoring. Used by both CameraViewModel and
 * PipelineSettingsViewModel to avoid ~60 lines of duplicated contour
 * detection code in three places.
 */
class DocumentDetector(
    private val matBundle: IMatBundle,
    private val params: PipelineParams = PipelineParams.Default
) {

    /**
     * Extract document candidates from a morph/edge Mat.
     * Returns the best quad or null if no document found.
     */
    fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int
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

        for (contour in contours) {
            val area = Imgproc.contourArea(contour)
            if (area < minArea) continue

            val hull = matBundle.getHull()
            matBundle.getHullPoints().release()
            matBundle.getHullPoints().create(0, 1, CvType.CV_32FC2)
            val approx = matBundle.getApprox()

            Imgproc.convexHull(contour, hull)
            val contourArray = contour.toArray()
            val hullIndices = IntArray(hull.rows().toInt())
            hull.get(0, 0, hullIndices)
            val hullPointList = hullIndices.map { contourArray[it] }
            matBundle.getHullPoints().fromList(hullPointList.map { Point(it.x, it.y) })

            val peri = Imgproc.arcLength(matBundle.getHullPoints(), true)
            Imgproc.approxPolyDP(
                matBundle.getHullPoints(),
                approx,
                params.approxPolyDPTolerance * peri,
                true
            )

            if (approx.total() != 4L) continue

            val scaleX = originalWidth.toDouble() / scaledWidth
            val scaleY = originalHeight.toDouble() / scaledHeight
            val scaledPoints = approx.toArray().map { Point(it.x * scaleX, it.y * scaleY) }
            val quad = MatOfPoint(*scaledPoints.toTypedArray())

            if (!isRectangle(approx)) {
                quad.release()
                continue
            }

            val scaledArea = area * (scaleX * scaleY)
            val rect = Imgproc.boundingRect(quad)
            val solidity = scaledArea / (rect.width * rect.height).toDouble()
            if (solidity < 0.3) {
                quad.release()
                continue
            }

            candidates.add(quad)
        }

        return candidates.maxByOrNull { scoreContourWithParams(it, scaledWidth, scaledHeight, params) }
    }

    companion object {
        /**
         * Checks if a 4-point contour approximates a rectangle by verifying
         * that all interior angles are close to 90°.
         */
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

        /**
         * Computes the interior angle (in degrees) between three points at a vertex.
         */
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
