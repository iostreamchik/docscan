package io.github.iostreamchik.scanner.data.utils

import android.util.Log
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import java.lang.Math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Sorts 4 corner points into a consistent clockwise order:
 * [top-left, top-right, bottom-right, bottom-left].
 *
 * Uses centroid angle sorting with top-left (smallest x+y) as the anchor,
 * then verifies clockwise winding via the signed area (shoelace formula)
 * and reverses if counter-clockwise.
 */
fun sortQuadPoints(points: List<Point>): List<Point> {
    if (points.size != 4) {
        Log.w(
            "QuadGeometry",
            "sortQuadPoints: Invalid quad with ${points.size} points, returning empty list"
        )
        return emptyList()
    }

    // Compute centroid
    val centerX = points.sumOf { it.x } / 4.0
    val centerY = points.sumOf { it.y } / 4.0

    // Sort by angle around centroid — gives a consistent circular order
    val sortedByAngle = points.sortedBy {
        atan2(it.y - centerY, it.x - centerX)
    }

    // Rotate list so top-left (smallest x+y) is first
    val topLeftIndex = sortedByAngle
        .mapIndexed { index, p -> index to (p.x + p.y) }
        .minBy { it.second }
        .first

    val rotated = List(4) { i ->
        sortedByAngle[(topLeftIndex + i) % 4]
    }

    // Ensure clockwise winding in image coordinates (positive signed area).
    // In image coords (y-down), positive signed area = clockwise winding.
    // If negative, the order is counter-clockwise → reverse the last 3 points.
    val signedArea = rotated[0].x * rotated[1].y - rotated[1].x * rotated[0].y +
                     rotated[1].x * rotated[2].y - rotated[2].x * rotated[1].y +
                     rotated[2].x * rotated[3].y - rotated[3].x * rotated[2].y +
                     rotated[3].x * rotated[0].y - rotated[0].x * rotated[3].y

    return if (signedArea > 0) rotated
    else listOf(rotated[0], rotated[3], rotated[2], rotated[1])
}

/**
 * Computes the average corner-to-corner distance between two sorted quads,
 * normalized by the frame diagonal. Returns a value in [0, 1] where 0 = identical.
 */
fun quadDistance(
    quad1: List<Point>,
    quad2: List<Point>,
    frameWidth: Double,
    frameHeight: Double
): Double {
    if (quad1.size != 4 || quad2.size != 4) return Double.MAX_VALUE

    val diagonal = sqrt(frameWidth * frameWidth + frameHeight * frameHeight)

    var totalDistance = 0.0

    for (i in 0 until 4) {
        val dx = quad1[i].x - quad2[i].x
        val dy = quad1[i].y - quad2[i].y
        totalDistance += sqrt(dx * dx + dy * dy)
    }

    // average corner shift normalized by frame diagonal
    return (totalDistance / 4.0) / diagonal
}

/**
 * Checks if a 4-point contour approximates a rectangle by verifying
 * that all interior angles are close to 90°.
 */
fun isRectangle(approx: MatOfPoint2f, toleranceDegrees: Double = 15.0): Boolean {
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

    return maxDeviation < toleranceDegrees
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

/**
 * Computes the maximum deviation from 90° across all four interior angles of a quad.
 */
fun computeMaxAngleDeviation(corners: List<Point>): Double {
    if (corners.size != 4) return Double.MAX_VALUE

    var maxDeviation = 0.0
    for (i in 0..3) {
        val angle = computeAngle(
            corners[(i + 1) % 4],
            corners[(i + 3) % 4],
            corners[i]
        )
        maxDeviation = max(maxDeviation, abs(90.0 - angle))
    }
    return maxDeviation
}

/**
 * Validates that all four interior angles of a quad are within [maxDeviationDegrees] of 90°.
 */
fun validateQuadRectangularity(
    corners: List<Point>,
    maxDeviationDegrees: Double = 15.0
): Boolean = computeMaxAngleDeviation(corners) < maxDeviationDegrees

/**
 * Computes a simple hash for a quad's point coordinates.
 * Used to detect when the same quad is detected repeatedly.
 */
fun quadHash(quad: MatOfPoint): Long {
    val points = quad.toArray()
    var hash: Long = 1
    for (p in points) {
        hash = 31 * hash + p.x.toInt()
        hash = 31 * hash + p.y.toInt()
    }
    return hash
}
