package io.github.iostreamchik.scanner

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
 * Sorts 4 corner points into a consistent order:
 * [top-left, top-right, bottom-right, bottom-left].
 *
 * Uses centroid angle sorting with top-left (smallest x+y) as the anchor.
 */
fun sortQuadPoints(points: List<Point>): List<Point> {
    if (points.size != 4) {
        Log.w(
            "QuadGeometry",
            "sortQuadPoints: Invalid quad with ${points.size} points, returning empty list"
        )
        return emptyList()
    }

    // 1️⃣ Compute centroid
    val centerX = points.sumOf { it.x } / 4.0
    val centerY = points.sumOf { it.y } / 4.0

    // 2️⃣ Sort by angle around centroid (clockwise)
    val sortedByAngle = points.sortedBy {
        atan2(it.y - centerY, it.x - centerX)
    }

    // 3️⃣ Now ensure consistent starting point (top-left first)
    // Top-left = smallest (x + y)
    val topLeftIndex = sortedByAngle
        .mapIndexed { index, p -> index to (p.x + p.y) }
        .minBy { it.second }
        .first

    // Rotate list so top-left is first
    return List(4) { i ->
        sortedByAngle[(topLeftIndex + i) % 4]
    }
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
