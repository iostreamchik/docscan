package io.github.iostreamchik.scanner.data.utils

import org.opencv.core.Point

class QuadFusion(
    private val maxHistory: Int = 4,
    private val movementThreshold: Double = 0.05,
) {
    private val history = ArrayDeque<List<Point>>()

    fun update(points: List<Point>) {
        if (points.size != 4) return
        if (history.size >= maxHistory) {
            history.removeFirst()
        }
        history.addLast(sortQuadPoints(points))
    }

    fun isStable(frameWidth: Double, frameHeight: Double): Boolean {
        if (history.size < maxHistory) return false

        val quads = history.toList()
        var totalMovement = 0.0
        var validPairs = 0

        for (i in 1 until quads.size) {
            totalMovement += quadDistance(
                quads[i - 1],
                quads[i],
                frameWidth,
                frameHeight
            )
            validPairs++
        }

        return validPairs > 0 && (totalMovement / validPairs) < movementThreshold
    }

    fun fusedQuad(): List<Point>? {
        if (history.isEmpty()) return null

        return List(4) { i ->
            Point(
                history.sumOf { it[i].x } / history.size,
                history.sumOf { it[i].y } / history.size
            )
        }
    }

    fun clear() {
        history.clear()
    }
}
