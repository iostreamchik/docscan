package io.github.iostreamchik.scanner

import io.github.iostreamchik.scanner.data.utils.calculateWarpedDimensions
import io.github.iostreamchik.scanner.data.utils.quadDistance
import io.github.iostreamchik.scanner.data.utils.sortQuadPoints
import org.junit.Assert.assertEquals
import org.junit.Test
import org.opencv.core.Point

class OpenCvJvmSmokeTest {

    @Test
    fun pureJavaPointFunctionsRunOnJvm() {
        val shuffled = listOf(
            Point(10.0, 10.0),
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(0.0, 10.0)
        )
        val sorted = sortQuadPoints(shuffled)
        assertEquals(4, sorted.size)
        assertEquals(0.0, sorted[0].x, 0.001)
        assertEquals(0.0, sorted[0].y, 0.001)

        assertEquals(0.0, quadDistance(sorted,
            sortQuadPoints(shuffled.sortedBy { it.x + it.y }), 100.0, 100.0), 0.001)

        val dims = calculateWarpedDimensions(Point(0.0, 0.0), Point(100.0, 0.0), Point(100.0, 150.0), Point(0.0, 150.0))
        assertEquals(100, dims.first)
        assertEquals(150, dims.second)
    }
}
