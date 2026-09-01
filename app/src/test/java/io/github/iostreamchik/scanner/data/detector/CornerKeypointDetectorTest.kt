package io.github.iostreamchik.scanner.data.detector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.exp

class CornerKeypointDetectorTest {

    @Test
    fun sigmoidZeroIsHalf() {
        assertEquals(0.5, CornerKeypointDetector.sigmoid(0f).toDouble(), 1e-6)
    }

    @Test
    fun sigmoidSymmetry() {
        val x = 1.7f
        assertEquals(
            (1f - CornerKeypointDetector.sigmoid(x)).toDouble(),
            CornerKeypointDetector.sigmoid(-x).toDouble(),
            1e-6
        )
    }

    @Test
    fun sigmoidMonotonic() {
        var previous = CornerKeypointDetector.sigmoid(-10f)
        for (i in -9..10) {
            val value = CornerKeypointDetector.sigmoid(i.toFloat())
            assertTrue(value > previous)
            previous = value
        }
    }

    @Test
    fun sigmoidSaturation() {
        assertTrue(CornerKeypointDetector.sigmoid(-50f).toDouble() < 1e-6)
        assertTrue(CornerKeypointDetector.sigmoid(50f).toDouble() > 1.0 - 1e-6)
    }

    @Test
    fun sigmoidKnownValue() {
        assertEquals(1.0 / (1.0 + exp(-2.0)), CornerKeypointDetector.sigmoid(2f).toDouble(), 1e-5)
    }
}
