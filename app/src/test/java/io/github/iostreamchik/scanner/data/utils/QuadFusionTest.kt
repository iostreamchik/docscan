package io.github.iostreamchik.scanner.data.utils

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.core.Point

class QuadFusionTest {

    private val square = listOf(
        Point(0.0, 0.0),
        Point(10.0, 0.0),
        Point(10.0, 10.0),
        Point(0.0, 10.0)
    )

    private fun shiftedSquare(dx: Double, dy: Double) = square.map { Point(it.x + dx, it.y + dy) }

    private fun requireFused(fusion: QuadFusion): List<Point> {
        val fused = fusion.fusedQuad()
        assertNotNull(fused)
        return fused!!
    }

    @Test
    fun unstableBelowMaxHistory() {
        val fusion = QuadFusion()
        repeat(3) { fusion.update(square) }
        assertFalse(fusion.isStable(100.0, 100.0))
    }

    @Test
    fun stableWithIdenticalQuads() {
        val fusion = QuadFusion()
        repeat(4) { fusion.update(square) }
        assertTrue(fusion.isStable(100.0, 100.0))
    }

    @Test
    fun stableWithJitterBelowThreshold() {
        val fusion = QuadFusion()
        repeat(4) { i -> fusion.update(shiftedSquare(i * 1.0, 0.0)) }
        assertTrue(fusion.isStable(1000.0, 1000.0))
    }

    @Test
    fun unstableWithJitterAboveThreshold() {
        val fusion = QuadFusion()
        repeat(4) { i -> fusion.update(shiftedSquare(i * 100.0, 0.0)) }
        assertFalse(fusion.isStable(1000.0, 1000.0))
    }

    @Test
    fun stabilityThresholdIsFrameRelative() {
        val smallFrame = QuadFusion()
        repeat(4) { i -> smallFrame.update(shiftedSquare(i * 10.0, 0.0)) }
        assertFalse(smallFrame.isStable(100.0, 100.0))

        val largeFrame = QuadFusion()
        repeat(4) { i -> largeFrame.update(shiftedSquare(i * 10.0, 0.0)) }
        assertTrue(largeFrame.isStable(10000.0, 10000.0))
    }

    @Test
    fun fusedQuadNullWhenEmpty() {
        assertNull(QuadFusion().fusedQuad())
    }

    @Test
    fun fusedQuadAveragesHistory() {
        val fusion = QuadFusion()
        fusion.update(shiftedSquare(0.0, 0.0))
        fusion.update(shiftedSquare(2.0, 4.0))
        val fused = requireFused(fusion)
        val expected = listOf(
            Point(1.0, 2.0),
            Point(11.0, 2.0),
            Point(11.0, 12.0),
            Point(1.0, 12.0)
        )
        for (i in 0..3) {
            assertEquals(expected[i].x, fused[i].x, 1e-9)
            assertEquals(expected[i].y, fused[i].y, 1e-9)
        }
    }

    @Test
    fun historyCappedAtMaxOldestEvicted() {
        val fusion = QuadFusion(maxHistory = 4)
        repeat(5) { i -> fusion.update(shiftedSquare(i * 2.0, 0.0)) }
        val fused = requireFused(fusion)
        val lastFour = (1..4).map { it * 2.0 }
        assertEquals(lastFour.average(), fused[0].x, 1e-9)
    }

    @Test
    fun pointsSortedBeforeInsertion() {
        val fusion = QuadFusion()
        val shuffled = listOf(
            Point(10.0, 10.0),
            Point(0.0, 0.0),
            Point(0.0, 10.0),
            Point(10.0, 0.0)
        )
        fusion.update(shuffled)
        val fused = requireFused(fusion)
        assertEquals(0.0, fused[0].x, 1e-9)
        assertEquals(0.0, fused[0].y, 1e-9)
        assertEquals(10.0, fused[1].x, 1e-9)
        assertEquals(0.0, fused[1].y, 1e-9)
    }

    @Test
    fun wrongSizeIgnored() {
        val fusion = QuadFusion()
        fusion.update(square.take(3))
        fusion.update(emptyList())
        assertNull(fusion.fusedQuad())
        assertFalse(fusion.isStable(100.0, 100.0))
    }

    @Test
    fun clearEmptiesHistory() {
        val fusion = QuadFusion()
        repeat(4) { fusion.update(square) }
        fusion.clear()
        assertNull(fusion.fusedQuad())
        assertFalse(fusion.isStable(100.0, 100.0))
    }

    @Test
    fun customMaxHistoryRespected() {
        val fusion = QuadFusion(maxHistory = 2)
        fusion.update(square)
        assertFalse(fusion.isStable(100.0, 100.0))
        fusion.update(square)
        assertTrue(fusion.isStable(100.0, 100.0))
    }
}
