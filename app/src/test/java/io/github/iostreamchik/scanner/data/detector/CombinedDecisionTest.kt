package io.github.iostreamchik.scanner.data.detector

import org.opencv.core.Point
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CombinedDecisionTest {

    private companion object {
        val SQUARE = listOf(
            Point(100.0, 100.0),
            Point(300.0, 100.0),
            Point(300.0, 300.0),
            Point(100.0, 300.0)
        )

        val PARALLELOGRAM = listOf(
            Point(0.0, 0.0),
            Point(200.0, 0.0),
            Point(300.0, 100.0),
            Point(100.0, 100.0)
        )

        val SHUFFLED_SQUARE = listOf(
            Point(300.0, 300.0),
            Point(100.0, 100.0),
            Point(100.0, 300.0),
            Point(300.0, 100.0)
        )

        const val EPS = 1e-9
    }

    private fun candidate(
        source: AsyncDetectorSource,
        quad: List<Point>?,
        valid: Boolean = true
    ) = DetectionCandidate(source, quad, valid)

    @Test
    fun `deviation of null quad is MAX_VALUE`() {
        assertTrue(candidate(AsyncDetectorSource.MINIMAL, null).deviation == Double.MAX_VALUE)
    }

    @Test
    fun `deviation of axis-aligned square is zero`() {
        assertEquals(0.0, candidate(AsyncDetectorSource.MINIMAL, SQUARE).deviation, EPS)
    }

    @Test
    fun `deviation of 45 degree parallelogram is 45`() {
        assertEquals(45.0, candidate(AsyncDetectorSource.MINIMAL, PARALLELOGRAM).deviation, 1e-6)
    }

    @Test
    fun `deviation of non-4-point quad is MAX_VALUE`() {
        assertTrue(candidate(AsyncDetectorSource.MINIMAL, SQUARE.take(3)).deviation == Double.MAX_VALUE)
    }

    @Test
    fun `deviation sorts unsorted points before scoring`() {
        assertEquals(0.0, candidate(AsyncDetectorSource.MINIMAL, SHUFFLED_SQUARE).deviation, EPS)
    }

    @Test
    fun `primary selection on empty list is null`() {
        assertNull(CombinedDecision.selectPrimaryWinner(emptyList()))
    }

    @Test
    fun `primary selection with single valid candidate returns it`() {
        val winner = CombinedDecision.selectPrimaryWinner(
            listOf(candidate(AsyncDetectorSource.DIRECTIONAL_SUPPRESSION, SQUARE))
        )
        assertEquals(AsyncDetectorSource.DIRECTIONAL_SUPPRESSION, winner?.source)
    }

    @Test
    fun `primary selection picks lower deviation`() {
        val winner = CombinedDecision.selectPrimaryWinner(
            listOf(
                candidate(AsyncDetectorSource.MINIMAL, PARALLELOGRAM),
                candidate(AsyncDetectorSource.DIRECTIONAL_SUPPRESSION, SQUARE)
            )
        )
        assertEquals(AsyncDetectorSource.DIRECTIONAL_SUPPRESSION, winner?.source)
    }

    @Test
    fun `primary selection is independent of candidate order`() {
        val candidates = listOf(
            candidate(AsyncDetectorSource.MINIMAL, PARALLELOGRAM),
            candidate(AsyncDetectorSource.DIRECTIONAL_SUPPRESSION, SQUARE)
        )
        val forward = CombinedDecision.selectPrimaryWinner(candidates)
        val backward = CombinedDecision.selectPrimaryWinner(candidates.reversed())
        assertEquals(forward?.source, backward?.source)
        assertEquals(AsyncDetectorSource.DIRECTIONAL_SUPPRESSION, forward?.source)
    }

    @Test
    fun `primary selection excludes invalid quad even with lower deviation`() {
        val winner = CombinedDecision.selectPrimaryWinner(
            listOf(
                candidate(AsyncDetectorSource.MINIMAL, SQUARE, valid = false),
                candidate(AsyncDetectorSource.DIRECTIONAL_SUPPRESSION, PARALLELOGRAM)
            )
        )
        assertEquals(AsyncDetectorSource.DIRECTIONAL_SUPPRESSION, winner?.source)
    }

    @Test
    fun `primary selection with all null quads is null`() {
        assertNull(
            CombinedDecision.selectPrimaryWinner(
                listOf(
                    candidate(AsyncDetectorSource.MINIMAL, null),
                    candidate(AsyncDetectorSource.DIRECTIONAL_SUPPRESSION, null)
                )
            )
        )
    }

    @Test
    fun `primary selection with all invalid quads is null`() {
        assertNull(
            CombinedDecision.selectPrimaryWinner(
                listOf(
                    candidate(AsyncDetectorSource.MINIMAL, SQUARE, valid = false),
                    candidate(AsyncDetectorSource.DIRECTIONAL_SUPPRESSION, PARALLELOGRAM, valid = false)
                )
            )
        )
    }

    @Test
    fun `primary selection skips null quad in favor of valid one`() {
        val winner = CombinedDecision.selectPrimaryWinner(
            listOf(
                candidate(AsyncDetectorSource.MINIMAL, null),
                candidate(AsyncDetectorSource.DIRECTIONAL_SUPPRESSION, PARALLELOGRAM)
            )
        )
        assertEquals(AsyncDetectorSource.DIRECTIONAL_SUPPRESSION, winner?.source)
    }

    @Test
    fun `primary selection tie keeps first candidate`() {
        val winner = CombinedDecision.selectPrimaryWinner(
            listOf(
                candidate(AsyncDetectorSource.MINIMAL, SQUARE),
                candidate(AsyncDetectorSource.DIRECTIONAL_SUPPRESSION, SQUARE)
            )
        )
        assertEquals(AsyncDetectorSource.MINIMAL, winner?.source)
    }

    @Test
    fun `primary selection treats degenerate quad as losing to square`() {
        val winner = CombinedDecision.selectPrimaryWinner(
            listOf(
                candidate(AsyncDetectorSource.MINIMAL, SQUARE.take(3)),
                candidate(AsyncDetectorSource.DIRECTIONAL_SUPPRESSION, SQUARE)
            )
        )
        assertEquals(AsyncDetectorSource.DIRECTIONAL_SUPPRESSION, winner?.source)
    }

    @Test
    fun `fallback selection on empty list is null`() {
        assertNull(CombinedDecision.selectFallbackWinner(emptyList()))
    }

    @Test
    fun `fallback selection with all null quads is null`() {
        assertNull(
            CombinedDecision.selectFallbackWinner(
                listOf(
                    candidate(AsyncDetectorSource.HEATMAP_CORNER, null),
                    candidate(AsyncDetectorSource.CORNER_KEYPOINT, null),
                    candidate(AsyncDetectorSource.SEGMENTATION, null)
                )
            )
        )
    }

    @Test
    fun `fallback selection prefers first source over better deviation`() {
        val winner = CombinedDecision.selectFallbackWinner(
            listOf(
                candidate(AsyncDetectorSource.HEATMAP_CORNER, PARALLELOGRAM),
                candidate(AsyncDetectorSource.SEGMENTATION, SQUARE)
            )
        )
        assertEquals(AsyncDetectorSource.HEATMAP_CORNER, winner?.source)
    }

    @Test
    fun `fallback selection falls through null heatmap to keypoint`() {
        val winner = CombinedDecision.selectFallbackWinner(
            listOf(
                candidate(AsyncDetectorSource.HEATMAP_CORNER, null),
                candidate(AsyncDetectorSource.CORNER_KEYPOINT, PARALLELOGRAM)
            )
        )
        assertEquals(AsyncDetectorSource.CORNER_KEYPOINT, winner?.source)
    }

    @Test
    fun `fallback selection falls through to segmentation`() {
        val winner = CombinedDecision.selectFallbackWinner(
            listOf(
                candidate(AsyncDetectorSource.HEATMAP_CORNER, null),
                candidate(AsyncDetectorSource.CORNER_KEYPOINT, null),
                candidate(AsyncDetectorSource.SEGMENTATION, PARALLELOGRAM)
            )
        )
        assertEquals(AsyncDetectorSource.SEGMENTATION, winner?.source)
    }

    @Test
    fun `fallback selection skips invalid quad and continues chain`() {
        val winner = CombinedDecision.selectFallbackWinner(
            listOf(
                candidate(AsyncDetectorSource.HEATMAP_CORNER, SQUARE, valid = false),
                candidate(AsyncDetectorSource.SEGMENTATION, PARALLELOGRAM)
            )
        )
        assertEquals(AsyncDetectorSource.SEGMENTATION, winner?.source)
    }

    @Test
    fun `fallback selection with all valid quads returns first in priority order`() {
        val winner = CombinedDecision.selectFallbackWinner(
            listOf(
                candidate(AsyncDetectorSource.HEATMAP_CORNER, PARALLELOGRAM),
                candidate(AsyncDetectorSource.CORNER_KEYPOINT, SQUARE),
                candidate(AsyncDetectorSource.SEGMENTATION, SQUARE)
            )
        )
        assertEquals(AsyncDetectorSource.HEATMAP_CORNER, winner?.source)
    }

    @Test
    fun `fallback selection with all invalid quads is null`() {
        assertNull(
            CombinedDecision.selectFallbackWinner(
                listOf(
                    candidate(AsyncDetectorSource.HEATMAP_CORNER, SQUARE, valid = false),
                    candidate(AsyncDetectorSource.SEGMENTATION, PARALLELOGRAM, valid = false)
                )
            )
        )
    }

    @Test
    fun `candidate equality compares source quad and validity`() {
        val a = candidate(AsyncDetectorSource.MINIMAL, SQUARE)
        val b = candidate(AsyncDetectorSource.MINIMAL, SQUARE)
        val c = candidate(AsyncDetectorSource.DIRECTIONAL_SUPPRESSION, SQUARE)
        assertEquals(a, b)
        if (a == c) throw AssertionError("different sources must not be equal")
    }
}
