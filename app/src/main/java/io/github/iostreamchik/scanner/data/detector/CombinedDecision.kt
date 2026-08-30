package io.github.iostreamchik.scanner.data.detector

import io.github.iostreamchik.scanner.data.utils.computeMaxAngleDeviation
import io.github.iostreamchik.scanner.data.utils.sortQuadPoints
import org.opencv.core.Point

data class DetectionCandidate(
    val source: AsyncDetectorSource,
    val quad: List<Point>?,
    val valid: Boolean,
) {
    val deviation: Double
        get() = quad?.let { computeMaxAngleDeviation(sortQuadPoints(it)) } ?: Double.MAX_VALUE
}

object CombinedDecision {

    fun selectPrimaryWinner(candidates: List<DetectionCandidate>): DetectionCandidate? =
        candidates
            .filter { it.quad != null && it.valid }
            .minByOrNull { it.deviation }

    fun selectFallbackWinner(candidates: List<DetectionCandidate>): DetectionCandidate? =
        candidates.firstOrNull { it.quad != null && it.valid }
}
