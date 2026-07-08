package io.github.iostreamchik.scanner.opencv

import org.opencv.core.Mat

/**
 * No-op threshold calculator for Compose preview. Returns zero thresholds and
 * has no-op reset/release — avoids UnsatisfiedLinkError when OpenCV native
 * libs aren't loaded (preview).
 */
class MockCannyThresholdCalculator : ICannyThresholdCalculator {

    override fun computeThreshold(grayMat: Mat): Pair<Double, Double> = Pair(0.0, 0.0)

    override fun reset() {
        // No-op for preview
    }

    override fun release() {
        // No-op for preview
    }
}
