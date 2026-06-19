package io.github.iostreamchik.scanner.opencv

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Interface for computing adaptive Canny edge thresholds using Otsu's method
 * with temporal EMA smoothing.
 */
interface ICannyThresholdCalculator {
    /**
     * Computes cannyHigh and cannyLow thresholds from the image that Canny will process.
     * Uses Otsu's binarization on a structurally blurred version of the image,
     * then applies temporal EMA smoothing to prevent threshold flicker.
     *
     * The input image should be the same preprocessing stage that Canny will run on
     * (e.g., CLAHE-enhanced for camera pipeline, or raw gray for file-scan fallback).
     * This ensures Otsu thresholds match the actual contrast Canny sees.
     *
     * @param preCannyMat Single-channel matrix at the preprocessing stage that Canny will process
     * @return Pair of (cannyHigh, cannyLow) thresholds
     */
    fun computeThreshold(preCannyMat: Mat): Pair<Double, Double>

    /**
     * Resets the temporal EMA filter state. Call when camera session resets.
     */
    fun reset()

    /**
     * Releases any owned resources.
     */
    fun release()
}

/**
 * Default implementation of ICannyThresholdCalculator.
 *
 * Uses a 3x3 Gaussian blur to collapse minor high-frequency noise while
 * preserving the document boundary signal in the histogram, then runs Otsu's
 * method to find the optimal global threshold. The result is smoothed over
 * time via EMA to eliminate frame-by-frame jitter.
 *
 * @param matBundle Pre-allocated OpenCV matrix pool for zero-allocation processing
 * @param emaAlpha Exponential moving average alpha (0.0–1.0). Lower = smoother but slower to adapt.
 */
class CannyThresholdCalculator(
    private val matBundle: IMatBundle,
    private val emaAlpha: Double = 0.15
) : ICannyThresholdCalculator {

    private var smoothedHigh = -1.0

    // Pre-allocate a 1x1 double array field at the class level to capture Core.meanStdDev values safely
    private val stdValOut = DoubleArray(1)

    override fun computeThreshold(preCannyMat: Mat): Pair<Double, Double> { // Note: Change to pass destination if you want zero allocations
        // Step 1: Light 3x3 blur to collapse high-frequency text grain safely
        Imgproc.GaussianBlur(preCannyMat, matBundle.getOtsuBlur(), Size(3.0, 3.0), 1.0)

        // Step 2: Extract the optimal inter-class global threshold via Otsu
        val rawOtsu = Imgproc.threshold(
            matBundle.getOtsuBlur(),
            matBundle.getOtsuThreshold(),
            0.0, 255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )

        // Step 3: Allocation-Free Contrast Extraction
        Core.meanStdDev(preCannyMat, matBundle.getMean(), matBundle.getStd())
        // Extracts raw double values directly into our pre-allocated array heap space
        matBundle.getStd().get(0, 0, stdValOut)
        val contrast = stdValOut[0]

        // Step 4: Temporal smoothing via EMA (Locks down micro-exposure fluctuations)
        smoothedHigh = if (smoothedHigh < 0.0) {
            rawOtsu
        } else {
            (emaAlpha * rawOtsu) + ((1.0 - emaAlpha) * smoothedHigh)
        }

        // Step 5: Continuous Contrast-to-Ratio Interpolation (Eradicates Edge Jitter)
        // Maps contrast linearly between 20.0 (Ratio 0.25) and 40.0 (Ratio 0.5) smoothly
        val clampedContrast = contrast.coerceIn(20.0, 40.0)
        val normalizationFactor = (clampedContrast - 20.0) / (40.0 - 20.0) // Becomes a smooth 0.0 to 1.0 line
        val ratio = 0.25 + (normalizationFactor * (0.5 - 0.25))

        // Optional: Return values. In full production loops, pass a custom object or set class variables directly.
        return Pair(smoothedHigh, smoothedHigh * ratio)
    }

    override fun reset() {
        smoothedHigh = -1.0
    }

    override fun release() {
        // No native resources owned directly — MatBundle handles its own cleanup
    }
}
