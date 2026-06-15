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
     * Computes cannyHigh and cannyLow thresholds from a grayscale frame.
     * Uses Otsu's binarization on a structurally blurred version of the frame,
     * then applies temporal EMA smoothing to prevent threshold flicker.
     *
     * @param grayMat Single-channel grayscale matrix (Y plane from camera frame)
     * @return Pair of (cannyHigh, cannyLow) thresholds
     */
    fun computeThreshold(grayMat: Mat): Pair<Double, Double>

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

    override fun computeThreshold(grayMat: Mat): Pair<Double, Double> {
        // Step 1: Light blur — collapse minor high-frequency noise while preserving
        // the document boundary signal in the histogram. A 3×3 blur is small enough
        // that Otsu sees accurate contrast levels for both static files and camera
        // frames, avoiding the inflated thresholds that a 7×7 blur produces.
        Imgproc.GaussianBlur(grayMat, matBundle.getOtsuBlur(), Size(3.0, 3.0), 1.0)

        // Step 2: Extract the optimal global threshold using Otsu's method.
        // We use getOtsuThreshold() as a scratch buffer — only the returned double value matters.
        val rawOtsu = Imgproc.threshold(
            matBundle.getOtsuBlur(),
            matBundle.getOtsuThreshold(),
            0.0, 255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )

        // Step 3: Compute image contrast (std dev of grayscale) to adapt the
        // hysteresis ratio. Low-contrast images need a wider gap between low
        // and high thresholds to capture faint document edges.
        Core.meanStdDev(grayMat, matBundle.getMean(), matBundle.getStd())
        val contrast = matBundle.getStd().toArray()[0]

        // Step 4: Temporal smoothing via Exponential Moving Average (EMA).
        // Eliminates micro-flicker caused by sensor noise and minor lighting oscillations.
        smoothedHigh = if (smoothedHigh < 0.0) {
            rawOtsu // Initialize on first frame
        } else {
            emaAlpha * rawOtsu + (1.0 - emaAlpha) * smoothedHigh
        }

        // Step 5: Contrast-aware hysteresis ratio.
        // - High contrast (> 40): 2:1 ratio — sharp edges, easy to detect
        // - Medium contrast (20–40): 3:1 ratio — moderate edges, lower threshold helps
        // - Low contrast (< 20): 4:1 ratio — faint edges, much lower low threshold
        val ratio = when {
            contrast > 40.0 -> 0.5   // 2:1
            contrast > 20.0 -> 0.33  // 3:1
            else               -> 0.25 // 4:1
        }

        Log.d("CannyThreshold", "OtsuHigh=$smoothedHigh contrast=$contrast ratio=$ratio")
        return Pair(smoothedHigh, smoothedHigh * ratio)
    }

    override fun reset() {
        smoothedHigh = -1.0
    }

    override fun release() {
        // No native resources owned directly — MatBundle handles its own cleanup
    }
}
