package io.github.iostreamchik.scanner.opencv

import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max

/**
 * Interface for computing adaptive Canny edge thresholds using Otsu's method
 * with temporal EMA smoothing and a fixed low/high ratio.
 */
interface ICannyThresholdCalculator {
    /**
     * Computes cannyHigh and cannyLow thresholds from a grayscale frame.
     * Uses Otsu's method for bimodal document/background scenes, EMA smoothing
     * for temporal stability, and a fixed 1:2 ratio (low = high * 0.5) for
     * Canny hysteresis.
     *
     * @param grayMat Single-channel grayscale matrix
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
 * Uses Otsu's method for bimodal document/background scenes, EMA smoothing
 * for temporal stability, and a fixed 1:2 ratio (low = high * 0.5) for
 * Canny hysteresis edge linking.
 *
 * Otsu is ideal for document scanning because the histogram naturally separates
 * the white document from the darker background into two peaks. No hard clamp
 * is applied — Otsu produces sensible thresholds across a wide range of lighting.
 * Only a gentle floor (10.0) prevents near-zero thresholds in pathological cases.
 *
 * @param matBundle Pre-allocated OpenCV matrix pool for zero-allocation processing
 * @param emaAlpha Exponential moving average alpha (0.0–1.0). Lower = smoother but slower to adapt.
 */
class CannyThresholdCalculator(
    private val matBundle: IMatBundle,
    private val emaAlpha: Double = 0.15
) : ICannyThresholdCalculator {

    private var smoothedHigh = -1.0

    // Fixed 1:2 ratio for Canny hysteresis (classic recommendation)
    private val LowHighRatio = 0.5

    // Gentle floor to prevent near-zero thresholds in extreme overexposure
    private val ThresholdFloor = 10.0

    override fun computeThreshold(grayMat: Mat): Pair<Double, Double> {
        // Step 1: Light 3×3 Gaussian blur to collapse high-frequency noise while
        // preserving the document boundary signal in the histogram.
        Imgproc.GaussianBlur(grayMat, matBundle.getOtsuBlur(), Size(3.0, 3.0), 1.0)

        // Step 2: Otsu's method — ideal for bimodal document/background scenes.
        // Finds the natural separation between the white document and darker background.
        val rawOtsu = Imgproc.threshold(
            matBundle.getOtsuBlur(),
            matBundle.getOtsuThreshold(),
            0.0, 255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )

        // Step 3: Temporal EMA smoothing on the Otsu threshold to prevent frame-to-frame
        // flicker while still adapting to lighting changes.
        smoothedHigh = if (smoothedHigh < 0.0) {
            rawOtsu
        } else {
            (emaAlpha * rawOtsu) + ((1.0 - emaAlpha) * smoothedHigh)
        }

        // Step 4: Gentle floor to prevent near-zero thresholds in extreme overexposure.
        // No ceiling — Otsu naturally produces sensible thresholds across a wide
        // range of lighting conditions (no arbitrary [50, 100] clamp).
        val finalHigh = max(smoothedHigh, ThresholdFloor)

        // Step 5: Fixed 1:2 ratio for Canny hysteresis edge linking.
        // Classic Canny recommendation — low threshold is half the high.
        val finalLow = finalHigh * LowHighRatio

        return Pair(finalHigh, finalLow)
    }

    override fun reset() {
        smoothedHigh = -1.0
    }

    override fun release() {
        // No owned resources — MatBundle handles its own cleanup
    }
}
