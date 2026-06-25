package io.github.iostreamchik.scanner.opencv

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Interface for computing adaptive Canny edge thresholds using gradient-based
 * Otsu's method with temporal EMA smoothing and a configurable low/high ratio.
 */
interface ICannyThresholdCalculator {
    /**
     * Computes cannyHigh and cannyLow thresholds from a pre-processed grayscale frame.
     * The input should be the SAME image Canny will run on (after CLAHE, morph,
     * and pre-Canny blur) so that Otsu gradient statistics match actual edge strength.
     *
     * Uses Sobel gradient magnitude + Otsu for edge-strength-aligned thresholds,
     * EMA smoothing for temporal stability, and a 0.25 ratio for Canny hysteresis.
     *
     * @param processedMat Pre-processed single-channel grayscale matrix (already blurred)
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
 * Computes Otsu on the Sobel gradient magnitude (|Gx| + |Gy|) rather than raw
 * pixel intensities. This aligns the threshold with what Canny actually measures
 * — gradient strength — which is critical for bright images where intensity-based
 * Otsu returns values (150-220) far above actual edge gradients (20-60).
 *
 * EMA smoothing provides temporal stability; a [ThresholdFloor, ThresholdCeiling]
 * band prevents pathological thresholds in extreme lighting.
 *
 * @param matBundle Pre-allocated OpenCV matrix pool for zero-allocation processing
 * @param emaAlpha Exponential moving average alpha (0.0–1.0). Lower = smoother but slower to adapt.
 */
class CannyThresholdCalculator(
    private val matBundle: IMatBundle,
    private val emaAlpha: Double = 0.15
) : ICannyThresholdCalculator {

    private var smoothedHigh = -1.0

    // Ratio for Canny hysteresis.
    private val LowHighRatio = 0.25

    // Floor prevents pathological thresholds in extreme lighting.
    private val ThresholdFloor = 10.0

    // Ceiling prevents extreme thresholds.
    private val ThresholdCeiling = 80.0

    // Reusable Sobel mats to avoid per-frame allocation
    private val gradX: Mat = Mat()
    private val gradY: Mat = Mat()

    override fun computeThreshold(processedMat: Mat): Pair<Double, Double> {
        // Input is already the pre-Canny blurred enhanced image — no extra blur needed.
        // Sobel gradients on this image directly match what Canny will measure.
        Imgproc.Sobel(processedMat, gradX, CvType.CV_32F, 1, 0, 3)
        Imgproc.Sobel(processedMat, gradY, CvType.CV_32F, 0, 1, 3)

        // Step 3: Combine |Gx| + |Gy| as a fast gradient magnitude approximation.
        // (L1 norm is faster than sqrt(Gx² + Gy²) and sufficient for Otsu histogram.)
        Core.convertScaleAbs(gradX, matBundle.getOtsuThreshold(), 1.0, 0.0)
        Core.convertScaleAbs(gradY, matBundle.getTemp(), 1.0, 0.0)
        Core.add(matBundle.getOtsuThreshold(), matBundle.getTemp(), matBundle.getOtsuThreshold())

        // Step 4: Otsu on gradient magnitude histogram.
        val rawOtsu = Imgproc.threshold(
            matBundle.getOtsuThreshold(),
            matBundle.getTemp(),
            0.0, 255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )

        // Step 5: Temporal EMA smoothing to prevent frame-to-frame flicker.
        smoothedHigh = if (smoothedHigh < 0.0) {
            rawOtsu
        } else {
            (emaAlpha * rawOtsu) + ((1.0 - emaAlpha) * smoothedHigh)
        }

        // Step 6: Clamp to [floor, ceiling] band.
        val finalHigh = smoothedHigh.coerceIn(ThresholdFloor, ThresholdCeiling)

        // Step 7: Ratio for Canny hysteresis.
        val finalLow = finalHigh * LowHighRatio

        return Pair(finalHigh, finalLow)
    }

    override fun reset() {
        smoothedHigh = -1.0
    }

    override fun release() {
        gradX.release()
        gradY.release()
    }
}
