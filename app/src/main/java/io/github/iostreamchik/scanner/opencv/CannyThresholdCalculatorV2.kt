package io.github.iostreamchik.scanner.opencv

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc


/**
 * Canny threshold calculator based on **gradient-magnitude Otsu** instead of
 * intensity-domain Otsu.
 *
 * ## How it works
 *
 * 1. **Gaussian blur** — reduces paper texture, JPEG artifacts, and sensor noise
 *    before edge detection.
 * 2. **Sobel gradients** — computes Gx and Gy in both axes.
 * 3. **Gradient magnitude** — combines Gx and Gy via `mag = √(Gx² + Gy²)`,
 *    producing a per-pixel edge-strength map.
 * 4. **Otsu on gradient magnitudes** — finds the bimodal split in the
 *    gradient histogram (weak vs. strong edges), yielding a high threshold
 *    in the **gradient domain** (theoretically correct for Canny).
 * 5. **EMA smoothing** — temporal stabilization prevents frame-to-frame jitter.
 * 6. **Clamping** — high is bounded to [floor, ceiling] for predictable behavior.
 * 7. **Hysteresis** — low = high × ratio (default 0.33 for tighter edge linking).
 *
 * ## Confidence-based fallback
 *
 * When the gradient histogram is nearly flat (almost all-white or almost all-
 * dark frames), Otsu on gradients returns unreliable values. In that case we
 * fall back to safe defaults:
 *
 * - `rawHigh < 15` → camera pointed at a wall / severe underexposure
 * - `rawHigh > 220` → bright white paper in strong sunlight / overexposure
 *
 * Both trigger: `high = 60, low = 20`.
 *
 * ## Design philosophy
 *
 * For document detection, edge quality beyond a reasonable range adds little
 * value. The contour scoring, quad validation, and temporal stabilization
 * stages have far more impact than fine-tuning the threshold estimator.
 * This calculator keeps things simple and predictable.
 *
 * @param matBundle Pre-allocated OpenCV matrix pool for zero-allocation processing
 * @param sobelKsize Sobel kernel size (1=Scharr, 3/5/7)
 * @param blurSize Gaussian blur kernel (5×5 recommended for phone cameras)
 * @param blurSigma Gaussian blur sigma
 * @param lowHighRatio Low/high threshold ratio (0.33 recommended for document scanning)
 * @param emaAlpha EMA smoothing alpha (0.0–1.0)
 * @param thresholdFloor Minimum high threshold
 * @param thresholdCeiling Maximum high threshold (120 recommended for documents)
 */
class CannyThresholdCalculatorV2(
    private val matBundle: IMatBundle,
    private val sobelKsize: Int = 3,
    private val blurSize: Int = 5,
    private val blurSigma: Double = 1.5,
    private val lowHighRatio: Double = 0.33,
    private val emaAlpha: Double = 0.15,
    private val thresholdFloor: Double = 40.0,
    private val thresholdCeiling: Double = 120.0
) : ICannyThresholdCalculator {

    private var smoothedHigh = -1.0

    override fun computeThreshold(grayMat: Mat): Pair<Double, Double> {
        // Step 1: Gaussian blur to suppress paper texture, JPEG artifacts, and sensor noise.
        // Larger kernel (5×5) with σ=1.5 works better for phone cameras than 3×3.
        Imgproc.GaussianBlur(grayMat, matBundle.getTemp(), Size(blurSize.toDouble(), blurSize.toDouble()), blurSigma)

        // Step 2: Compute Sobel gradients in both directions.
        // CV_64F depth required — Sobel outputs signed values that can be negative.
        Imgproc.Sobel(
            matBundle.getTemp(),
            matBundle.getSobelX(),
            CvType.CV_64F,
            1, 0,  // x-direction derivative
            sobelKsize
        )
        Imgproc.Sobel(
            matBundle.getTemp(),
            matBundle.getSobelY(),
            CvType.CV_64F,
            0, 1,  // y-direction derivative
            sobelKsize
        )

        // Step 3: Compute gradient magnitude: mag = √(Gx² + Gy²)
        // This is the edge-strength map that Canny thresholds should operate on.
        Core.magnitude(matBundle.getSobelX(), matBundle.getSobelY(), matBundle.getGradMag())

        // Step 4: Otsu thresholding on the gradient magnitude histogram.
        // This finds the bimodal split between weak edges (texture/noise) and
        // strong edges (document boundaries). Unlike intensity-domain Otsu, this
        // operates in the gradient domain — theoretically correct for Canny.
        val rawOtsu = Imgproc.threshold(
            matBundle.getGradMag(),
            matBundle.getOtsuThreshold(),
            0.0, 255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )

        // Step 5: Confidence-based fallback for extreme cases.
        // rawOtsu < 15 → almost no edges (all-white, camera at wall, underexposed)
        // rawOtsu > 220 → almost all edges (overexposed, extreme contrast)
        // In both cases, Otsu is unreliable — use safe defaults.
        val rawHigh = if (rawOtsu < 15.0 || rawOtsu > 220.0) {
            60.0
        } else {
            rawOtsu.toDouble()
        }
        val rawLow = rawHigh * lowHighRatio

        // Step 6: Temporal EMA smoothing on the high threshold.
        smoothedHigh = if (smoothedHigh < 0.0) {
            rawHigh
        } else {
            (emaAlpha * rawHigh) + ((1.0 - emaAlpha) * smoothedHigh)
        }

        // Step 7: Clamp high to [floor, ceiling] for predictable behavior.
        // Edge quality rarely improves beyond ~120 for document detection.
        val finalHigh = smoothedHigh.coerceIn(thresholdFloor, thresholdCeiling)
        val finalLow = finalHigh * lowHighRatio

        Log.d(
            "CannyV2",
            "grad otsu thresholds: high=$finalHigh low=$finalLow (rawOtsu=$rawOtsu smoothed=$smoothedHigh floor=$thresholdFloor ceiling=$thresholdCeiling)"
        )

        return Pair(finalHigh, finalLow)
    }

    override fun reset() {
        smoothedHigh = -1.0
    }

    override fun release() {
        // No owned resources — MatBundle handles its own cleanup
    }

}
