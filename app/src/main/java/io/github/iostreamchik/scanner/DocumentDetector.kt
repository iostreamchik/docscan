package io.github.iostreamchik.scanner

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import io.github.iostreamchik.scanner.opencv.ICannyThresholdCalculator
import io.github.iostreamchik.scanner.opencv.IMatBundle
import io.github.iostreamchik.scanner.opencv.PipelineParams
import io.github.iostreamchik.scanner.opencv.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.lang.Math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Shared document detection logic — image preprocessing, contour finding,
 * hull computation, quad validation, and scoring.
 *
 * Used by both CameraViewModel and PipelineSettingsViewModel to avoid
 * ~200 lines of duplicated pipeline code.
 *
 * @param matBundle Pre-allocated OpenCV matrix pool for zero-allocation processing
 * @param params Pipeline parameters controlling all preprocessing knobs
 * @param thresholdCalculator Optional calculator for auto Canny thresholds (used by camera pipeline)
 */
/** Maximum short-edge dimension for processing (scaled down from original). */
const val PROCESS_WIDTH = 448.0

class DocumentDetector(
    private val matBundle: IMatBundle,
    private val params: PipelineParams = PipelineParams(),
    private val thresholdCalculator: ICannyThresholdCalculator = CannyThresholdCalculator(matBundle)
) {

    private val _detectionParams = MutableStateFlow(DetectionParameters())
    val detectionParams = _detectionParams.asStateFlow()

    /**
     * Runs the full image preprocessing pipeline with adaptive CLAHE and
     * context-aware morph close skip logic. Used by the camera detection path.
     */
    internal fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat {
        // --- Resize + Grayscale ---
        val smallMat = Mat()
        Imgproc.resize(rawMat, smallMat, Size(scaledWidth.toDouble(), scaledHeight.toDouble()))
        Imgproc.cvtColor(smallMat, matBundle.getGray(), Imgproc.COLOR_RGBA2GRAY)
        smallMat.release()

        Core.meanStdDev(matBundle.getGray(), matBundle.getMean(), matBundle.getStd())
        val avgBrightness = matBundle.getMean().toArray()[0]
        val p = params

        Log.d("DocScan", "  Avg brightness: ${"%.1f".format(avgBrightness)}")
        _detectionParams.value = _detectionParams.value.copy(
            brightness = "%.1f".format(avgBrightness)
        )

        // --- Median Blur ---
        val blurKsize = p.medianBlurKsize.coerceAtLeast(3)
        Imgproc.medianBlur(matBundle.getGray(), matBundle.getBlurred(), blurKsize)

        // --- CLAHE (auto when params is Auto, user-configured or brightness-adaptive) ---
        val useAutoClahe = params.isClaheAuto
        val claheClipLimit: Double = if (useAutoClahe) {
            // Brightness-adaptive: both very dim AND very bright scenes need stronger
            // contrast enhancement. Dim scenes lack shadow detail; bright scenes have
            // flat histograms with no natural contrast separation.
            // Sweet spot is mid-range brightness (~80-120) where natural contrast exists.
            val brightness = avgBrightness.coerceIn(20.0, 200.0)
            val dimBoost = if (brightness < 80.0) {
                100.0 / (brightness + 10.0)  // strong boost for dim: up to ~3.6
            } else {
                0.0
            }
            val brightBoost = if (brightness > 130.0) {
                (brightness - 130.0) / 30.0  // ramps 0→1.7 as brightness goes 130→200
            } else {
                0.0
            }
            (1.5 + dimBoost + brightBoost).coerceIn(1.0, 4.0)
        } else {
            p.claheClipLimit.toDouble().coerceIn(1.0, 4.0)
        }
        _detectionParams.value = _detectionParams.value.copy(
            claheClipLimit = claheClipLimit.toString()
        )
        val tileSize = p.claheTileSize.coerceAtLeast(8).toDouble()
        Log.d("DocScan", "  CLAHE: clipLimit=${"%.2f".format(claheClipLimit)}, tileSize=${"%.1f".format(tileSize)}, useAutoClahe=$useAutoClahe")
        val clahe = Imgproc.createCLAHE(claheClipLimit, Size(tileSize, tileSize))
        clahe.apply(matBundle.getBlurred(), matBundle.getEnhanced())

        // --- Morph Close (contrast-gated skip) ---
        Core.meanStdDev(matBundle.getEnhanced(), matBundle.getMean(), matBundle.getStd())
        val enhancedContrast = matBundle.getStd().toArray()[0]
        val skipMorphClose = params.isClaheAuto && enhancedContrast < 25.0

        Log.d("DocScan", "  Morph Close: kernel=${p.morphCloseSize}, enhancedContrast=${"%.1f".format(enhancedContrast)}, skip=$skipMorphClose")

        if (skipMorphClose) {
            matBundle.getEnhanced().copyTo(matBundle.getMorph())
        } else {
            val morphCloseKsize = p.morphCloseSize.coerceAtLeast(3).toDouble()
            Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT,
                Size(morphCloseKsize, morphCloseKsize)
            ).also { kernel ->
                matBundle.getKernel().release()
                kernel.copyTo(matBundle.getKernel())
            }
            Imgproc.morphologyEx(matBundle.getEnhanced(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, matBundle.getKernel())
        }

        // Copy the image Canny will actually run on.
        // After morph skip logic, the source is either getMorph() or getEnhanced().
        val cannySource = if (skipMorphClose) matBundle.getEnhanced() else matBundle.getMorph()
        Imgproc.GaussianBlur(cannySource, matBundle.getTemp(), Size(3.0, 3.0), 0.8)

        // --- Canny ---
        // Thresholds are computed from the SAME pre-Canny blurred image that Canny
        // operates on. This ensures Otsu gradient statistics match actual edge strength.
        val (cannyHigh, cannyLow) = if (params.isCannyAuto) {
            thresholdCalculator.computeThreshold(matBundle.getTemp())
        } else if (p.cannyAutoDetect) {
            thresholdCalculator.computeThreshold(matBundle.getTemp())
        } else {
            Pair(p.cannyHigh.toDouble(), p.cannyLow.toDouble())
        }

        Log.d("DocScan", "  Canny: low=${"%.0f".format(cannyLow)}, high=${"%.0f".format(cannyHigh)}, autoDetect=${p.cannyAutoDetect}, mode=${if (params.isCannyAuto) "Auto" else "Manual"}")
        _detectionParams.value = _detectionParams.value.copy(
            cannyHigh = cannyHigh.toInt().toString(),
            cannyLow = cannyLow.toInt().toString()
        )

        Imgproc.Canny(matBundle.getTemp(), matBundle.getEdges(), cannyLow, cannyHigh)

        // --- Strong Closing ---
        var closeKsize = p.strongCloseSize.coerceIn(3, 15).toInt()
        if (closeKsize % 2 == 0) closeKsize++
        Log.d("DocScan", "  Strong Close: original=${p.strongCloseSize}, computed=$closeKsize")
        Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(closeKsize.toDouble(), closeKsize.toDouble())).also { k2 ->
            matBundle.getKernel2().release()
            k2.copyTo(matBundle.getKernel2())
        }
        Imgproc.morphologyEx(matBundle.getEdges(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, matBundle.getKernel2())

        // --- Directional Suppression ---
        val dirKsize = p.directionalKernelSize.coerceIn(3, 15).toInt()
        Log.d("DocScan", "  Directional Suppression: original=${p.directionalKernelSize}, computed=$dirKsize")
        Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT,
            Size(dirKsize.toDouble(), 1.0)
        ).also { kernel ->
            matBundle.getHorizontalKernel().release()
            kernel.copyTo(matBundle.getHorizontalKernel())
        }
        Imgproc.morphologyEx(
            matBundle.getMorph(),
            matBundle.getHorizontalClose(),
            Imgproc.MORPH_CLOSE,
            matBundle.getHorizontalKernel()
        )

        Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT,
            Size(1.0, dirKsize.toDouble())
        ).also { kernel ->
            matBundle.getVerticalKernel().release()
            kernel.copyTo(matBundle.getVerticalKernel())
        }
        Imgproc.morphologyEx(
            matBundle.getHorizontalClose(),
            matBundle.getVerticalClose(),
            Imgproc.MORPH_CLOSE,
            matBundle.getVerticalKernel()
        )

        matBundle.getVerticalClose().copyTo(matBundle.getMorph())

        return matBundle.getMorph()
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Document Detection (contour finding, quad validation, scoring)
    // ─────────────────────────────────────────────────────────────────────────────

    /**
     * Extract document candidates from a morph/edge Mat.
     * Returns the best quad or null if no document found.
     */
    fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        params: PipelineParams = this.params
    ): MatOfPoint? {
        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(
            morphImage,
            contours,
            matBundle.getHierarchy(),
            Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        val frameArea = scaledWidth * scaledHeight
        val minArea = frameArea * params.minAreaFraction
        val candidates = mutableListOf<MatOfPoint>()

        for (contour in contours) {
            val area = abs(Imgproc.contourArea(contour))
            if (area < minArea) continue

            // Early exit: skip contours with very few points (likely noise/text).
            // Skips expensive convexHull + approxPolyDP for tiny contours.
            if (contour.total() < 10) continue

            val hull = matBundle.getHull()
            matBundle.getHullPoints().release()
            matBundle.getHullPoints().create(0, 1, CvType.CV_32FC2)
            val approx = matBundle.getApprox()

            Imgproc.convexHull(contour, hull)
            val contourArray = contour.toArray()
            val hullIndices = IntArray(hull.rows())
            hull.get(0, 0, hullIndices)
            val hullPointList = hullIndices.map { contourArray[it] }
            matBundle.getHullPoints().fromList(hullPointList.map { Point(it.x, it.y) })

            val peri = Imgproc.arcLength(matBundle.getHullPoints(), true)
            Imgproc.approxPolyDP(
                matBundle.getHullPoints(),
                approx,
                params.approxPolyDPTolerance * peri,
                true
            )

            if (approx.total() != 4L) continue

            val scaleX = originalWidth.toDouble() / scaledWidth
            val scaleY = originalHeight.toDouble() / scaledHeight
            val scaledPoints = approx.toArray().map { Point(it.x * scaleX, it.y * scaleY) }
            val quad = MatOfPoint(*scaledPoints.toTypedArray())

            if (!isRectangle(approx)) {
                quad.release()
                continue
            }

            val scaledArea = area * (scaleX * scaleY)
            val rect = Imgproc.boundingRect(quad)
            val solidity = scaledArea / (rect.width * rect.height).toDouble()
            if (solidity < 0.3) {
                quad.release()
                continue
            }

            candidates.add(quad)
        }

        val best = candidates.maxByOrNull { scoreContourWithParams(it, originalWidth, originalHeight, params) }
        // Clone winner before releasing all candidate Mats to avoid native memory leak.
        val result = best?.let { MatOfPoint(*it.toArray()) }
        candidates.forEach { it.release() }
        return result
    }

    companion object {
        /**
         * Computes the auto CLAHE clip limit for a given average brightness.
         * Both very dim and very bright scenes need stronger contrast enhancement.
         * Sweet spot is mid-range brightness (~80-120) where natural contrast exists.
         */
        fun computeAutoClaheClipLimit(avgBrightness: Double): Double {
            val brightness = avgBrightness.coerceIn(20.0, 200.0)
            val dimBoost = if (brightness < 80.0) {
                100.0 / (brightness + 10.0)
            } else {
                0.0
            }
            val brightBoost = if (brightness > 130.0) {
                (brightness - 130.0) / 30.0
            } else {
                0.0
            }
            return (1.5 + dimBoost + brightBoost).coerceIn(1.0, 4.0)
        }

        /**
         * Validates that a detected quad doesn't fill the entire frame,
         * which indicates a likely false positive (background texture).
         */
        fun validateQuadSize(
            quad: MatOfPoint,
            originalWidth: Int,
            originalHeight: Int
        ): Boolean {
            val rect = Imgproc.boundingRect(quad)
            val quadArea = rect.width * rect.height
            val frameArea = originalWidth * originalHeight
            return quadArea <= frameArea * 0.95
        }

        /**
         * Checks if a 4-point contour approximates a rectangle by verifying
         * that all interior angles are close to 90°.
         */
        fun isRectangle(approx: MatOfPoint2f): Boolean {
            val pts = approx.toArray()
            var maxDeviation = 0.0

            for (i in 0..3) {
                val angle = computeAngle(
                    pts[(i + 1) % 4],
                    pts[(i + 3) % 4],
                    pts[i]
                )
                maxDeviation = max(maxDeviation, abs(90 - angle))
            }

            return maxDeviation < 15
        }

        /**
         * Computes the interior angle (in degrees) between three points at a vertex.
         */
        fun computeAngle(p1: Point, p2: Point, center: Point): Double {
            val dx1 = p1.x - center.x
            val dy1 = p1.y - center.y
            val dx2 = p2.x - center.x
            val dy2 = p2.y - center.y

            val dot = dx1 * dx2 + dy1 * dy2
            val norm1 = sqrt(dx1 * dx1 + dy1 * dy1)
            val norm2 = sqrt(dx2 * dx2 + dy2 * dy2)

            return acos(dot / (norm1 * norm2)) * 180.0 / PI
        }
    }


}

data class DetectionParameters(
    val claheClipLimit: String = "",
    val cannyHigh: String = "",
    val cannyLow: String = "",
    val brightness: String = "",
)
