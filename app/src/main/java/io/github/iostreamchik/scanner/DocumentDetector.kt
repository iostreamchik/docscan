package io.github.iostreamchik.scanner

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.geometry.Geometry
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
) : IDocumentDetector {

    private val _detectionParams = MutableStateFlow(DetectionParameters())
    override val detectionParams = _detectionParams.asStateFlow()

    /**
     * Runs the full image preprocessing pipeline with adaptive CLAHE and
     * context-aware morph close skip logic. Used by the camera detection path.
     */
    override fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat {
        Log.d("DocScan", "=== preprocess START: ${scaledWidth}x${scaledHeight} ===")
        // --- Resize + Grayscale ---
        val smallMat = Mat()
        Imgproc.resize(rawMat, smallMat, Size(scaledWidth.toDouble(), scaledHeight.toDouble()))
        Imgproc.cvtColor(smallMat, matBundle.getGray(), Imgproc.COLOR_RGBA2GRAY)
        smallMat.release()

        Core.meanStdDev(matBundle.getGray(), matBundle.getMean(), matBundle.getStd())
        val avgBrightness = matBundle.getMean().toArray()[0]
        val p = params

        Log.d("DocScan", "  Avg brightness: ${"%.1f".format(avgBrightness)}")
        Log.d("DocScan", "  Gray mat: rows=${matBundle.getGray().rows()}, cols=${matBundle.getGray().cols()}, type=${matBundle.getGray().type()}, channels=${matBundle.getGray().channels()}")
        _detectionParams.value = _detectionParams.value.copy(
            brightness = "%.1f".format(avgBrightness)
        )

        // --- Median Blur ---
        val blurKsize = p.medianBlurKsize.coerceAtLeast(3)
        Log.d("DocScan", "  Median Blur: ksize=$blurKsize")
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
        Log.d("DocScan", "  CLAHE done: enhanced type=${matBundle.getEnhanced().type()}, channels=${matBundle.getEnhanced().channels()}")

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
        Log.d("DocScan", "  Morph done: morph type=${matBundle.getMorph().type()}, channels=${matBundle.getMorph().channels()}")

        // Copy the image Canny will actually run on.
        // After morph skip logic, the source is either getMorph() or getEnhanced().
        val cannySource = if (skipMorphClose) matBundle.getEnhanced() else matBundle.getMorph()
        Imgproc.GaussianBlur(cannySource, matBundle.getTemp(), Size(3.0, 3.0), 0.8)
        Log.d("DocScan", "  Pre-Canny Gaussian done: temp type=${matBundle.getTemp().type()}, channels=${matBundle.getTemp().channels()}")

        // --- Canny ---
        // Thresholds are computed from the SAME pre-Canny blurred image that Canny
        // operates on. This ensures Otsu gradient statistics match actual edge strength.
        val (cannyLow, cannyHigh) = if (params.isCannyAuto) {
            Log.d("DocScan", "  [Canny] Calling thresholdCalculator.computeThreshold (Auto mode)")
            thresholdCalculator.computeThreshold(matBundle.getTemp())
        } else if (p.cannyAutoDetect) {
            Log.d("DocScan", "  [Canny] Calling thresholdCalculator.computeThreshold (cannyAutoDetect=true)")
            thresholdCalculator.computeThreshold(matBundle.getTemp())
        } else {
            Log.d("DocScan", "  [Canny] Using manual thresholds: high=${p.cannyHigh}, low=${p.cannyLow}")
            Pair(p.cannyHigh.toDouble(), p.cannyLow.toDouble())
        }

        Log.d("DocScan", "  Canny: low=${"%.0f".format(cannyLow)}, high=${"%.0f".format(cannyHigh)}, autoDetect=${p.cannyAutoDetect}, mode=${if (params.isCannyAuto) "Auto" else "Manual"}")
        _detectionParams.value = _detectionParams.value.copy(
            cannyHigh = cannyHigh.toInt().toString(),
            cannyLow = cannyLow.toInt().toString()
        )

        Imgproc.Canny(matBundle.getTemp(), matBundle.getEdges(), cannyLow, cannyHigh)
        Log.d("DocScan", "  Canny done: edges type=${matBundle.getEdges().type()}, channels=${matBundle.getEdges().channels()}")

        // --- Strong Closing ---
        var closeKsize = p.strongCloseSize.coerceIn(3, 15).toInt()
        if (closeKsize % 2 == 0) closeKsize++
        Log.d("DocScan", "  Strong Close: original=${p.strongCloseSize}, computed=$closeKsize")
        Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(closeKsize.toDouble(), closeKsize.toDouble())).also { k2 ->
            matBundle.getKernel2().release()
            k2.copyTo(matBundle.getKernel2())
        }
        Imgproc.morphologyEx(matBundle.getEdges(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, matBundle.getKernel2())
        Log.d("DocScan", "  Strong Close done: morph type=${matBundle.getMorph().type()}")

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
        Log.d("DocScan", "  Directional Suppression done: final morph type=${matBundle.getMorph().type()}, channels=${matBundle.getMorph().channels()}")

        // Count non-zero pixels in the final morph/edge image
        val nonzeroCount = Core.countNonZero(matBundle.getMorph())
        Log.d("DocScan", "  Final morph: nonzero pixels=$nonzeroCount / ${matBundle.getMorph().total()}, nonzeroRatio=${"%.4f".format(nonzeroCount.toDouble() / matBundle.getMorph().total())}")
        Log.d("DocScan", "=== preprocess END ===")

        return matBundle.getMorph()
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Document Detection (contour finding, quad validation, scoring)
    // ─────────────────────────────────────────────────────────────────────────────

    /**
     * Extract document candidates from a morph/edge Mat.
     * Returns the best quad or null if no document found.
     */
    override fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        params: PipelineParams
    ): MatOfPoint? {
        Log.d("DocScan", "=== detectQuad START ===")
        Log.d("DocScan", "  morphImage: rows=${morphImage.rows()}, cols=${morphImage.cols()}, type=${morphImage.type()}, channels=${morphImage.channels()}")

        val contours = mutableListOf<MatOfPoint>()
        // OpenCV 5 moved contour geometry functions from Imgproc to Geometry,
        // but findContours still lives in Imgproc with the same API.
        Imgproc.findContours(
            morphImage,
            contours,
            matBundle.getHierarchy(),
            Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        Log.d("DocScan", "  findContours: found ${contours.size} contours")

        val frameArea = scaledWidth * scaledHeight
        val minArea = frameArea * params.minAreaFraction
        Log.d("DocScan", "  frameArea=$frameArea, minArea=$minArea, minAreaFraction=${params.minAreaFraction}")

        val candidates = mutableListOf<MatOfPoint>()
        var skippedArea = 0
        var skippedPoints = 0
        var skippedNot4 = 0
        var skippedNotRect = 0
        var skippedSolidity = 0

        // Debug: log top 5 contour areas
        val areaStats = contours.map { abs(Geometry.contourArea(it)) }.sortedDescending().take(5)
        Log.d("DocScan", "  Top contour areas: ${areaStats.joinToString(", ") { "%.0f".format(it) }}")

        for (contour in contours) {
            val area = abs(Geometry.contourArea(contour))
            if (area < minArea) {
                skippedArea++
                continue
            }

            // Early exit: skip contours with very few points (likely noise/text).
            // Skips expensive convexHull + approxPolyDP for tiny contours.
            if (contour.total() < 10) {
                skippedPoints++
                continue
            }

            val hull = matBundle.getHull()
            matBundle.getHullPoints().release()
            matBundle.getHullPoints().create(0, 1, CvType.CV_32FC2)
            val approx = matBundle.getApprox()

            // OpenCV 5.0.0.1: Geometry.convexHull stores hull as CV_32SC2 (2-channel int)
            // with (x,y) coordinate pairs per row when returnPoints=true (the default).
            // We read hull.get(_,_) directly to extract point coordinates.
            Geometry.convexHull(contour, hull)
            val hullCount = hull.rows() * hull.cols()
            val hullData = IntArray(hullCount * hull.channels())
            hull.get(0, 0, hullData)
            val hullPointList = mutableListOf<Point>()
            for (i in hullData.indices step 2) {
                if (i + 1 < hullData.size) {
                    hullPointList.add(Point(hullData[i].toDouble(), hullData[i + 1].toDouble()))
                }
            }
            matBundle.getHullPoints().fromList(hullPointList)

            val hullPtCount = hullPointList.size
            val peri = Geometry.arcLength(matBundle.getHullPoints(), true)
            // Try progressive epsilon values to find a 4-point quadrilateral approximation.
            // Different contours need different simplification levels — a single tolerance
            // doesn't work uniformly across all contour sizes.
            val epsilons = listOf(0.015, 0.025, 0.04, 0.06, 0.10)
            var foundQuad = false
            for (tol in epsilons) {
                val epsilon = tol * peri
                Geometry.approxPolyDP(matBundle.getHullPoints(), approx, epsilon, true)
                Log.d("DocScan", "    approxPolyDP: tol=$tol, epsilon=${"%.2f".format(epsilon)}, approxPts=${approx.total()}")
                if (approx.total() == 4L) {
                    Log.d("DocScan", "    hull: pts=$hullPtCount, peri=${"%.0f".format(peri)}, epsilon=${"%.2f".format(epsilon)} -> QUAD")
                    foundQuad = true
                    break
                }
            }

            if (!foundQuad) {
                Log.d("DocScan", "    SKIPPED(not4): area=${"%.0f".format(area)}, hullPts=$hullPtCount, approxPts=${approx.total()}, peri=${"%.0f".format(peri)}")
                skippedNot4++
                continue
            }

            val scaleX = originalWidth.toDouble() / scaledWidth
            val scaleY = originalHeight.toDouble() / scaledHeight
            val scaledPoints = approx.toArray().map { Point(it.x * scaleX, it.y * scaleY) }
            val quad = MatOfPoint(*scaledPoints.toTypedArray())

            if (!isRectangle(approx)) {
                quad.release()
                skippedNotRect++
                continue
            }

            val scaledArea = area * (scaleX * scaleY)
            val rect = Geometry.boundingRect(quad)
            val solidity = scaledArea / (rect.width * rect.height).toDouble()
            if (solidity < 0.3) {
                quad.release()
                skippedSolidity++
                continue
            }

            Log.d("DocScan", "  CANDIDATE: area=$area, solidity=${"%.2f".format(solidity)}, rect=${rect.width}x${rect.height}, score=${scoreContourWithParams(quad, originalWidth, originalHeight, params)}")
            candidates.add(quad)
        }

        Log.d("DocScan", "  detectQuad summary: totalContours=${contours.size}, candidates=${candidates.size}, skippedArea=$skippedArea, skippedPoints=$skippedPoints, skippedNot4=$skippedNot4, skippedNotRect=$skippedNotRect, skippedSolidity=$skippedSolidity")

        val best = candidates.maxByOrNull { scoreContourWithParams(it, originalWidth, originalHeight, params) }
        // Clone winner before releasing all candidate Mats to avoid native memory leak.
        val result = best?.let { MatOfPoint(*it.toArray()) }
        candidates.forEach { it.release() }
        Log.d("DocScan", "  detectQuad END: result=${if (result != null) "found (${result.total()} pts)" else "null"}")
        return result
    }

    override fun validateQuadSize(
        quad: MatOfPoint,
        originalWidth: Int,
        originalHeight: Int
    ): Boolean {
        val rect = Geometry.boundingRect(quad)
        val quadArea = rect.width * rect.height
        val frameArea = originalWidth * originalHeight
        return quadArea <= frameArea * 0.95
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
            val rect = Geometry.boundingRect(quad)
            val quadArea = rect.width * rect.height
            val frameArea = originalWidth * originalHeight
            return quadArea <= frameArea * 0.95
        }

        /**
         * Checks if a 4-point contour approximates a rectangle by verifying
         * that all interior angles are close to 90°. Tolerance is relaxed to
         * 25° to accommodate perspective distortion from angled document photos.
         */
        fun isRectangle(approx: MatOfPoint2f): Boolean {
            val pts = approx.toArray()
            var maxDeviation = 0.0
            val angles = DoubleArray(4)

            for (i in 0..3) {
                angles[i] = computeAngle(
                    pts[(i + 1) % 4],
                    pts[(i + 3) % 4],
                    pts[i]
                )
                val deviation = abs(90 - angles[i])
                maxDeviation = max(maxDeviation, deviation)
            }

            Log.d("DocScan", "    isRectangle: angles=${angles.joinToString { "%.1f".format(it) }}°, maxDeviation=%.1f° -> ${maxDeviation < 25}", )
            return maxDeviation < 25
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
    val maskThreshold: String = "",
)
