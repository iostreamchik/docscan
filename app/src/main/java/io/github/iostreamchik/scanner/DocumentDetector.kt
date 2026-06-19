package io.github.iostreamchik.scanner

import android.graphics.Bitmap
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
import java.math.BigDecimal
import java.math.RoundingMode
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
class DocumentDetector(
    private val matBundle: IMatBundle,
    private val params: PipelineParams = PipelineParams.Auto,
    private val thresholdCalculator: ICannyThresholdCalculator? = null
) {

    private val _detectionParams = MutableStateFlow(DetectionParameters())
    val detectionParams = _detectionParams.asStateFlow()

    fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams,
        previews: MutableMap<String, Bitmap?>
    ): Mat {
        val p = params

        // --- Resize + Grayscale ---
        val smallMat = Mat()
        Imgproc.resize(rawMat, smallMat, Size(scaledWidth.toDouble(), scaledHeight.toDouble()))
        Imgproc.cvtColor(smallMat, matBundle.getGray(), Imgproc.COLOR_RGBA2GRAY)
        smallMat.release()
        previews["Grayscale"] = matBundle.getGray().toBitmap().copy(Bitmap.Config.ARGB_8888, false)

        // --- Median Blur ---
        val blurKsize = p.medianBlurKsize
        Imgproc.medianBlur(matBundle.getGray(), matBundle.getBlurred(), blurKsize)
        previews["Median Blur"] = matBundle.getBlurred().toBitmap().copy(Bitmap.Config.ARGB_8888, false)

        // --- CLAHE (auto when params is Auto, user-configured when set) ---
        val claheClipLimit: Double = if (params is PipelineParams.Auto) {
            // Brightness-adaptive: dimmer scenes → stronger contrast enhancement
            val meanVal = Core.mean(matBundle.getBlurred())
            val brightness = meanVal.`val`[0].coerceIn(0.0, 255.0)
            calculateClacheClipLimit(brightness) - 0.1
        } else {
            params.claheClipLimit.toDouble()
        }
        val tileSize = params.claheTileSize.toDouble()
        val clahe = Imgproc.createCLAHE(claheClipLimit, Size(tileSize, tileSize))
        clahe.apply(matBundle.getBlurred(), matBundle.getEnhanced())
        previews["CLAHE"] = matBundle.getEnhanced().toBitmap().copy(Bitmap.Config.ARGB_8888, false)

        // --- Morph Close ---
        Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(p.morphCloseSize.toDouble(), p.morphCloseSize.toDouble())).also { kernel ->
            matBundle.getKernel().release()
            kernel.copyTo(matBundle.getKernel())
        }
        Imgproc.morphologyEx(matBundle.getEnhanced(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, matBundle.getKernel())
        previews["Morph Close"] = matBundle.getMorph().toBitmap().copy(Bitmap.Config.ARGB_8888, false)

        // --- Canny ---
        Imgproc.Canny(matBundle.getEnhanced(), matBundle.getEdges(), p.cannyLow.toDouble(), p.cannyHigh.toDouble())
        previews["Canny Edges"] = matBundle.getEdges().toBitmap().copy(Bitmap.Config.ARGB_8888, false)

        // --- Strong Closing ---
        val scale = max(scaledWidth, scaledHeight).toDouble()
        var closeKsize = (p.strongCloseSize / scale * 640.0).coerceAtLeast(3.0).toInt()
        if (closeKsize % 2 == 0) closeKsize++
        Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(closeKsize.toDouble(), closeKsize.toDouble())).also { k2 ->
            matBundle.getKernel2().release()
            k2.copyTo(matBundle.getKernel2())
        }
        Imgproc.morphologyEx(matBundle.getEdges(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, matBundle.getKernel2())
        previews["Strong Close"] = matBundle.getMorph().toBitmap().copy(Bitmap.Config.ARGB_8888, false)

        // --- Directional Suppression ---
        if (p.directionalKernelSize > 0) {
            val dirKsize = (p.directionalKernelSize / scale * 640.0).coerceAtLeast(3.0).toInt()
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(dirKsize.toDouble(), 1.0)).also { k ->
                matBundle.getHorizontalKernel().release()
                k.copyTo(matBundle.getHorizontalKernel())
            }
            Imgproc.morphologyEx(matBundle.getMorph(), matBundle.getHorizontalClose(), Imgproc.MORPH_CLOSE, matBundle.getHorizontalKernel())

            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(1.0, dirKsize.toDouble())).also { k ->
                matBundle.getVerticalKernel().release()
                k.copyTo(matBundle.getVerticalKernel())
            }
            Imgproc.morphologyEx(matBundle.getHorizontalClose(), matBundle.getVerticalClose(), Imgproc.MORPH_CLOSE, matBundle.getVerticalKernel())

            matBundle.getVerticalClose().copyTo(matBundle.getMorph())
            previews["Directional Suppression"] = matBundle.getMorph().toBitmap().copy(Bitmap.Config.ARGB_8888, false)
        }

        return matBundle.getMorph()
    }

    /**
     * Runs the full image preprocessing pipeline with adaptive CLAHE and
     * context-aware morph close skip logic. Used by the camera detection path.
     */
    internal fun preprocessWithAdaptiveCLAHE(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams,
        useAutoParams: Boolean
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
        val useAutoClahe = params is PipelineParams.Auto
        val claheClipLimit: Double = if (useAutoClahe) {
            // Brightness-adaptive: dimmer scenes → stronger contrast enhancement
            val brightness = avgBrightness.coerceIn(0.0, 255.0)
            calculateClacheClipLimit(brightness).coerceIn(0.5, 4.0)
        } else {
            p.claheClipLimit.toDouble()
        }
        _detectionParams.value = _detectionParams.value.copy(
            claheClipLimit = claheClipLimit.toString()
        )
        Log.d("DocScan", "  CLAHE: clipLimit=${"%.2f".format(claheClipLimit)}, tileSize=${p.claheTileSize}, useAutoClahe=$useAutoClahe")
        val tileSize = (p.claheTileSize).toDouble()
        val clahe = Imgproc.createCLAHE(claheClipLimit, Size(tileSize, tileSize))
        clahe.apply(matBundle.getBlurred(), matBundle.getEnhanced())

        // --- Morph Close (contrast-gated skip) ---
        Core.meanStdDev(matBundle.getEnhanced(), matBundle.getMean(), matBundle.getStd())
        val enhancedContrast = matBundle.getStd().toArray()[0]
        val skipMorphClose = useAutoParams && enhancedContrast < 25.0

        Log.d("DocScan", "  Morph Close: kernel=${p.morphCloseSize}, enhancedContrast=${"%.1f".format(enhancedContrast)}, skip=$skipMorphClose")

        if (skipMorphClose) {
            matBundle.getEnhanced().copyTo(matBundle.getMorph())
        } else {
            Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT,
                Size(p.morphCloseSize.toDouble(), p.morphCloseSize.toDouble())
            ).also { kernel ->
                matBundle.getKernel().release()
                kernel.copyTo(matBundle.getKernel())
            }
            Imgproc.morphologyEx(matBundle.getEnhanced(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, matBundle.getKernel())
        }

        // --- Canny ---
        var (cannyHigh, cannyLow) = if (useAutoParams) {
            thresholdCalculator?.computeThreshold(matBundle.getEnhanced())
                ?: Pair(p.cannyHigh.toDouble(), p.cannyLow.toDouble())
        } else if (p.cannyAutoDetect) {
            thresholdCalculator?.computeThreshold(matBundle.getEnhanced())
                ?: Pair(p.cannyHigh.toDouble(), p.cannyLow.toDouble())
        } else {
            Pair(p.cannyHigh.toDouble(), p.cannyLow.toDouble())
        }

        Log.d("DocScan", "  Canny: low=${"%.0f".format(cannyLow)}, high=${"%.0f".format(cannyHigh)}, autoDetect=${p.cannyAutoDetect}, useAutoParams=$useAutoParams")

        // Auto-fallback: if Otsu produced thresholds > 100, scale down
        if (!useAutoParams && p.cannyAutoDetect && cannyHigh > 100.0) {
            Log.d("DocScan", "  Canny auto-fallback: scaling down high=${"%.0f".format(cannyHigh)} -> 50.0, low=${"%.0f".format(cannyLow)} -> 25.0")
            cannyHigh = 50.0
            cannyLow = cannyHigh * 0.5
        }
        _detectionParams.value = _detectionParams.value.copy(
            cannyHigh = cannyHigh.toInt().toString(),
            cannyLow = cannyLow.toInt().toString()
        )

        Imgproc.Canny(matBundle.getEnhanced(), matBundle.getEdges(), cannyLow, cannyHigh)

        // --- Strong Closing ---
        val scale = max(scaledWidth, scaledHeight).toDouble()
        var closeKsize = (p.strongCloseSize / scale * 640.0).coerceAtLeast(3.0).toInt()
        if (closeKsize % 2 == 0) closeKsize++
        Log.d("DocScan", "  Strong Close: original=${p.strongCloseSize}, computed=$closeKsize")
        Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(closeKsize.toDouble(), closeKsize.toDouble())).also { k2 ->
            matBundle.getKernel2().release()
            k2.copyTo(matBundle.getKernel2())
        }
        Imgproc.morphologyEx(matBundle.getEdges(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, matBundle.getKernel2())

        // --- Directional Suppression ---
        val dirKsize = (p.directionalKernelSize / scale * 640.0).coerceAtLeast(3.0).toInt()
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
        originalHeight: Int
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
            val area = Imgproc.contourArea(contour)
            if (area < minArea) continue

            val hull = matBundle.getHull()
            matBundle.getHullPoints().release()
            matBundle.getHullPoints().create(0, 1, CvType.CV_32FC2)
            val approx = matBundle.getApprox()

            Imgproc.convexHull(contour, hull)
            val contourArray = contour.toArray()
            val hullIndices = IntArray(hull.rows().toInt())
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

        return candidates.maxByOrNull { scoreContourWithParams(it, scaledWidth, scaledHeight, params) }
    }

    fun calculateClacheClipLimit(brightness: Double): Double {
        val raw = BigDecimal(100.0 / brightness)
            .coerceIn(BigDecimal(0.5), BigDecimal(2.0))
            .multiply(BigDecimal(10))
        return (raw.divide(BigDecimal(10), RoundingMode.HALF_UP) - BigDecimal("0.2"))
            .setScale(1, RoundingMode.HALF_DOWN)
            .toDouble().coerceIn(0.5, 2.0)
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Companion object: shared geometry helpers
    // ─────────────────────────────────────────────────────────────────────────────

    companion object {
        /**
         * Validates that a detected quad doesn't fill the entire frame,
         * which indicates a likely false positive (background texture).
         *
         * @return true if the quad passes the size check, false otherwise
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
