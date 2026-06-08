package io.github.iostreamchik.scanner

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import io.github.iostreamchik.scanner.opencv.IMatBundle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.lang.Math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.sqrt
import io.github.iostreamchik.scanner.opencv.MatBundle

/**
 * Data class holding all adjustable pipeline parameters.
 */
data class PipelineParams(
    // Blur
    val medianBlurKsize: Int = 3,
    val gaussianSigma: Double = 1.0,

    // CLAHE
    val claheClipLimit: Float = 0.8f,
    val claheTileSize: Int = 16,

    // Morph Close (pre-Canny)
    val morphCloseSize: Int = 9,

    // Canny
    val cannyLow: Float = 20f,
    val cannyHigh: Float = 60f,

    // Strong Closing (post-Canny)
    val strongCloseSize: Int = 5,

    // Directional Suppression
    val directionalKernelSize: Int = 15,

    // Contour Detection
    val approxPolyDPTolerance: Float = 0.015f,
    val minAreaFraction: Float = 0.025f,

    // Scoring weights
    val scoreAreaWeight: Float = 0.5f,
    val scoreCenterWeight: Float = 0.3f,
    val scoreAreaRatioWeight: Float = 0.2f
) {
    companion object {
        val Default = PipelineParams()
    }
}

/**
 * ViewModel for the Pipeline Settings screen.
 * Processes a picked image through the detection pipeline with adjustable parameters,
 * emitting intermediate preview bitmaps for each stage.
 */
class PipelineSettingsViewModel(
    private val matBundle: IMatBundle = MatBundle()
) : ViewModel() {

    private val _originalBitmap = MutableStateFlow<Bitmap?>(null)
    val originalBitmap: StateFlow<Bitmap?> = _originalBitmap.asStateFlow()

    private val _previewBitmaps = MutableStateFlow<Map<String, Bitmap?>>(emptyMap())
    val previewBitmaps: StateFlow<Map<String, Bitmap?>> = _previewBitmaps.asStateFlow()

    private val _resultBitmap = MutableStateFlow<Bitmap?>(null)
    val resultBitmap: StateFlow<Bitmap?> = _resultBitmap.asStateFlow()

    private val _detectedQuad = MutableStateFlow<List<Point>?>(null)
    val detectedQuad: StateFlow<List<Point>?> = _detectedQuad.asStateFlow()

    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    private val _hasDetectedDocument = MutableStateFlow(false)
    val hasDetectedDocument: StateFlow<Boolean> = _hasDetectedDocument.asStateFlow()

    private var _currentParams = MutableStateFlow(PipelineParams.Default)
    val currentParams: StateFlow<PipelineParams> = _currentParams.asStateFlow()

    private val _avgBrightness = MutableStateFlow<Double?>(null)
    val avgBrightness: StateFlow<Double?> = _avgBrightness.asStateFlow()

    private val _contrast = MutableStateFlow<Double?>(null)
    val contrast: StateFlow<Double?> = _contrast.asStateFlow()

    private var lastImageUri: Uri? = null

    fun updateParams(newParams: PipelineParams, context: Context) {
        _currentParams.value = newParams
        if (lastImageUri != null) {
            viewModelScope.launch {
                _isProcessing.value = true
                try {
                    processWithParams(context, newParams)
                } catch (e: Exception) {
                    _error.value = "Processing error: ${e.message}"
                } finally {
                    _isProcessing.value = false
                }
            }
        }
    }

    /**
     * Update individual parameters with debounce — the UI layer delays the call
     * for 300ms after the last change before invoking this method.
     */
    fun updateParamSafely(newParams: PipelineParams, contextProvider: suspend () -> Context) {
        _currentParams.value = newParams
        if (lastImageUri != null) {
            viewModelScope.launch {
                _isProcessing.value = true
                try {
                    processWithParams(contextProvider(), newParams)
                } catch (e: Exception) {
                    _error.value = "Processing error: ${e.message}"
                } finally {
                    _isProcessing.value = false
                }
            }
        }
    }

    /**
     * Reset all parameters to defaults and reprocess.
     */
    fun resetParams(context: Context) {
        updateParams(PipelineParams.Default, context)
    }

    /**
     * Replace preview bitmaps with the new set.
     * Old bitmaps are NOT recycled here to avoid a race condition with Compose:
     * if we recycle while Compose is still reading the old refs during composition,
     * we get "Canvas: trying to use a recycled bitmap" crashes.
     * Old bitmaps are naturally collected by GC once Compose releases its references.
     */
    private fun setPreviewBitmaps(newPreviews: Map<String, Bitmap?>) {
        _previewBitmaps.value = newPreviews
    }

    fun loadImage(context: Context, uri: Uri) {
        viewModelScope.launch {
            _isProcessing.value = true
            _error.value = null
            try {
                val sourceBitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, uri)) { decoder, _, _ ->
                        decoder.isMutableRequired = true
                    }
                } else {
                    MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
                }

                // Don't recycle the old bitmap here — let GC handle it to avoid a race
                // condition with Compose composition (same issue as setPreviewBitmaps).
                _originalBitmap.value = sourceBitmap.copy(Bitmap.Config.ARGB_8888, false)

                lastImageUri = uri
                processWithParams(context, _currentParams.value)
            } catch (e: Exception) {
                _error.value = "Failed to load image: ${e.message}"
                Log.e("PipelineSettings", "Error loading image", e)
            } finally {
                _isProcessing.value = false
            }
        }
    }

    private suspend fun processWithParams(context: Context, params: PipelineParams) = withContext(Dispatchers.IO) {
        val uri = lastImageUri ?: return@withContext
        val sourceBitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, uri)) { decoder, _, _ ->
                decoder.isMutableRequired = true
            }
        } else {
            MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
        }

        val mat = Mat()
        Utils.bitmapToMat(sourceBitmap, mat)

        val originalWidth = mat.cols()
        val originalHeight = mat.rows()
        val maxDim = max(originalWidth, originalHeight)
        val scale = 640.0 / maxDim
        val scaledWidth = (originalWidth * scale).toInt()
        val scaledHeight = (originalHeight * scale).toInt()

        val previews = mutableMapOf<String, Bitmap?>()

        try {
            val smallMat = Mat()
            Imgproc.resize(mat, smallMat, Size(scaledWidth.toDouble(), scaledHeight.toDouble()))

            // --- Step 1: Grayscale ---
            Imgproc.cvtColor(smallMat, matBundle.getGray(), Imgproc.COLOR_RGBA2GRAY)
            previews["Grayscale"] = matBundle.getGray().toBitmap()
            Core.meanStdDev(matBundle.getGray(), matBundle.getMean(), matBundle.getStd())
            val avgBrightness = matBundle.getMean().toArray()[0]
            val contrast = matBundle.getStd().toArray()[0]
            _avgBrightness.value = avgBrightness
            _contrast.value = contrast
            smallMat.release()

            // --- Step 2: Median Blur ---
            Imgproc.medianBlur(matBundle.getGray(), matBundle.getBlurred(), params.medianBlurKsize.coerceAtLeast(3))
            previews["Median Blur"] = matBundle.getBlurred().toBitmap()

            // --- Step 3: CLAHE ---
            val clahe = Imgproc.createCLAHE(params.claheClipLimit.toDouble(), Size(params.claheTileSize.toDouble(), params.claheTileSize.toDouble()))
            clahe.apply(matBundle.getBlurred(), matBundle.getEnhanced())
            previews["CLAHE"] = matBundle.getEnhanced().toBitmap()

            // --- Step 4: Morph Close ---
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(params.morphCloseSize.toDouble(), params.morphCloseSize.toDouble())).also { kernel ->
                matBundle.getKernel().release()
                kernel.copyTo(matBundle.getKernel())
            }
            Imgproc.morphologyEx(matBundle.getEnhanced(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, matBundle.getKernel())
            previews["Morph Close"] = matBundle.getMorph().toBitmap()

            // --- Step 5: Canny ---
            Imgproc.Canny(matBundle.getEnhanced(), matBundle.getEdges(), params.cannyLow.toDouble(), params.cannyHigh.toDouble())
            previews["Canny Edges"] = matBundle.getEdges().toBitmap()

            // --- Step 6: Strong Closing ---
            var closeKsize = (params.strongCloseSize * scale).coerceAtLeast(3.0).toInt()
            if (closeKsize % 2 == 0) closeKsize++
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(closeKsize.toDouble(), closeKsize.toDouble())).also { k2 ->
                matBundle.getKernel2().release()
                k2.copyTo(matBundle.getKernel2())
            }
            Imgproc.morphologyEx(matBundle.getEdges(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, matBundle.getKernel2())
            previews["Strong Close"] = matBundle.getMorph().toBitmap()

            // --- Step 7: Directional Suppression ---
            if (params.directionalKernelSize > 0) {
                val dirKsize = (params.directionalKernelSize * scale).coerceAtLeast(3.0).toInt()
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
                previews["Directional Suppression"] = matBundle.getMorph().toBitmap()
            }

            // --- Step 8: Find Contours & Detect Document ---
            val contours = mutableListOf<MatOfPoint>()
            Imgproc.findContours(
                matBundle.getMorph(),
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

            if (candidates.isNotEmpty()) {
                val best = candidates.maxByOrNull { contour ->
                    scoreContour(contour, scaledWidth, scaledHeight, params)
                }

                if (best != null) {
                    _detectedQuad.value = best.toArray().toList()
                    _hasDetectedDocument.value = true

                    val warped = warpDocument(mat, best)
                    _resultBitmap.value = warped
                    previews["Detected Document"] = warped

                    // Create quad overlay: original image (scaled) with detected quad drawn on top
                    val originalScaled = Mat()
                    Imgproc.resize(mat, originalScaled, Size(scaledWidth.toDouble(), scaledHeight.toDouble()))

                    val sorted = sortQuadPoints(best.toArray().toList())
                    val scaleX = scaledWidth / originalWidth.toDouble()
                    val scaleY = scaledHeight / originalHeight.toDouble()
                    val pts = sorted.map { Point(it.x * scaleX, it.y * scaleY) }
                    val quadMatPts = MatOfPoint(*pts.toTypedArray())

                    // Create a mask of the quad (white inside, black outside)
                    val mask = Mat(scaledHeight, scaledWidth, CvType.CV_8UC1, org.opencv.core.Scalar(0.0))
                    Imgproc.fillPoly(mask, listOf(quadMatPts), org.opencv.core.Scalar(255.0))

                    // Create a green-blended version: original * 0.8 + green * 0.2 (masked to quad only)
                    val greenOverlay = Mat(scaledHeight, scaledWidth, CvType.CV_8UC4, org.opencv.core.Scalar(0.0, 255.0, 0.0, 255.0))
                    val greenBlended = Mat(scaledHeight, scaledWidth, CvType.CV_8UC4)
                    val originalMasked = Mat(scaledHeight, scaledWidth, CvType.CV_8UC4)
                    val greenMasked = Mat(scaledHeight, scaledWidth, CvType.CV_8UC4)

                    originalScaled.copyTo(originalMasked, mask)
                    greenOverlay.copyTo(greenMasked, mask)
                    Core.addWeighted(originalMasked, 0.8, greenMasked, 0.2, 0.0, greenBlended)

                    // Start with full original image as base
                    val quadOverlay = Mat(scaledHeight, scaledWidth, CvType.CV_8UC4)
                    originalScaled.copyTo(quadOverlay)

                    // Keep original outside quad, green-blended inside quad
                    val invMask = Mat()
                    Core.bitwise_not(mask, invMask)
                    val originalOutside = Mat()
                    val greenInside = Mat()
                    Core.bitwise_and(quadOverlay, quadOverlay, originalOutside, invMask)
                    Core.bitwise_and(greenBlended, greenBlended, greenInside, mask)
                    Core.add(originalOutside, greenInside, quadOverlay)

                    // Draw solid green outline on top
                    Imgproc.drawContours(quadOverlay, listOf(quadMatPts), -1, org.opencv.core.Scalar(0.0, 255.0, 0.0, 255.0), 3)

                    quadMatPts.release()
                    mask.release()
                    greenOverlay.release()
                    greenBlended.release()
                    originalMasked.release()
                    greenMasked.release()
                    originalScaled.release()
                    previews["Quad"] = quadOverlay.toBitmap()
                    quadOverlay.release()

                    best.release()
                }
            }

            candidates.forEach { it.release() }
            contours.forEach { it.release() }

            if (candidates.isEmpty()) {
                _hasDetectedDocument.value = false
            }

            setPreviewBitmaps(previews)

        } catch (e: Exception) {
            _error.value = "Processing error: ${e.message}"
            Log.e("PipelineSettings", "Error processing", e)
        } finally {
            mat.release()
            sourceBitmap.recycle()
            matBundle.releaseAll()
        }
    }

    private fun isRectangle(approx: MatOfPoint2f): Boolean {
        val pts = approx.toArray()
        var maxDev = 0.0
        for (i in 0..3) {
            val angle = computeAngle(pts[(i + 1) % 4], pts[(i + 3) % 4], pts[i])
            maxDev = max(maxDev, abs(90.0 - angle))
        }
        return maxDev < 50.0
    }

    private fun computeAngle(p1: Point, p2: Point, center: Point): Double {
        val dx1 = p1.x - center.x
        val dy1 = p1.y - center.y
        val dx2 = p2.x - center.x
        val dy2 = p2.y - center.y
        val dot = dx1 * dx2 + dy1 * dy2
        val n1 = sqrt(dx1 * dx1 + dy1 * dy1)
        val n2 = sqrt(dx2 * dx2 + dy2 * dy2)
        return acos(dot / (n1 * n2)) * 180.0 / PI
    }

    private fun sortQuadPoints(points: List<Point>): List<Point> {
        if (points.size != 4) return emptyList()
        val cx = points.sumOf { it.x } / 4.0
        val cy = points.sumOf { it.y } / 4.0
        val sorted = points.sortedBy { atan2(it.y - cy, it.x - cx) }
        val tlp = sorted.mapIndexed { i, p -> i to (p.x + p.y) }.minBy { it.second }.first
        return List(4) { i -> sorted[(tlp + i) % 4] }
    }

    private fun scoreContour(
        contour: MatOfPoint,
        width: Int,
        height: Int,
        params: PipelineParams
    ): Double {
        val area = Imgproc.contourArea(contour)
        val center = Imgproc.boundingRect(contour).let {
            Point(it.x + it.width / 2.0, it.y + it.height / 2.0)
        }
        val frameCenter = Point(width / 2.0, height / 2.0)
        val centerDist = sqrt((center.x - frameCenter.x) * (center.x - frameCenter.x) + (center.y - frameCenter.y) * (center.y - frameCenter.y))
        val maxDist = sqrt((width / 2.0) * (width / 2.0) + (height / 2.0) * (height / 2.0))
        val centerScore = 1.0 - (centerDist / maxDist)
        val frameArea = width * height.toDouble()
        val areaRatio = area / frameArea
        val areaRatioScore = if (areaRatio > 0.5) 0.2 else 1.0

        return area * params.scoreAreaWeight + centerScore * params.scoreCenterWeight * width * height + areaRatioScore * params.scoreAreaRatioWeight * width * height
    }

    private fun warpDocument(src: Mat, quad: MatOfPoint): Bitmap {
        val sorted = sortQuadPoints(quad.toArray().toList())
        val (tl, tr, br, bl) = sorted
        val outputW = src.cols().toDouble()
        val outputH = src.rows().toDouble()

        val srcPts = MatOfPoint2f(tl, tr, br, bl)
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(outputW, 0.0),
            Point(outputW, outputH),
            Point(0.0, outputH)
        )

        val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val output = Mat()
        Imgproc.warpPerspective(src, output, transform, Size(outputW, outputH))

        return output.fixRotation(0).sharpen().toBitmap().also {
            output.release()
        }
    }

    override fun onCleared() {
        super.onCleared()
        _originalBitmap.value?.recycle()
        _resultBitmap.value?.recycle()
        _previewBitmaps.value.values.forEach { it?.recycle() }
        matBundle.releaseAll()
    }
}
