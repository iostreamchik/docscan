package io.github.iostreamchik.scanner.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import io.github.iostreamchik.scanner.DocumentDetector
import io.github.iostreamchik.scanner.calculateWarpedDimensions
import io.github.iostreamchik.scanner.fixRotation
import io.github.iostreamchik.scanner.opencv.IMatBundle
import io.github.iostreamchik.scanner.opencv.MatBundle
import io.github.iostreamchik.scanner.opencv.PipelineParams
import io.github.iostreamchik.scanner.sharpen
import io.github.iostreamchik.scanner.sortQuadPoints
import io.github.iostreamchik.scanner.toBitmap
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
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max

/**
 * ViewModel for the Pipeline Settings screen.
 * Processes a picked image through the detection pipeline with adjustable parameters,
 * emitting intermediate preview bitmaps for each stage.
 */
class PipelineSettingsViewModel(
    private val matBundle: IMatBundle = MatBundle(),
    private val detector: DocumentDetector = DocumentDetector(matBundle)
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

    private var _currentParams = MutableStateFlow<PipelineParams?>(null)
    val currentParams: StateFlow<PipelineParams?> = _currentParams.asStateFlow()

    private val _avgBrightness = MutableStateFlow<Double?>(null)
    val avgBrightness: StateFlow<Double?> = _avgBrightness.asStateFlow()

    private val _contrast = MutableStateFlow<Double?>(null)
    val contrast: StateFlow<Double?> = _contrast.asStateFlow()

    private var lastImageUri: Uri? = null

    fun updateParams(newParams: PipelineParams?, context: Context) {
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
    fun updateParamSafely(newParams: PipelineParams?, contextProvider: suspend () -> Context) {
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
        updateParams(null, context)
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

    private suspend fun processWithParams(context: Context, params: PipelineParams?) = withContext(Dispatchers.IO) {
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

        Log.d("PipelineSettings", "=== PipelineSettingsViewModel.processWithParams START ===")
        Log.d("PipelineSettings", "Input: ${originalWidth}x${originalHeight}, maxDim=$maxDim, scale=${"%.4f".format(scale)}, scaled=${scaledWidth}x${scaledHeight}")
        val p = params ?: PipelineParams.Default
        Log.d("PipelineSettings", "Params: medianBlur=${p.medianBlurKsize}, claheClip=${params?.claheClipLimit}, claheTile=${params?.claheTileSize}, morphClose=${p.morphCloseSize}, cannyLow=${p.cannyLow}, cannyHigh=${p.cannyHigh}, strongClose=${p.strongCloseSize}, dirKernel=${p.directionalKernelSize}, approxTol=${p.approxPolyDPTolerance}, minAreaFrac=${p.minAreaFraction}")

        val previews = mutableMapOf<String, Bitmap?>()

        try {
            // Preprocess: resize → grayscale → blur → CLAHE → morph → Canny → strong close → directional suppression
            detector.preprocessWithPreviews(mat, scaledWidth, scaledHeight, params, previews)

            // Capture brightness/contrast from the grayscale step
            Core.meanStdDev(matBundle.getGray(), matBundle.getMean(), matBundle.getStd())
            _avgBrightness.value = matBundle.getMean().toArray()[0]
            _contrast.value = matBundle.getStd().toArray()[0]

            // --- Detect Document ---
            val bestQuad = detector.detectQuad(
                morphImage = matBundle.getMorph(),
                scaledWidth = scaledWidth,
                scaledHeight = scaledHeight,
                originalWidth = originalWidth,
                originalHeight = originalHeight
            )

            if (bestQuad != null) {
                _detectedQuad.value = bestQuad.toArray().toList()
                _hasDetectedDocument.value = true

                val warped = warpDocument(mat, bestQuad)
                // Clone before emitting to StateFlow so Compose gets its own
                // independent copy that won't be affected by recycling in
                // onCleared() or a subsequent processWithParams call.
                val warpedClone = warped.copy(Bitmap.Config.ARGB_8888, false)
                _resultBitmap.value = warpedClone
                previews["Detected Document"] = warpedClone

                // Create quad overlay: original image (scaled) with detected quad drawn on top
                val originalScaled = Mat()
                Imgproc.resize(mat, originalScaled, Size(scaledWidth.toDouble(), scaledHeight.toDouble()))

                val sorted = sortQuadPoints(bestQuad.toArray().toList())
                val scaleX = scaledWidth / originalWidth.toDouble()
                val scaleY = scaledHeight / originalHeight.toDouble()
                val pts = sorted.map { Point(it.x * scaleX, it.y * scaleY) }
                val quadMatPts = MatOfPoint(*pts.toTypedArray())

                // Create a mask of the quad (white inside, black outside)
                val mask = Mat(scaledHeight, scaledWidth, CvType.CV_8UC1, Scalar(0.0))
                Imgproc.fillPoly(mask, listOf(quadMatPts), Scalar(255.0))

                // Create a green-blended version: original * 0.8 + green * 0.2 (masked to quad only)
                val greenOverlay = Mat(scaledHeight, scaledWidth, CvType.CV_8UC4,
                    Scalar(0.0, 255.0, 0.0, 255.0)
                )
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
                Imgproc.drawContours(quadOverlay, listOf(quadMatPts), -1,
                    Scalar(0.0, 255.0, 0.0, 255.0), 3)

                quadMatPts.release()
                mask.release()
                greenOverlay.release()
                greenBlended.release()
                originalMasked.release()
                greenMasked.release()
                originalScaled.release()
                previews["Quad"] = quadOverlay.toBitmap().copy(Bitmap.Config.ARGB_8888, false)
                quadOverlay.release()

                bestQuad.release()

                Log.d("PipelineSettings", "=== PipelineSettingsViewModel.processWithParams SUCCESS: detected quad ===")
            } else {
                _hasDetectedDocument.value = false
                Log.d("PipelineSettings", "=== PipelineSettingsViewModel.processWithParams NO DETECTION ===")
            }

            setPreviewBitmaps(previews)

        } catch (e: Exception) {
            _error.value = "Processing error: ${e.message}"
            Log.e("PipelineSettings", "Error processing", e)
            Log.d("PipelineSettings", "=== PipelineSettingsViewModel.processWithParams ERROR: ${e.message} ===")
        } finally {
            mat.release()
            sourceBitmap.recycle()
            matBundle.releaseAll()
        }
    }

    private fun warpDocument(src: Mat, quad: MatOfPoint): Bitmap {
        val sorted: List<Point> = sortQuadPoints(quad.toArray().toList())
        val tl = sorted[0]
        val tr = sorted[1]
        val br = sorted[2]
        val bl = sorted[3]

        // Calculate proper output dimensions from the document's actual edge lengths.
        // This preserves the document's natural aspect ratio, handling perspective
        // distortion where edges may appear at different lengths.
        val dimensions = calculateWarpedDimensions(tl, tr, br, bl)
        val outputW = dimensions.first
        val outputH = dimensions.second

        val srcPts = MatOfPoint2f(tl, tr, br, bl)
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(outputW.toDouble(), 0.0),
            Point(outputW.toDouble(), outputH.toDouble()),
            Point(0.0, outputH.toDouble())
        )

        val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val output = Mat()
        Imgproc.warpPerspective(src, output, transform, Size(outputW.toDouble(), outputH.toDouble()))

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
