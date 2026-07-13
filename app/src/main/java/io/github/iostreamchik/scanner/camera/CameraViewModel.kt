package io.github.iostreamchik.scanner.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.viewModelScope
import io.github.iostreamchik.scanner.detector.DetectionParameters
import io.github.iostreamchik.scanner.detector.IDocumentDetector
import io.github.iostreamchik.scanner.detector.OnnxDocumentDetector
import io.github.iostreamchik.scanner.detector.PROCESS_WIDTH
import io.github.iostreamchik.scanner.enhanceDocument
import io.github.iostreamchik.scanner.fixRotation
import io.github.iostreamchik.scanner.opencv.CannyThresholdCalculatorV3
import io.github.iostreamchik.scanner.opencv.ICannyThresholdCalculator
import io.github.iostreamchik.scanner.opencv.IMatBundle
import io.github.iostreamchik.scanner.opencv.MatBundle
import io.github.iostreamchik.scanner.opencv.PipelineParams
import io.github.iostreamchik.scanner.quadDistance
import io.github.iostreamchik.scanner.quadHash
import io.github.iostreamchik.scanner.sortQuadPoints
import io.github.iostreamchik.scanner.toBitmap
import io.github.iostreamchik.scanner.toMatRGBA
import io.github.iostreamchik.scanner.toSortedQuad
import io.github.iostreamchik.scanner.warpDocumentHighQuality
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Size
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.max
import kotlin.time.Duration.Companion.milliseconds

class CameraViewModel(
    val matBundle: IMatBundle = MatBundle(),
    val thresholdCalculator: ICannyThresholdCalculator = CannyThresholdCalculatorV3(matBundle),
    val detector: IDocumentDetector
) : androidx.lifecycle.ViewModel() {



    val detectionParams: StateFlow<DetectionParameters> = detector.detectionParams ?: MutableStateFlow(DetectionParameters()).asStateFlow()

    val cameraExecutor = Executors.newSingleThreadExecutor()

    private val quadHistory = ArrayDeque<MatOfPoint>()
    private var lastFrameSize: Size? = null
    private val MAX_HISTORY = 10
    private var frameCounter = 0
    private val STABILITY_CHECK_INTERVAL = 3

    private var lastWarpedQuadHash: Long = 0
    private var lastWarpedBitmap: Bitmap? = null

    private val _blurBitmap = MutableStateFlow<Bitmap?>(null)
    val blurBitmap = _blurBitmap.asStateFlow()

    private val _claheBitmap = MutableStateFlow<Bitmap?>(null)
    val claheBitmap = _claheBitmap.asStateFlow()

    private val _morphBitmap = MutableStateFlow<Bitmap?>(null)
    val morphBitmap = _morphBitmap.asStateFlow()

    private val _filteredBitmap = MutableStateFlow<Bitmap?>(null)
    val filteredBitmap = _filteredBitmap.asStateFlow()

    private val _onnxMaskBitmap = MutableStateFlow<Bitmap?>(null)
    val onnxMaskBitmap = _onnxMaskBitmap.asStateFlow()

    private val _originalBitmap = MutableStateFlow<Bitmap?>(null)
    val originalBitmap = _originalBitmap.asStateFlow()

    private val _resultBitmap = MutableStateFlow<Bitmap?>(null)
    val resultBitmap = _resultBitmap.asStateFlow()

    // Torch state
    private val _torchOn = MutableStateFlow(false)
    val torchOn: StateFlow<Boolean> = _torchOn.asStateFlow()

    fun setTorchOpposite(value: Boolean) {
        _torchOn.value = value
    }

    fun toggleTorch() {
        setTorchOpposite(!_torchOn.value)
    }

    private val _exposureStateFlow = MutableStateFlow("")
    val exposureStateFlow = _exposureStateFlow.asStateFlow()

    private val _errorState = MutableStateFlow<String?>(null)
    val errorState = _errorState.asStateFlow()

    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()

    // Store last picked URI for reprocessing
    private var lastPickedUri: Uri? = null

    fun setError(message: String?) {
        _errorState.value = message
    }

    // Pipeline parameters — reads from pipelineConfigurationManager
    private val _pipelineParams = MutableStateFlow<PipelineParams>(PipelineParams())
    val pipelineParams = _pipelineParams.asStateFlow()

    /**
     * Update pipeline parameters — called from FileScanResultScreen when parameters change.
     */
    fun updateParams(newParams: PipelineParams) {
        _pipelineParams.value = newParams
    }

    fun processFrame(imageProxy: ImageProxy): List<MatOfPoint> {
        val width = imageProxy.width
        val height = imageProxy.height
        lastFrameSize = Size(width.toDouble(), height.toDouble())

        val mat = imageProxy.toMatRGBA()
        val rotation = imageProxy.imageInfo.rotationDegrees

        val result = runDetection(mat, rotation, _pipelineParams.value)

        if (result.isNotEmpty()) {
            if (isStable()) {
                val fusedQuad = getFusedQuad()
                if (fusedQuad != null) {
                    // Always add detection results to history — never add the fused quad
                    // itself, which would saturate the history with near-duplicate values
                    // and prevent getFusedQuad() from responding to new detections.
                    result.forEach { updateHistory(it) }
                    val quadHash = quadHash(fusedQuad)

                    // fusedQuad is already in original resolution (detectQuad scales up),
                    // so use it directly — no need to re-scale.
                    Log.d(
                        "CameraViewModel",
                        "\nFused Quad: ${fusedQuad.toArray().joinToString(", ")}"
                    )
                    // Warp only if the quad has changed — skip expensive op when hash matches
                    val warped = if (quadHash != lastWarpedQuadHash) {
                        warpDocumentHighQuality(mat, fusedQuad, rotation).also {
                            lastWarpedBitmap?.recycle()
                            lastWarpedBitmap = it
                            lastWarpedQuadHash = quadHash
                        }
                    } else {
                        lastWarpedBitmap
                    }
                    // Clone the bitmap before emitting to state flow so Compose gets
                    // its own independent copy that won't be affected by recycling.
                    _resultBitmap.value =
                        warped?.copy(Bitmap.Config.ARGB_8888, false)
                    // Clone: fusedQuad lives in quadHistory, caller owns the clone
                    return listOf(MatOfPoint(*fusedQuad.toArray()))
                }
            } else {
                // Accumulate during pre-stability bootstrapping
                result.forEach { updateHistory(it) }
            }
        }
        // Clone each result: objects also live in quadHistory, caller owns the clones
        return result.map { MatOfPoint(*it.toArray()) }
    }

    private fun updateHistory(quad: MatOfPoint) {
        if (quadHistory.size >= MAX_HISTORY) {
            quadHistory.removeFirst().release()
        }
        quadHistory.addLast(quad)
    }

    private fun isStable(): Boolean {
        if (quadHistory.size < MAX_HISTORY) return false
        if (++frameCounter % STABILITY_CHECK_INTERVAL != 0) return true

        val frameSize = lastFrameSize ?: return false
        val quads = quadHistory.toList()

        var totalMovement = 0.0
        var validPairs = 0

        for (i in 1 until quads.size) {
            val prevSorted = quads[i - 1].toSortedQuad()
            val currSorted = quads[i].toSortedQuad()

            // Skip if either quad is invalid
            if (prevSorted.isEmpty() || currSorted.isEmpty()) continue

            totalMovement += quadDistance(
                prevSorted,
                currSorted,
                frameSize.width,
                frameSize.height
            )
            validPairs++
        }

        // Need at least one valid pair to calculate stability
        return validPairs > 0 && (totalMovement / validPairs) < 0.02
    }

    private fun getFusedQuad(): MatOfPoint? {
        if (quadHistory.isEmpty()) return null

        val validSortedQuads = quadHistory.mapNotNull { matOfPoint ->
            val points = matOfPoint.toArray().toList()
            if (points.size == 4) sortQuadPoints(points) else null
        }

        // Need at least some valid quads to fuse
        if (validSortedQuads.isEmpty()) return null

        val averaged = arrayOf(Point(0.0, 0.0), Point(0.0, 0.0), Point(0.0, 0.0), Point(0.0, 0.0))

        for (i in 0..3) {
            for (quad in validSortedQuads) {
                averaged[i].x += quad[i].x
                averaged[i].y += quad[i].y
            }
            averaged[i].x /= validSortedQuads.size
            averaged[i].y /= validSortedQuads.size
        }

        return MatOfPoint(*averaged)
    }

    /**
     * Shared detection pipeline — delegates preprocessing to [DocumentDetector]
     * and handles quad detection + size validation. Reduced from ~120 lines to ~30.
     */
    private fun runDetection(
        mat: Mat,
        rotation: Int,
        params: PipelineParams
    ): List<MatOfPoint> {
        val originalWidth = mat.cols()
        val originalHeight = mat.rows()
        val maxDimension = max(originalWidth, originalHeight)
        val scale = PROCESS_WIDTH / maxDimension
        val scaledWidth = (originalWidth * scale).toInt()
        val scaledHeight = (originalHeight * scale).toInt()

        Log.d("CameraViewModel", "=== runDetection START: ${originalWidth}x${originalHeight} -> scaled=${scaledWidth}x${scaledHeight} ===")

        // Capture original frame before any processing.
        val originalFrame = mat.fixRotation(rotation).toBitmap()
            .copy(Bitmap.Config.ARGB_8888, false)
        _originalBitmap.value = originalFrame

        try {
            // Preprocess: resize → grayscale → blur → CLAHE → morph → Canny → strong close → directional suppression
            detector.preprocess(mat, scaledWidth, scaledHeight, params)

            // Capture intermediate stage previews before releaseAll().
            // Clone before emitting to StateFlow so Compose gets its own independent copy.
            if (detector is OnnxDocumentDetector) {
                // ONNX detector skips the classical pipeline — no blur/CLAHE/morph stages.
                // Show the original frame as the baseline, and the binary mask separately.
                val maskMat = matBundle.getMorph().clone()
                val maskBitmap = maskMat.fixRotation(rotation).toBitmap()
                    .copy(Bitmap.Config.ARGB_8888, false)
                _onnxMaskBitmap.value = maskBitmap
                _filteredBitmap.value = originalFrame
                maskMat.release()
            } else {
                _blurBitmap.value = matBundle.getBlurred().fixRotation(rotation).toBitmap()
                    .copy(Bitmap.Config.ARGB_8888, false)
                _claheBitmap.value = matBundle.getEnhanced().fixRotation(rotation).toBitmap()
                    .copy(Bitmap.Config.ARGB_8888, false)
                _morphBitmap.value = matBundle.getMorph().fixRotation(rotation).toBitmap()
                    .copy(Bitmap.Config.ARGB_8888, false)
                _filteredBitmap.value = _morphBitmap.value
            }

            // Detect document
            val morphBeforeDetect = matBundle.getMorph()
            Log.d("CameraViewModel", "  Calling detectQuad: morph=${morphBeforeDetect.rows()}x${morphBeforeDetect.cols()}, type=${morphBeforeDetect.type()}, nonzero=${org.opencv.core.Core.countNonZero(morphBeforeDetect)}")
            val bestQuad = detector.detectQuad(
                morphImage = morphBeforeDetect,
                scaledWidth = scaledWidth,
                scaledHeight = scaledHeight,
                originalWidth = originalWidth,
                originalHeight = originalHeight,
                params = params
            )

            if (bestQuad == null) {
                Log.d("CameraViewModel", "  runDetection: NO QUAD DETECTED")
                return emptyList()
            }
            Log.d("CameraViewModel", "  runDetection: bestQuad found with ${bestQuad.total()} points")

            // Validate document size
            if (!detector.validateQuadSize(bestQuad, originalWidth, originalHeight)) {
                bestQuad.release()
                return emptyList()
            }

            return listOf(bestQuad)

        } catch (e: Exception) {
            e.printStackTrace()
            return emptyList()
        } finally {
            matBundle.releaseAll()
        }
    }

    fun processPickedDocument(context: Context, uri: Uri, onScanComplete: () -> Unit) {
        _isProcessing.value = true
        viewModelScope.launch(Dispatchers.IO) {
            try {
                lastPickedUri = uri
                // Clear stale intermediate stage bitmaps from camera scanner
                // Don't recycle — the new value is a clone, so the old one can be
                // safely discarded without recycling to avoid a race condition
                // with Compose composition.
                _blurBitmap.value = null
                _claheBitmap.value = null
                _morphBitmap.value = null
                _filteredBitmap.value = null

                val sourceBitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    ImageDecoder.decodeBitmap(
                        ImageDecoder.createSource(
                            context.contentResolver,
                            uri
                        )
                    ) { decoder, _, _ ->
                        decoder.isMutableRequired = true
                    }
                } else {
                    MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
                }

                // Capture original before processing
                // Don't recycle — let GC handle it to avoid race condition with Compose.
                _originalBitmap.value = sourceBitmap.copy(Bitmap.Config.ARGB_8888, false)

                val mat = Mat()
                Utils.bitmapToMat(sourceBitmap, mat)

                // Use 0 rotation for picked images (no camera rotation)
                val rotation = 0

                // Clone mat for runDetection since it modifies the input in-place.
                // The original mat must stay intact for warpDocumentHighQuality.
                val matForDetection = mat.clone()
                val result = runDetection(matForDetection, rotation, _pipelineParams.value)

                if (result.isNotEmpty()) {
                    // Use the best quad directly (no fusion needed for single image)
                    // Unlike live camera, we don't have multiple frames to average
                    val bestQuad = result.first()

                    // Warp using the detected quad
                    val warped = warpDocumentHighQuality(mat, bestQuad, rotation)
                    // Clone before emitting to StateFlow so Compose gets its own
                    // independent copy that won't be affected by recycling in
                    // onCleared() or a subsequent processPickedDocument call.
                    _resultBitmap.value = (warped ?: mat.enhanceDocument().toBitmap())
                        .copy(Bitmap.Config.ARGB_8888, false)

                    // Create quad overlay bitmap for FileScanResultScreen
                    val originalMat = Mat()
                    Utils.bitmapToMat(_originalBitmap.value ?: sourceBitmap, originalMat)
                    originalMat.release()
                } else {
                    // No document detected — show enhanced original
                    // Clone before emitting to StateFlow.
                    _resultBitmap.value = mat.enhanceDocument().toBitmap()
                        .copy(Bitmap.Config.ARGB_8888, false)
                }

                matForDetection.release()
                mat.release()
                sourceBitmap.recycle()

                // Notify UI that scan is complete — navigation is handled by callback
                onScanComplete()

            } catch (e: Exception) {
                Log.e("CameraViewModel", "Error processing picked document", e)
            } finally {
                _isProcessing.value = false
            }
        }
    }

    /**
     * Reprocess the last picked document with current pipeline parameters.
     * Called from FileScanResultScreen when parameters change.
     */
    var processImageJob: Job? = null
    fun reprocessPickedDocument(context: Context) {
        val uri = lastPickedUri ?: return
        processImageJob?.cancel()
        processImageJob = viewModelScope.launch {
            delay(500.milliseconds)
            processPickedDocument(context, uri) {}
        }
    }

    /**
     * Enable auto Canny detection. The actual reprocessing is handled by
     * the caller (e.g., FileScanResultScreen's reprocessKey mechanism) to
     * avoid double-processing when combined with param-change triggers.
     * Thresholds are computed inside [DocumentDetector.preprocess] from the
     * pre-Canny blurred enhanced image — no separate calculation needed here.
     */
    fun enableCannyAuto() {
        updateParams(_pipelineParams.value.copy(cannyAutoDetect = true))
    }

    /**
     * Disable auto Canny detection — set the flag to false so camera pipeline
     * uses the manual thresholds instead. The actual reprocessing is handled
     * by the caller to avoid double-processing.
     */
    fun disableCannyAuto() {
        _pipelineParams.value = _pipelineParams.value.copy(
            cannyAutoDetect = false
        )
    }

    override fun onCleared() {
        super.onCleared()
        cameraExecutor.shutdown()
        cameraExecutor.awaitTermination(5, TimeUnit.SECONDS)
        clearBitmaps()
        lastWarpedQuadHash = 0
    }

    private fun clearBitmaps() {
        // Don't recycle bitmaps here — let GC handle it to avoid a race condition
        // with Compose composition (same issue as setPreviewBitmaps in
        // PipelineSettingsViewModel). If we recycle while Compose is still reading
        // the old refs during composition, we get "Canvas: trying to use a recycled
        // bitmap" crashes.
        _blurBitmap.value = null
        _claheBitmap.value = null
        _morphBitmap.value = null
        _filteredBitmap.value = null
        _originalBitmap.value = null
        _resultBitmap.value = null
        lastWarpedBitmap = null
    }

}
