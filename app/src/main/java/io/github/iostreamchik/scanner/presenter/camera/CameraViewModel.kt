package io.github.iostreamchik.scanner.presenter.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import io.github.iostreamchik.scanner.data.utils.enhanceDocument
import io.github.iostreamchik.scanner.data.utils.fixRotation
import io.github.iostreamchik.scanner.data.utils.quadDistance
import io.github.iostreamchik.scanner.data.utils.quadHash
import io.github.iostreamchik.scanner.data.utils.sortQuadPoints
import io.github.iostreamchik.scanner.data.utils.toBitmap
import io.github.iostreamchik.scanner.data.utils.toMatRGBA
import io.github.iostreamchik.scanner.data.utils.warpDocumentHighQuality
import io.github.iostreamchik.scanner.domain.repository.IDocumentDetectorRepository
import io.github.iostreamchik.scanner.entity.DetectionParameters
import io.github.iostreamchik.scanner.entity.IntermediateBitmaps
import io.github.iostreamchik.scanner.entity.PipelineParams
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.debounce
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Size
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.max
import kotlin.time.Duration.Companion.milliseconds

const val PROCESS_WIDTH = 448.0

class CameraViewModel(
    val repository: IDocumentDetectorRepository
) : ViewModel() {

    private val _state = MutableStateFlow(CameraState())
    val state: StateFlow<CameraState> = _state.asStateFlow()

    private val _detectedQuads = MutableStateFlow<List<MatOfPoint>>(emptyList())

    private val _contourData = MutableStateFlow<ContourData?>(null)
    val contourData: StateFlow<ContourData?> = _contourData.asStateFlow()

    val detectionParams: StateFlow<DetectionParameters> = repository.detectionParams ?: MutableStateFlow(DetectionParameters()).asStateFlow()

    private var currentDetectionJob: Job? = null

    val cameraExecutor = Executors.newSingleThreadExecutor()

    private val quadHistory = ArrayDeque<List<Point>>()
    private var lastFrameSize: Size? = null
    private var lastContourData: ContourData? = null
    private var lastFrameWidth = 0
    private var lastFrameHeight = 0
    private var cameraStartTime = 0L
    private val CAMERA_WARMUP_MS = 500L
    private val MAX_HISTORY = 4
    private var frameCounter = 0
    private val STABILITY_CHECK_INTERVAL = 1

    private var lastWarpedQuadHash: Long = 0
    private var lastWarpedBitmap: Bitmap? = null

    private var lastPickedUri: Uri? = null

    init {
        viewModelScope.launch {
            _detectedQuads
                .debounce(CONTOUR_UPDATE_THROTTLE_MS.milliseconds)
                .collect { quads ->
                    lastContourData?.release()
                    lastContourData = if (quads.isEmpty()) {
                        null
                    } else {
                        ContourData(
                            contours = quads,
                            frameWidth = lastFrameWidth,
                            frameHeight = lastFrameHeight
                        )
                    }
                    _contourData.value = lastContourData
                }
        }
    }

    private fun setState(transform: CameraState.() -> CameraState) {
        _state.value = _state.value.transform()
    }

    fun process(intent: CameraIntent) {
        when (intent) {
            is CameraIntent.ToggleTorch -> setState { copy(torchOn = !torchOn) }
            is CameraIntent.SetTorch -> setState { copy(torchOn = intent.on) }
            is CameraIntent.SetError -> setState { copy(errorId = intent.messageId) }
            is CameraIntent.UpdateParams -> setState { copy(pipelineParams = intent.params) }
            is CameraIntent.ProcessDocument -> processDocument(intent.context, intent.uri, intent.onComplete)
        }
    }

    fun processFrame(imageProxy: ImageProxy) {
        if (cameraStartTime == 0L) cameraStartTime = System.currentTimeMillis()
        if (System.currentTimeMillis() - cameraStartTime < CAMERA_WARMUP_MS) {
            imageProxy.close()
            return
        }

        val rawMat = imageProxy.toMatRGBA()
        val rotation = imageProxy.imageInfo.rotationDegrees
        imageProxy.close()

        val mat = rawMat.fixRotation(rotation)
        if (mat !== rawMat) rawMat.release()

        lastFrameWidth = mat.cols()
        lastFrameHeight = mat.rows()
        lastFrameSize = Size(mat.cols().toDouble(), mat.rows().toDouble())

        if (currentDetectionJob?.isActive == true) {
            mat.release()
            return
        }

        currentDetectionJob = viewModelScope.launch {
            try {
                withContext(Dispatchers.Default) {
                    val result = runDetection(mat, _state.value.pipelineParams)

                    _detectedQuads.value = result

                    if (result.isNotEmpty()) {
                        if (isStable()) {
                            val fusedQuad = getFusedQuad()
                            if (fusedQuad != null) {
                                result.forEach { updateHistory(it) }
                                val quadHash = quadHash(fusedQuad)

                                Log.d(
                                    "CameraViewModel",
                                    "\nFused Quad: ${fusedQuad.toArray().joinToString(", ")}"
                                )
                                val warped = if (quadHash != lastWarpedQuadHash) {
                                    warpDocumentHighQuality(mat, fusedQuad).also {
                                        lastWarpedBitmap?.recycle()
                                        lastWarpedBitmap = it
                                        lastWarpedQuadHash = quadHash
                                    }
                                } else {
                                    lastWarpedBitmap
                                }
                                setState {
                                    copy(resultBitmap = warped?.copy(Bitmap.Config.ARGB_8888, false))
                                }
                                fusedQuad.release()
                            }
                        } else {
                            result.forEach { updateHistory(it) }
                        }
                    }
                }
            } finally {
                mat.release()
            }
        }
    }

    private fun updateHistory(quad: MatOfPoint) {
        val points = quad.toArray().toList()
        if (points.size != 4) return
        if (quadHistory.size >= MAX_HISTORY) {
            quadHistory.removeFirst()
        }
        quadHistory.addLast(sortQuadPoints(points))
    }

    private fun isStable(): Boolean {
        if (quadHistory.size < MAX_HISTORY) return false
        if (++frameCounter % STABILITY_CHECK_INTERVAL != 0) return true

        val frameSize = lastFrameSize ?: return false
        val quads = quadHistory.toList()

        var totalMovement = 0.0
        var validPairs = 0

        for (i in 1 until quads.size) {
            totalMovement += quadDistance(
                quads[i - 1],
                quads[i],
                frameSize.width,
                frameSize.height
            )
            validPairs++
        }

        return validPairs > 0 && (totalMovement / validPairs) < 0.05
    }

    private fun getFusedQuad(): MatOfPoint? {
        if (quadHistory.isEmpty()) return null

        val averaged = arrayOf(Point(0.0, 0.0), Point(0.0, 0.0), Point(0.0, 0.0), Point(0.0, 0.0))

        for (i in 0..3) {
            for (quad in quadHistory) {
                averaged[i].x += quad[i].x
                averaged[i].y += quad[i].y
            }
            averaged[i].x /= quadHistory.size
            averaged[i].y /= quadHistory.size
        }

        return MatOfPoint(*averaged)
    }

    private suspend fun runDetection(
        mat: Mat,
        params: PipelineParams
    ): List<MatOfPoint> {
        val originalWidth = mat.cols()
        val originalHeight = mat.rows()
        val maxDimension = max(originalWidth, originalHeight)
        val scale = PROCESS_WIDTH / maxDimension
        val scaledWidth = (originalWidth * scale).toInt()
        val scaledHeight = (originalHeight * scale).toInt()

        Log.d("CameraViewModel", "=== runDetection START: ${originalWidth}x${originalHeight} -> scaled=${scaledWidth}x${scaledHeight} ===")

        val originalFrame = mat.toBitmap()
            .copy(Bitmap.Config.ARGB_8888, false)
        setState { copy(originalBitmap = originalFrame) }

        try {
            val morphResult = repository.preprocess(mat, scaledWidth, scaledHeight, params)

            Log.d("CameraViewModel", "  Calling detectQuad: morph=${morphResult.rows()}x${morphResult.cols()}, type=${morphResult.type()}, nonzero=${Core.countNonZero(morphResult)}")
            val bestQuad = repository.detectQuad(
                morphImage = morphResult,
                scaledWidth = scaledWidth,
                scaledHeight = scaledHeight,
                originalWidth = originalWidth,
                originalHeight = originalHeight,
                params = params,
                rawMat = mat
            )

            val snapshots = repository.captureIntermediateSnapshots()
            val postSnapshots = repository.capturePostDetectionSnapshots()
            setState {
                copy(
                    intermediateBitmaps = snapshots.copy(
                        edges = snapshots.edges ?: originalFrame,
                        mask = postSnapshots.mask ?: snapshots.mask
                    )
                )
            }

            if (bestQuad == null) {
                Log.d("CameraViewModel", "  runDetection: NO QUAD DETECTED")
                return emptyList()
            }
            Log.d("CameraViewModel", "  runDetection: bestQuad found with ${bestQuad.total()} points")

            if (!repository.validateQuadSize(bestQuad, originalWidth, originalHeight)) {
                bestQuad.release()
                return emptyList()
            }

            return listOf(bestQuad)

        } catch (e: Exception) {
            e.printStackTrace()
            return emptyList()
        }
    }

    private fun processDocument(context: Context, uri: Uri, onScanComplete: () -> Unit) {
        setState { copy(isProcessing = true) }
        viewModelScope.launch(Dispatchers.IO) {
            try {
                lastPickedUri = uri
                setState {
                    copy(
                        intermediateBitmaps = intermediateBitmaps.copy(
                            blur = null,
                            clahe = null,
                            morph = null,
                            edges = null,
                            corners = null
                        )
                    )
                }

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
                    @Suppress("DEPRECATION")
                    MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
                }

                setState { copy(originalBitmap = sourceBitmap.copy(Bitmap.Config.ARGB_8888, false), documentDetected = false) }

                val mat = Mat()
                Utils.bitmapToMat(sourceBitmap, mat)

                val matForDetection = mat.clone()
                val result = withContext(Dispatchers.Default) {
                    runDetection(matForDetection, _state.value.pipelineParams)
                }

                if (result.isNotEmpty()) {
                    val bestQuad = result.first()

                    val warped = warpDocumentHighQuality(mat, bestQuad)
                    setState {
                        copy(resultBitmap = (warped ?: mat.enhanceDocument().toBitmap())
                            .copy(Bitmap.Config.ARGB_8888, false), documentDetected = true)
                    }

                } else {
                    setState {
                        copy(resultBitmap = mat.enhanceDocument().toBitmap()
                            .copy(Bitmap.Config.ARGB_8888, false))
                    }
                }

                matForDetection.release()
                mat.release()
                sourceBitmap.recycle()

                onScanComplete()

            } catch (e: Exception) {
                Log.e("CameraViewModel", "Error processing picked document", e)
            } finally {
                setState { copy(isProcessing = false) }
            }
        }
    }

    override fun onCleared() {
        cameraExecutor.shutdown()
        cameraExecutor.awaitTermination(5, TimeUnit.SECONDS)
        clearBitmaps()
        clearQuadHistory()
        lastContourData?.release()
        lastContourData = null
        lastWarpedQuadHash = 0
    }

    private companion object {
        const val CONTOUR_UPDATE_THROTTLE_MS = 30L
    }

    private fun clearBitmaps() {
        setState {
            copy(
                intermediateBitmaps = IntermediateBitmaps(),
                originalBitmap = null,
                resultBitmap = null,
                documentDetected = false
            )
        }
        lastWarpedBitmap?.recycle()
        lastWarpedBitmap = null
    }

    private fun clearQuadHistory() {
        quadHistory.clear()
    }
}
