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
import io.github.iostreamchik.scanner.data.utils.toSortedQuad
import io.github.iostreamchik.scanner.data.utils.warpDocumentHighQuality
import io.github.iostreamchik.scanner.domain.repository.IDocumentDetectorRepository
import io.github.iostreamchik.scanner.entity.DetectionParameters
import io.github.iostreamchik.scanner.entity.PipelineParams
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Size
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.max

const val PROCESS_WIDTH = 448.0

class CameraViewModel(
    val repository: IDocumentDetectorRepository
) : ViewModel() {

    private val _state = MutableStateFlow(CameraState())
    val state: StateFlow<CameraState> = _state.asStateFlow()

    val detectionParams: StateFlow<DetectionParameters> = repository.detectionParams ?: MutableStateFlow(DetectionParameters()).asStateFlow()

    val cameraExecutor = Executors.newSingleThreadExecutor()

    private val quadHistory = ArrayDeque<MatOfPoint>()
    private var lastFrameSize: Size? = null
    private val MAX_HISTORY = 8
    private var frameCounter = 0
    private val STABILITY_CHECK_INTERVAL = 2

    private var lastWarpedQuadHash: Long = 0
    private var lastWarpedBitmap: Bitmap? = null

    private var lastPickedUri: Uri? = null

    private fun setState(transform: CameraState.() -> CameraState) {
        _state.value = _state.value.transform()
    }

    fun process(intent: CameraIntent) {
        when (intent) {
            is CameraIntent.ToggleTorch -> setState { copy(torchOn = !torchOn) }
            is CameraIntent.SetTorch -> setState { copy(torchOn = intent.on) }
            is CameraIntent.SetError -> setState { copy(error = intent.message) }
            is CameraIntent.UpdateParams -> setState { copy(pipelineParams = intent.params) }
            is CameraIntent.ProcessDocument -> processDocument(intent.context, intent.uri, intent.onComplete)
        }
    }

    fun processFrame(imageProxy: ImageProxy): List<MatOfPoint> {
        val width = imageProxy.width
        val height = imageProxy.height
        lastFrameSize = Size(width.toDouble(), height.toDouble())

        val mat = imageProxy.toMatRGBA()
        val rotation = imageProxy.imageInfo.rotationDegrees

        val result = runDetection(mat, rotation, _state.value.pipelineParams)

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
                        warpDocumentHighQuality(mat, fusedQuad, rotation).also {
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
                    return listOf(MatOfPoint(*fusedQuad.toArray()))
                }
            } else {
                result.forEach { updateHistory(it) }
            }
        }
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

            if (prevSorted.isEmpty() || currSorted.isEmpty()) continue

            totalMovement += quadDistance(
                prevSorted,
                currSorted,
                frameSize.width,
                frameSize.height
            )
            validPairs++
        }

        return validPairs > 0 && (totalMovement / validPairs) < 0.02
    }

    private fun getFusedQuad(): MatOfPoint? {
        if (quadHistory.isEmpty()) return null

        val validSortedQuads = quadHistory.mapNotNull { matOfPoint ->
            val points = matOfPoint.toArray().toList()
            if (points.size == 4) sortQuadPoints(points) else null
        }

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

        val originalFrame = mat.fixRotation(rotation).toBitmap()
            .copy(Bitmap.Config.ARGB_8888, false)
        setState { copy(originalBitmap = originalFrame) }

        try {
            val morphResult = repository.preprocess(mat, scaledWidth, scaledHeight, params)

            val snapshots = repository.captureIntermediateSnapshots(rotation)
            setState {
                copy(
                    intermediateBitmaps = intermediateBitmaps.copy(
                        blur = snapshots.blur,
                        clahe = snapshots.clahe,
                        morph = snapshots.morph,
                        edges = snapshots.edges ?: originalFrame,
                        mask = snapshots.mask,
                        corners = snapshots.corners
                    )
                )
            }

            Log.d("CameraViewModel", "  Calling detectQuad: morph=${morphResult.rows()}x${morphResult.cols()}, type=${morphResult.type()}, nonzero=${Core.countNonZero(morphResult)}")
            val bestQuad = repository.detectQuad(
                morphImage = morphResult,
                scaledWidth = scaledWidth,
                scaledHeight = scaledHeight,
                originalWidth = originalWidth,
                originalHeight = originalHeight,
                rotation = rotation,
                params = params
            )

            val postSnapshots = repository.capturePostDetectionSnapshots(rotation)
            if (postSnapshots.mask != null) {
                setState {
                    copy(intermediateBitmaps = intermediateBitmaps.copy(mask = postSnapshots.mask))
                }
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
                    MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
                }

                setState { copy(originalBitmap = sourceBitmap.copy(Bitmap.Config.ARGB_8888, false)) }

                val mat = Mat()
                Utils.bitmapToMat(sourceBitmap, mat)

                val rotation = 0

                val matForDetection = mat.clone()
                val result = runDetection(matForDetection, rotation, _state.value.pipelineParams)

                if (result.isNotEmpty()) {
                    val bestQuad = result.first()

                    val warped = warpDocumentHighQuality(mat, bestQuad, rotation)
                    setState {
                        copy(resultBitmap = (warped ?: mat.enhanceDocument().toBitmap())
                            .copy(Bitmap.Config.ARGB_8888, false))
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
        lastWarpedQuadHash = 0
    }

    private fun clearBitmaps() {
        setState {
            copy(
                intermediateBitmaps = IntermediateBitmaps(),
                originalBitmap = null,
                resultBitmap = null
            )
        }
        lastWarpedBitmap = null
    }
}
