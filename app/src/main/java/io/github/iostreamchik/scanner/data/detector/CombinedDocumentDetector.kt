package io.github.iostreamchik.scanner.data.detector

import android.util.Log
import io.github.iostreamchik.scanner.data.utils.computeMaxAngleDeviation
import io.github.iostreamchik.scanner.entity.IntermediateBitmaps
import io.github.iostreamchik.scanner.entity.DetectionParameters
import io.github.iostreamchik.scanner.entity.PipelineParams
import io.github.iostreamchik.scanner.data.utils.sortQuadPoints
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint

enum class AsyncDetectorSource(val detectionParamsName: String) {
    NONE(""),
    MINIMAL("Minimal"),
    DIRECTIONAL_SUPPRESSION("Directional Suppression"),
    HEATMAP_CORNER("Heatmap Corner"),
    CORNER_KEYPOINT("Corner Keypoint"),
    SEGMENTATION("Segmentation")
}

class CombinedDocumentDetector(
    private val minimalDetector: IDocumentDetector,
    private val opencv5Detector: IDocumentDetector,
    private val heatmapCornerDetector: IDocumentDetector,
    private val cornerKeypointDetector: IDocumentDetector,
    private val onnxDetector: IDocumentDetector,
) : IDocumentDetector {

    private companion object {
        private const val TAG = "AsyncCombinedDetector"
    }

    private val cachedMorphImages = mutableMapOf<AsyncDetectorSource, Mat?>()
    private var cachedRawMat: Mat? = null
    private var cachedScaledWidth = 0
    private var cachedScaledHeight = 0
    private var lastUsedDetector = AsyncDetectorSource.NONE
    private var angleDeviations = mutableMapOf<AsyncDetectorSource, Double>()

    private val _detectionParams = MutableStateFlow(DetectionParameters())
    override val detectionParams = _detectionParams.asStateFlow()

    override val detectorName: String
        get() = lastUsedDetector.detectionParamsName.ifBlank { "Combined" }

    override suspend fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat {
        cachedMorphImages.values.forEach { it?.release() }
        cachedMorphImages.clear()
        cachedRawMat?.release()
        lastUsedDetector = AsyncDetectorSource.NONE
        angleDeviations.clear()

        cachedScaledWidth = scaledWidth
        cachedScaledHeight = scaledHeight
        cachedRawMat = rawMat.clone()

        val morphResult = coroutineScope {
            val deferredResults = mapOf(
                AsyncDetectorSource.MINIMAL to async(Dispatchers.Default) {
                    minimalDetector.preprocess(rawMat, scaledWidth, scaledHeight, params)
                },
                AsyncDetectorSource.DIRECTIONAL_SUPPRESSION to async(Dispatchers.Default) {
                    opencv5Detector.preprocess(rawMat, scaledWidth, scaledHeight, params)
                }
            )

            deferredResults.mapValues { (_, deferred) ->
                deferred.await()
            }
        }

        morphResult.forEach { (source, morph) ->
            cachedMorphImages[source] = morph
            Log.d(TAG, "  preprocess $source done")
        }

        return cachedMorphImages[AsyncDetectorSource.DIRECTIONAL_SUPPRESSION] ?: Mat()
    }

    override suspend fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        rotation: Int,
        params: PipelineParams
    ): MatOfPoint? {
        Log.d(TAG, "detectQuad START: scaled=${scaledWidth}x${scaledHeight}, original=${originalWidth}x${originalHeight}, rotation=$rotation")

        angleDeviations.clear()

        val minimalMorph = cachedMorphImages[AsyncDetectorSource.MINIMAL]
        val directionalMorph = cachedMorphImages[AsyncDetectorSource.DIRECTIONAL_SUPPRESSION]

        if (minimalMorph == null || directionalMorph == null) {
            Log.w(TAG, "detectQuad: missing cached classical morph images, skipping")
            return null
        }

        val result = coroutineScope {
            val primaryDeferredResults = mapOf(
                AsyncDetectorSource.MINIMAL to async(Dispatchers.Default) {
                    minimalDetector.detectQuad(
                        minimalMorph, scaledWidth, scaledHeight,
                        originalWidth, originalHeight, rotation, params
                    )
                },
                AsyncDetectorSource.DIRECTIONAL_SUPPRESSION to async(Dispatchers.Default) {
                    opencv5Detector.detectQuad(
                        directionalMorph, scaledWidth, scaledHeight,
                        originalWidth, originalHeight, rotation, params
                    )
                }
            )

            val scoredResults = mutableMapOf<AsyncDetectorSource, ScoredResult>()

            primaryDeferredResults.forEach { (source, deferred) ->
                val quad = deferred.await()
                val deviation = if (quad != null) {
                    computeMaxAngleDeviation(quad)
                } else {
                    Double.MAX_VALUE
                }
                angleDeviations[source] = deviation
                scoredResults[source] = ScoredResult(quad, deviation)

                Log.d(TAG, "  $source: quad=${if (quad != null) "${quad.total()} pts" else "null"}, deviation=${"%.2f".format(deviation)}°")
            }

            val validResults = mutableListOf<Pair<AsyncDetectorSource, ScoredResult>>()

            for ((source, scored) in scoredResults) {
                if (scored.quad != null) {
                    val detector = when (source) {
                        AsyncDetectorSource.MINIMAL -> minimalDetector
                        AsyncDetectorSource.DIRECTIONAL_SUPPRESSION -> opencv5Detector
                        else -> null
                    }
                    if (detector != null && detector.validateQuadSize(scored.quad!!, originalWidth, originalHeight)) {
                        validResults.add(source to scored)
                    } else {
                        scored.quad?.release()
                    }
                }
            }

            val primaryWinner = validResults.minByOrNull { it.second.deviation }

            if (primaryWinner != null) {
                val (winnerSource, winnerScored) = primaryWinner
                lastUsedDetector = winnerSource
                when (winnerSource) {
                    AsyncDetectorSource.MINIMAL ->
                        minimalDetector.detectionParams?.value?.let {
                            _detectionParams.value = it.copy(detectorName = "Minimal")
                        }
                    AsyncDetectorSource.DIRECTIONAL_SUPPRESSION ->
                        opencv5Detector.detectionParams?.value?.let {
                            _detectionParams.value = it.copy(detectorName = "Directional Suppression")
                        }
                    else -> {}
                }
                validResults.filter { it.first != winnerSource }
                    .forEach { it.second.quad?.release() }
                Log.d(TAG, "  RESULT: $winnerSource won with deviation ${"%.2f".format(winnerScored.deviation)}°")
                return@coroutineScope winnerScored.quad
            }

            val rawMat = cachedRawMat?.clone() ?: run {
                Log.w(TAG, "  ONNX fallback: cachedRawMat is null, skipping")
                return@coroutineScope null
            }

            Log.d(TAG, "  Primary detectors failed, running parallel ONNX fallbacks...")
            val fallbackStart = System.currentTimeMillis()

            val heatmapDeferred = async(Dispatchers.Default) {
                runSingleDetector(heatmapCornerDetector, rawMat, params, originalWidth, originalHeight, rotation, scaledWidth, scaledHeight)
            }
            val cornerDeferred = async(Dispatchers.Default) {
                runSingleDetector(cornerKeypointDetector, rawMat, params, originalWidth, originalHeight, rotation, scaledWidth, scaledHeight)
            }
            val segDeferred = async(Dispatchers.Default) {
                runSingleDetector(onnxDetector, rawMat, params, originalWidth, originalHeight, rotation, scaledWidth, scaledHeight)
            }

            val results = arrayOf(
                AsyncDetectorSource.HEATMAP_CORNER to heatmapDeferred,
                AsyncDetectorSource.CORNER_KEYPOINT to cornerDeferred,
                AsyncDetectorSource.SEGMENTATION to segDeferred,
            )

            val winner = results.firstNotNullOfOrNull { (source, deferred) ->
                val quad = deferred.await()
                if (quad != null) {
                    val deviation = computeMaxAngleDeviation(quad)
                    angleDeviations[source] = deviation
                    Log.d(TAG, "  $source: quad=${quad.total()} pts, deviation=${"%.2f".format(deviation)}°")
                    source to quad
                } else {
                    angleDeviations[source] = Double.MAX_VALUE
                    null
                }
            }

            // Cancel losing detectors
            if (winner != null) {
                results.filter { it.first != winner.first }
                    .forEach { it.second.cancel() }
            }

            if (winner != null) {
                val (winnerSource, winnerQuad) = winner
                lastUsedDetector = winnerSource
                when (winnerSource) {
                    AsyncDetectorSource.HEATMAP_CORNER ->
                        heatmapCornerDetector.detectionParams?.value?.let {
                            _detectionParams.value = it.copy(detectorName = AsyncDetectorSource.HEATMAP_CORNER.detectionParamsName)
                        }
                    AsyncDetectorSource.CORNER_KEYPOINT ->
                        cornerKeypointDetector.detectionParams?.value?.let {
                            _detectionParams.value = it.copy(detectorName = AsyncDetectorSource.CORNER_KEYPOINT.detectionParamsName)
                        }
                    AsyncDetectorSource.SEGMENTATION ->
                        onnxDetector.detectionParams?.value?.let {
                            _detectionParams.value = it.copy(detectorName = AsyncDetectorSource.SEGMENTATION.detectionParamsName)
                        }
                    else -> {}
                }
                Log.d(TAG, "  RESULT: $winnerSource fallback detected in ${System.currentTimeMillis() - fallbackStart}ms")
                return@coroutineScope winnerQuad
            }

            Log.d(TAG, "  ONNX fallbacks completed in ${System.currentTimeMillis() - fallbackStart}ms")
            Log.w(TAG, "  RESULT: NO DETECTION (all detectors returned null)")
            null
        }

        return result
    }

    override fun captureIntermediateSnapshots(
        rotation: Int
    ): IntermediateBitmaps {
        return when (lastUsedDetector) {
            AsyncDetectorSource.SEGMENTATION -> onnxDetector.captureIntermediateSnapshots(rotation)
            AsyncDetectorSource.HEATMAP_CORNER -> heatmapCornerDetector.captureIntermediateSnapshots(rotation)
            AsyncDetectorSource.CORNER_KEYPOINT -> cornerKeypointDetector.captureIntermediateSnapshots(rotation)
            AsyncDetectorSource.DIRECTIONAL_SUPPRESSION -> opencv5Detector.captureIntermediateSnapshots(rotation)
            else -> minimalDetector.captureIntermediateSnapshots(rotation)
        }
    }

    override fun capturePostDetectionSnapshots(
        rotation: Int
    ): IntermediateBitmaps {
        return if (lastUsedDetector == AsyncDetectorSource.SEGMENTATION) {
            onnxDetector.capturePostDetectionSnapshots(rotation)
        } else {
            IntermediateBitmaps()
        }
    }

    private suspend fun runSingleDetector(
        detector: IDocumentDetector,
        rawMat: Mat,
        params: PipelineParams,
        originalWidth: Int,
        originalHeight: Int,
        rotation: Int,
        scaledWidth: Int,
        scaledHeight: Int,
    ): MatOfPoint? {
        val morph = detector.preprocess(rawMat, cachedScaledWidth, cachedScaledHeight, params)
        try {
            val quad = detector.detectQuad(
                morph, scaledWidth, scaledHeight,
                originalWidth, originalHeight, rotation, params
            )
            if (quad != null && detector.validateQuadSize(quad, originalWidth, originalHeight)) {
                return quad
            }
            quad?.release()
            return null
        } finally {
            morph.release()
        }
    }

    private fun computeMaxAngleDeviation(quad: MatOfPoint): Double {
        val pts = sortQuadPoints(quad.toArray().toList())
        return computeMaxAngleDeviation(pts)
    }

    override fun release() {
        cachedMorphImages.values.forEach { it?.release() }
        cachedMorphImages.clear()
        cachedRawMat?.release()
        cachedRawMat = null
        cachedScaledWidth = 0
        cachedScaledHeight = 0
        minimalDetector.release()
        opencv5Detector.release()
        heatmapCornerDetector.release()
        cornerKeypointDetector.release()
        onnxDetector.release()
    }

    private data class ScoredResult(
        val quad: MatOfPoint?,
        val deviation: Double
    )
}
