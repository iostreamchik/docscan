package io.github.iostreamchik.scanner.data.detector

import android.util.Log
import io.github.iostreamchik.scanner.entity.IntermediateBitmaps
import io.github.iostreamchik.scanner.entity.DetectionParameters
import io.github.iostreamchik.scanner.entity.PipelineParams
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
        lastUsedDetector = AsyncDetectorSource.NONE
        angleDeviations.clear()

        cachedScaledWidth = scaledWidth
        cachedScaledHeight = scaledHeight

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
        params: PipelineParams,
        rawMat: Mat?
    ): MatOfPoint? {
        Log.d(TAG, "detectQuad START: scaled=${scaledWidth}x${scaledHeight}, original=${originalWidth}x${originalHeight}")

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
                        originalWidth, originalHeight, params
                    )
                },
                AsyncDetectorSource.DIRECTIONAL_SUPPRESSION to async(Dispatchers.Default) {
                    opencv5Detector.detectQuad(
                        directionalMorph, scaledWidth, scaledHeight,
                        originalWidth, originalHeight, params
                    )
                }
            )

            val scoredResults = mutableListOf<DetectionCandidate>()

            primaryDeferredResults.forEach { (source, deferred) ->
                val quad = deferred.await()
                val points = quad?.toArray()?.toList()
                val detector = when (source) {
                    AsyncDetectorSource.MINIMAL -> minimalDetector
                    AsyncDetectorSource.DIRECTIONAL_SUPPRESSION -> opencv5Detector
                    else -> null
                }
                val valid = quad != null && (detector?.validateQuadSize(quad, originalWidth, originalHeight) ?: false)
                val quadSize = quad?.total() ?: 0
                quad?.release()
                val candidate = DetectionCandidate(source, if (valid) points else null, valid)
                angleDeviations[source] = candidate.deviation
                scoredResults.add(candidate)

                Log.d(TAG, "  $source: quad=${if (quad != null) "$quadSize pts" else "null"}, deviation=${formatDeviation(candidate.deviation)}")
            }

            val primaryWinner = CombinedDecision.selectPrimaryWinner(scoredResults)

            if (primaryWinner != null) {
                lastUsedDetector = primaryWinner.source
                when (primaryWinner.source) {
                    AsyncDetectorSource.MINIMAL ->
                        _detectionParams.value = minimalDetector.detectionParams?.value
                            ?.copy(detectorName = "Minimal")
                            ?: _detectionParams.value.copy(detectorName = "Minimal")
                    AsyncDetectorSource.DIRECTIONAL_SUPPRESSION ->
                        _detectionParams.value = opencv5Detector.detectionParams?.value
                            ?.copy(detectorName = "Directional Suppression")
                            ?: _detectionParams.value.copy(detectorName = "Directional Suppression")
                    else -> {}
                }
                Log.d(TAG, "  RESULT: ${primaryWinner.source} won with deviation ${"%.2f".format(primaryWinner.deviation)}°")
                return@coroutineScope MatOfPoint(*primaryWinner.quad!!.toTypedArray())
            }

            val sourceRawMat = rawMat ?: run {
                Log.w(TAG, "  ONNX fallback: rawMat is null, skipping")
                return@coroutineScope null
            }

            runOnnxFallbacks(sourceRawMat, params, originalWidth, originalHeight, scaledWidth, scaledHeight)
        }

        return result
    }

    private suspend fun runOnnxFallbacks(
        rawMat: Mat,
        params: PipelineParams,
        originalWidth: Int,
        originalHeight: Int,
        scaledWidth: Int,
        scaledHeight: Int,
    ): MatOfPoint? {
        Log.d(TAG, "  Primary detectors failed, running sequential ONNX fallbacks...")
        val fallbackStart = System.currentTimeMillis()

        val candidates = mutableListOf<DetectionCandidate>()

        val heatmapCandidate = runSingleDetector(
            heatmapCornerDetector, AsyncDetectorSource.HEATMAP_CORNER,
            rawMat, params, originalWidth, originalHeight, scaledWidth, scaledHeight
        )
        candidates.add(heatmapCandidate)

        val skipKeypointDetector = true
        var keypointCandidate: DetectionCandidate? = null
        if (heatmapCandidate.quad == null) {
            if (skipKeypointDetector.not()) {
                keypointCandidate = runSingleDetector(
                    cornerKeypointDetector, AsyncDetectorSource.CORNER_KEYPOINT,
                    rawMat, params, originalWidth, originalHeight, scaledWidth, scaledHeight
                )
                candidates.add(keypointCandidate)
            }
            if (keypointCandidate?.quad == null) {
                Log.d(TAG, "  Heatmap and CornerKeypoint failed, running Segmentation...")
                val segmentationCandidate = runSingleDetector(
                    onnxDetector, AsyncDetectorSource.SEGMENTATION,
                    rawMat, params, originalWidth, originalHeight, scaledWidth, scaledHeight
                )
                candidates.add(segmentationCandidate)
            }
        }

        val winner = CombinedDecision.selectFallbackWinner(candidates)

        if (winner != null) {
            lastUsedDetector = winner.source
            when (winner.source) {
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
            Log.d(TAG, "  RESULT: ${winner.source} fallback detected in ${System.currentTimeMillis() - fallbackStart}ms")
            return MatOfPoint(*winner.quad!!.toTypedArray())
        }

        Log.d(TAG, "  ONNX fallbacks completed in ${System.currentTimeMillis() - fallbackStart}ms")
        Log.w(TAG, "  RESULT: NO DETECTION (all detectors returned null)")
        return null
    }

    override fun captureIntermediateSnapshots(): IntermediateBitmaps {
        return when (lastUsedDetector) {
            AsyncDetectorSource.SEGMENTATION -> onnxDetector.captureIntermediateSnapshots()
            AsyncDetectorSource.HEATMAP_CORNER -> heatmapCornerDetector.captureIntermediateSnapshots()
            AsyncDetectorSource.CORNER_KEYPOINT -> cornerKeypointDetector.captureIntermediateSnapshots()
            AsyncDetectorSource.DIRECTIONAL_SUPPRESSION -> opencv5Detector.captureIntermediateSnapshots()
            else -> minimalDetector.captureIntermediateSnapshots()
        }
    }

    override fun capturePostDetectionSnapshots(): IntermediateBitmaps {
        return if (lastUsedDetector == AsyncDetectorSource.SEGMENTATION) {
            onnxDetector.capturePostDetectionSnapshots()
        } else {
            IntermediateBitmaps()
        }
    }

    private suspend fun runSingleDetector(
        detector: IDocumentDetector,
        source: AsyncDetectorSource,
        rawMat: Mat,
        params: PipelineParams,
        originalWidth: Int,
        originalHeight: Int,
        scaledWidth: Int,
        scaledHeight: Int,
    ): DetectionCandidate {
        val morph = detector.preprocess(rawMat, cachedScaledWidth, cachedScaledHeight, params)
        return try {
            val quad = detector.detectQuad(
                morph, scaledWidth, scaledHeight,
                originalWidth, originalHeight, params, rawMat
            )
            val points = quad?.toArray()?.toList()
            val valid = quad != null && detector.validateQuadSize(quad, originalWidth, originalHeight)
            quad?.release()
            val candidate = DetectionCandidate(source, if (valid) points else null, valid)
            angleDeviations[source] = candidate.deviation
            Log.d(TAG, "  $source: quad=${if (points != null) "${points.size} pts" else "null"}, deviation=${formatDeviation(candidate.deviation)}")
            candidate
        } finally {
            morph.release()
        }
    }

    private fun formatDeviation(deviation: Double): String =
        if (deviation.isFinite()) "${"%.2f".format(deviation)}°" else "N/A"

    override fun release() {
        cachedMorphImages.values.forEach { it?.release() }
        cachedMorphImages.clear()
        cachedScaledWidth = 0
        cachedScaledHeight = 0
        minimalDetector.release()
        opencv5Detector.release()
        heatmapCornerDetector.release()
        cornerKeypointDetector.release()
        onnxDetector.release()
    }
}
