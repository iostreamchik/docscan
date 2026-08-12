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
import kotlinx.coroutines.runBlocking
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint

enum class AsyncDetectorSource(val detectionParamsName: String) {
    NONE(""),
    MINIMAL("Minimal"),
    DIRECTIONAL_SUPPRESSION("Directional Suppression"),
    CORNER_KEYPOINT("Corner Keypoint"),
    SEGMENTATION("Segmentation")
}

class CombinedDocumentDetector(
    private val minimalDetector: IDocumentDetector,
    private val opencv5Detector: IDocumentDetector,
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

    override fun preprocess(
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

        cachedRawMat = rawMat.clone()
        cachedScaledWidth = scaledWidth
        cachedScaledHeight = scaledHeight

        val morphResult = runBlocking {
            coroutineScope {
                val deferredResults = mapOf(
                    AsyncDetectorSource.MINIMAL to async {
                        minimalDetector.preprocess(rawMat, scaledWidth, scaledHeight, params)
                    },
                    AsyncDetectorSource.DIRECTIONAL_SUPPRESSION to async {
                        opencv5Detector.preprocess(rawMat, scaledWidth, scaledHeight, params)
                    }
                )

                deferredResults.mapValues { (_, deferred) ->
                    deferred.await()
                }
            }
        }

        morphResult.forEach { (source, morph) ->
            cachedMorphImages[source] = morph
            Log.d(TAG, "  preprocess $source done")
        }

        return cachedMorphImages[AsyncDetectorSource.DIRECTIONAL_SUPPRESSION] ?: Mat()
    }

    override fun detectQuad(
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

        val result = runBlocking(Dispatchers.Default) {
            coroutineScope {
                val primaryDeferredResults = mapOf(
                    AsyncDetectorSource.MINIMAL to async {
                        minimalDetector.detectQuad(
                            minimalMorph, scaledWidth, scaledHeight,
                            originalWidth, originalHeight, rotation, params
                        )
                    },
                    AsyncDetectorSource.DIRECTIONAL_SUPPRESSION to async {
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

                val primaryBest = scoredResults.minByOrNull { it.value.deviation }

                val primaryValid = primaryBest?.value?.quad != null && when (primaryBest.key) {
                    AsyncDetectorSource.MINIMAL -> minimalDetector.validateQuadSize(primaryBest.value.quad!!, originalWidth, originalHeight)
                    AsyncDetectorSource.DIRECTIONAL_SUPPRESSION -> opencv5Detector.validateQuadSize(primaryBest.value.quad!!, originalWidth, originalHeight)
                    else -> false
                }

                if (primaryValid) {
                    lastUsedDetector = primaryBest.key
                    when (primaryBest.key) {
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
                    Log.d(TAG, "  RESULT: $primaryBest.key won with deviation ${"%.2f".format(primaryBest.value.deviation)}°")
                    return@coroutineScope primaryBest.value.quad
                }

                val rawMat = cachedRawMat
                if (rawMat == null) {
                    Log.w(TAG, "  ONNX fallback: cachedRawMat is null, skipping")
                    return@coroutineScope null
                }

                Log.d(TAG, "  Primary detectors failed, running CORNER_KEYPOINT fallback...")
                val cornerMorph = cornerKeypointDetector.preprocess(rawMat, cachedScaledWidth, cachedScaledHeight, params)
                val cornerQuad = cornerKeypointDetector.detectQuad(
                    cornerMorph, scaledWidth, scaledHeight,
                    originalWidth, originalHeight, rotation, params
                )
                if (cornerQuad != null) {
                    val deviation = computeMaxAngleDeviation(cornerQuad)
                    angleDeviations[AsyncDetectorSource.CORNER_KEYPOINT] = deviation
                    Log.d(TAG, "  CORNER_KEYPOINT: quad=${cornerQuad.total()} pts, deviation=${"%.2f".format(deviation)}°")
                } else {
                    angleDeviations[AsyncDetectorSource.CORNER_KEYPOINT] = Double.MAX_VALUE
                }

                if (cornerQuad != null && cornerKeypointDetector.validateQuadSize(cornerQuad, originalWidth, originalHeight)) {
                    lastUsedDetector = AsyncDetectorSource.CORNER_KEYPOINT
                    cornerKeypointDetector.detectionParams?.value?.let {
                        _detectionParams.value = it.copy(detectorName = AsyncDetectorSource.CORNER_KEYPOINT.detectionParamsName)
                    }
                    Log.d(TAG, "  RESULT: CORNER_KEYPOINT fallback detected")
                    return@coroutineScope cornerQuad
                }
                cornerQuad?.release()

                Log.d(TAG, "  CORNER_KEYPOINT failed, running SEGMENTATION fallback...")
                val onnxMorph = onnxDetector.preprocess(rawMat, cachedScaledWidth, cachedScaledHeight, params)
                val onnxQuad = onnxDetector.detectQuad(
                    onnxMorph, scaledWidth, scaledHeight,
                    originalWidth, originalHeight, rotation, params
                )

                if (onnxQuad != null && onnxDetector.validateQuadSize(onnxQuad, originalWidth, originalHeight)) {
                    lastUsedDetector = AsyncDetectorSource.SEGMENTATION
                    onnxDetector.detectionParams?.value?.let {
                        _detectionParams.value = it.copy(detectorName = AsyncDetectorSource.SEGMENTATION.detectionParamsName)
                    }
                    Log.d(TAG, "  RESULT: SEGMENTATION fallback detected")
                    return@coroutineScope onnxQuad
                }
                onnxQuad?.release()
                Log.w(TAG, "  RESULT: NO DETECTION (all detectors returned null)")
                null
            }
        }

        return result
    }

    override fun captureIntermediateSnapshots(
        rotation: Int
    ): IntermediateBitmaps {
        return when (lastUsedDetector) {
            AsyncDetectorSource.SEGMENTATION -> onnxDetector.captureIntermediateSnapshots(rotation)
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
        cornerKeypointDetector.release()
        onnxDetector.release()
    }

    private data class ScoredResult(
        val quad: MatOfPoint?,
        val deviation: Double
    )
}
