package io.github.iostreamchik.scanner.detector

import android.content.Context
import android.util.Log
import io.github.iostreamchik.scanner.computeAngle
import io.github.iostreamchik.scanner.opencv.MatBundle
import io.github.iostreamchik.scanner.opencv.PipelineParams
import io.github.iostreamchik.scanner.sortQuadPoints
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.runBlocking
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import kotlin.math.abs
import kotlin.math.max

enum class AsyncDetectorSource {
    NONE,
    MINIMAL,
    OPENCV5,
    ONNX
}

class CombinedDocumentDetector(
    context: Context,
) : IDocumentDetector {

    private companion object {
        private const val TAG = "AsyncCombinedDetector"
    }

    private val minimalDetector = DocumentDetectorMinimal(MatBundle())
    private val opencv5Detector = DocumentDetectorOpenCV5(MatBundle())
    private val onnxDetector = OnnxDocumentDetector(context, MatBundle())

    private val cachedMorphImages = mutableMapOf<AsyncDetectorSource, Mat?>()
    private var cachedRawMat: Mat? = null
    private var cachedScaledWidth = 0
    private var cachedScaledHeight = 0
    private var lastUsedDetector = AsyncDetectorSource.NONE
    private var angleDeviations = mutableMapOf<AsyncDetectorSource, Double>()

    private val _detectionParams = MutableStateFlow(DetectionParameters())
    override val detectionParams = _detectionParams.asStateFlow()

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
                    AsyncDetectorSource.OPENCV5 to async {
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

        return cachedMorphImages[AsyncDetectorSource.OPENCV5] ?: Mat()
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
        val opencv5Morph = cachedMorphImages[AsyncDetectorSource.OPENCV5]

        if (minimalMorph == null || opencv5Morph == null) {
            Log.w(TAG, "detectQuad: missing cached classical morph images, skipping")
            return null
        }

        val result = runBlocking(Dispatchers.Default) {
            coroutineScope {
                val deferredResults = mapOf(
                    AsyncDetectorSource.MINIMAL to async {
                        minimalDetector.detectQuad(
                            minimalMorph, scaledWidth, scaledHeight,
                            originalWidth, originalHeight, rotation, params
                        )
                    },
                    AsyncDetectorSource.OPENCV5 to async {
                        opencv5Detector.detectQuad(
                            opencv5Morph, scaledWidth, scaledHeight,
                            originalWidth, originalHeight, rotation, params
                        )
                    }
                )

                val scoredResults = mutableMapOf<AsyncDetectorSource, ScoredResult>()

                deferredResults.forEach { (source, deferred) ->
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

                val classicalBest = scoredResults.minByOrNull { it.value.deviation }

                val classicalValid = classicalBest?.value?.quad != null && when (classicalBest.key) {
                    AsyncDetectorSource.MINIMAL -> minimalDetector.validateQuadSize(classicalBest.value.quad!!, originalWidth, originalHeight)
                    AsyncDetectorSource.OPENCV5 -> opencv5Detector.validateQuadSize(classicalBest.value.quad!!, originalWidth, originalHeight)
                    else -> false
                }

                if (classicalValid) {
                    lastUsedDetector = classicalBest.key
                    when (classicalBest.key) {
                        AsyncDetectorSource.MINIMAL -> minimalDetector.detectionParams?.value?.let { _detectionParams.value = it }
                        AsyncDetectorSource.OPENCV5 -> _detectionParams.value = opencv5Detector.detectionParams.value
                        else -> {}
                    }
                    Log.d(TAG, "  RESULT: $classicalBest.key won with deviation ${"%.2f".format(classicalBest.value.deviation)}°")
                    return@coroutineScope classicalBest.value.quad
                }

                val rawMat = cachedRawMat
                if (rawMat == null) {
                    Log.w(TAG, "  ONNX fallback: cachedRawMat is null, skipping")
                    return@coroutineScope null
                }

                Log.d(TAG, "  Classical failed, running ONNX fallback...")
                val onnxMorph = onnxDetector.preprocess(rawMat, cachedScaledWidth, cachedScaledHeight, params)
                val onnxQuad = onnxDetector.detectQuad(
                    onnxMorph, scaledWidth, scaledHeight,
                    originalWidth, originalHeight, rotation, params
                )

                if (onnxQuad != null && onnxDetector.validateQuadSize(onnxQuad, originalWidth, originalHeight)) {
                    lastUsedDetector = AsyncDetectorSource.ONNX
                    _detectionParams.value = onnxDetector.detectionParams.value
                    Log.d(TAG, "  RESULT: ONNX fallback detected")
                    return@coroutineScope onnxQuad
                }

                onnxQuad?.release()
                Log.w(TAG, "  RESULT: NO DETECTION (all detectors returned null)")
                null
            }
        }

        return result
    }

    override fun validateQuadSize(
        quad: MatOfPoint,
        originalWidth: Int,
        originalHeight: Int
    ): Boolean {
        return when (lastUsedDetector) {
            AsyncDetectorSource.MINIMAL -> minimalDetector.validateQuadSize(quad, originalWidth, originalHeight)
            AsyncDetectorSource.OPENCV5 -> opencv5Detector.validateQuadSize(quad, originalWidth, originalHeight)
            AsyncDetectorSource.ONNX -> onnxDetector.validateQuadSize(quad, originalWidth, originalHeight)
            else -> minimalDetector.validateQuadSize(quad, originalWidth, originalHeight)
        }
    }

    private fun computeMaxAngleDeviation(quad: MatOfPoint): Double {
        val pts = sortQuadPoints(quad.toArray().toList())
        if (pts.size != 4) return Double.MAX_VALUE

        var maxDeviation = 0.0
        for (i in 0..3) {
            val angle = computeAngle(pts[(i + 1) % 4], pts[(i + 3) % 4], pts[i])
            maxDeviation = max(maxDeviation, abs(90.0 - angle))
        }
        return maxDeviation
    }

    fun release() {
        cachedMorphImages.values.forEach { it?.release() }
        cachedMorphImages.clear()
        cachedRawMat?.release()
        cachedRawMat = null
        cachedScaledWidth = 0
        cachedScaledHeight = 0
        onnxDetector.release()
    }

    private data class ScoredResult(
        val quad: MatOfPoint?,
        val deviation: Double
    )
}
