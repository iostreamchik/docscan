package io.github.iostreamchik.scanner.detector

import android.graphics.Bitmap
import io.github.iostreamchik.scanner.fixRotation
import io.github.iostreamchik.scanner.old_detectors.DetectionParameters
import io.github.iostreamchik.scanner.opencv.IMatBundle
import io.github.iostreamchik.scanner.opencv.PipelineParams
import io.github.iostreamchik.scanner.toBitmap
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint

enum class DetectorSource {
    NONE,
    PRIMARY,
    FALLBACK
}

class CombinedDocumentDetector(
    private val primary: IDocumentDetector,
    private val fallback: IDocumentDetector,
) : IDocumentDetector {

    var lastUsedDetector: DetectorSource = DetectorSource.NONE
        private set

    var lastFallbackMaskBitmap: Bitmap? = null
        private set

    private var cachedRawMat: Mat? = null

    private val _detectionParams = MutableStateFlow(DetectionParameters())
    override val detectionParams: StateFlow<DetectionParameters> = _detectionParams.asStateFlow()

    override fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat {
        cachedRawMat?.release()
        lastFallbackMaskBitmap = null
        lastUsedDetector = DetectorSource.NONE

        cachedRawMat = rawMat.clone()

        return primary.preprocess(rawMat, scaledWidth, scaledHeight, params)
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
        val primaryQuad = primary.detectQuad(
            morphImage, scaledWidth, scaledHeight,
            originalWidth, originalHeight, rotation, params
        )

        if (primaryQuad != null && primary.validateQuadSize(primaryQuad, originalWidth, originalHeight)) {
            lastUsedDetector = DetectorSource.PRIMARY
            primary.detectionParams?.value?.let { _detectionParams.value = it }
            return primaryQuad
        }

        primaryQuad?.release()

        val rawMat = cachedRawMat ?: return null

        val fallbackMorph = fallback.preprocess(rawMat, scaledWidth, scaledHeight, params)

        val fallbackQuad = fallback.detectQuad(
            fallbackMorph, scaledWidth, scaledHeight,
            originalWidth, originalHeight, rotation, params
        )

        if (fallbackQuad != null && fallback.validateQuadSize(fallbackQuad, originalWidth, originalHeight)) {
            lastUsedDetector = DetectorSource.FALLBACK
            if (fallback is OnnxDocumentDetector && fallback.cachedMask != null) {
                lastFallbackMaskBitmap = fallback.cachedMask!!.fixRotation(rotation).toBitmap()
                    .copy(Bitmap.Config.ARGB_8888, false)
            }
            fallback.detectionParams?.value?.let { _detectionParams.value = it }
            return fallbackQuad
        }

        fallbackQuad?.release()
        lastUsedDetector = DetectorSource.NONE
        return null
    }

    override fun validateQuadSize(
        quad: MatOfPoint,
        originalWidth: Int,
        originalHeight: Int
    ): Boolean {
        return when (lastUsedDetector) {
            DetectorSource.PRIMARY -> primary.validateQuadSize(quad, originalWidth, originalHeight)
            DetectorSource.FALLBACK -> fallback.validateQuadSize(quad, originalWidth, originalHeight)
            else -> primary.validateQuadSize(quad, originalWidth, originalHeight)
        }
    }
}
