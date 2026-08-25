package io.github.iostreamchik.scanner.data.detector

import io.github.iostreamchik.scanner.entity.DetectionParameters
import io.github.iostreamchik.scanner.entity.PipelineParams
import kotlinx.coroutines.flow.MutableStateFlow
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint

/**
 * No-op detector for Compose preview. All methods return empty values and
 * detectionParams is a constant empty flow — avoids UnsatisfiedLinkError
 * when OpenCV native libs aren't loaded (preview).
 */
class MockDocumentDetector : IDocumentDetector {

    override val detectionParams = MutableStateFlow(DetectionParameters())

    override suspend fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat = Mat()

    override suspend fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        params: PipelineParams
    ): MatOfPoint? = null

    override fun validateQuadSize(
        quad: MatOfPoint,
        originalWidth: Int,
        originalHeight: Int
    ): Boolean = false

    override fun release() {
    }
}
