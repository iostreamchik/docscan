package io.github.iostreamchik.scanner.domain.repository

import io.github.iostreamchik.scanner.entity.DetectionParameters
import io.github.iostreamchik.scanner.entity.PipelineParams
import io.github.iostreamchik.scanner.entity.IntermediateBitmaps
import kotlinx.coroutines.flow.StateFlow
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint

interface IDocumentDetectorRepository {

    suspend fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat

    suspend fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        params: PipelineParams = PipelineParams(),
        rawMat: Mat? = null
    ): MatOfPoint?

    fun validateQuadSize(
        quad: MatOfPoint,
        originalWidth: Int,
        originalHeight: Int
    ): Boolean

    val detectionParams: StateFlow<DetectionParameters>?

    val detectorName: String

    fun captureIntermediateSnapshots(): IntermediateBitmaps

    fun capturePostDetectionSnapshots(): IntermediateBitmaps

    fun release()
}
