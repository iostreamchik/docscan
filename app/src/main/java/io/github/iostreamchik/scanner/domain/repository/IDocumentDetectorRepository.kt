package io.github.iostreamchik.scanner.domain.repository

import io.github.iostreamchik.scanner.entity.DetectionParameters
import io.github.iostreamchik.scanner.entity.PipelineParams
import io.github.iostreamchik.scanner.entity.IntermediateBitmaps
import kotlinx.coroutines.flow.StateFlow
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint

interface IDocumentDetectorRepository {

    fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat

    fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        rotation: Int = 0,
        params: PipelineParams = PipelineParams()
    ): MatOfPoint?

    fun validateQuadSize(
        quad: MatOfPoint,
        originalWidth: Int,
        originalHeight: Int
    ): Boolean

    val detectionParams: StateFlow<DetectionParameters>?

    val detectorName: String

    fun captureIntermediateSnapshots(rotation: Int): IntermediateBitmaps

    fun capturePostDetectionSnapshots(rotation: Int): IntermediateBitmaps

    fun release()
}
