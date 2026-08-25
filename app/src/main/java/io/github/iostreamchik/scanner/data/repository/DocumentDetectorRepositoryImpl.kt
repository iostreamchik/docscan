package io.github.iostreamchik.scanner.data.repository

import io.github.iostreamchik.scanner.entity.DetectionParameters
import io.github.iostreamchik.scanner.entity.PipelineParams
import io.github.iostreamchik.scanner.entity.IntermediateBitmaps
import io.github.iostreamchik.scanner.domain.repository.IDocumentDetectorRepository
import io.github.iostreamchik.scanner.data.detector.IDocumentDetector
import kotlinx.coroutines.flow.StateFlow
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint

class DocumentDetectorRepositoryImpl(
    private val detector: IDocumentDetector
) : IDocumentDetectorRepository {

    override suspend fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat = detector.preprocess(rawMat, scaledWidth, scaledHeight, params)

    override suspend fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        params: PipelineParams
    ): MatOfPoint? = detector.detectQuad(
        morphImage, scaledWidth, scaledHeight,
        originalWidth, originalHeight, params
    )

    override fun validateQuadSize(
        quad: MatOfPoint,
        originalWidth: Int,
        originalHeight: Int
    ): Boolean = detector.validateQuadSize(quad, originalWidth, originalHeight)

    override val detectionParams: StateFlow<DetectionParameters>?
        get() = detector.detectionParams

    override val detectorName: String
        get() = detector.detectorName

    override fun captureIntermediateSnapshots(): IntermediateBitmaps {
        val snapshots = detector.captureIntermediateSnapshots()
        return IntermediateBitmaps(
            blur = snapshots.blur,
            clahe = snapshots.clahe,
            morph = snapshots.morph,
            edges = snapshots.edges,
            mask = snapshots.mask,
            corners = snapshots.corners
        )
    }

    override fun capturePostDetectionSnapshots(): IntermediateBitmaps {
        val snapshots = detector.capturePostDetectionSnapshots()
        return IntermediateBitmaps(
            blur = snapshots.blur,
            clahe = snapshots.clahe,
            morph = snapshots.morph,
            edges = snapshots.edges,
            mask = snapshots.mask,
            corners = snapshots.corners
        )
    }

    override fun release() = detector.release()
}
