package io.github.iostreamchik.scanner.data.detector

import io.github.iostreamchik.scanner.entity.IntermediateBitmaps
import io.github.iostreamchik.scanner.entity.DetectionParameters
import io.github.iostreamchik.scanner.entity.PipelineParams
import kotlinx.coroutines.flow.StateFlow
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.geometry.Geometry

/**
 * Document detection interface — abstracts the detection pipeline so
 * different implementations can be swapped via dependency injection.
 */
interface IDocumentDetector {

    /**
     * Runs the full image preprocessing pipeline (resize, grayscale, blur,
     * edge detection, etc.) and stores intermediate results in the matBundle.
     */
    suspend fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat

    /**
     * Extracts document candidates from a preprocessed edge/morph Mat.
     * Returns the best quad in original image coordinates, or null if none found.
     */
    suspend fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        params: PipelineParams = PipelineParams(),
        rawMat: Mat? = null
    ): MatOfPoint?

    /**
     * Validates that a detected quad doesn't fill the entire frame,
     * which indicates a likely false positive (background texture).
     */
    fun validateQuadSize(
        quad: MatOfPoint,
        originalWidth: Int,
        originalHeight: Int
    ): Boolean {
        val rect = Geometry.boundingRect(quad)
        val quadArea = rect.width * rect.height
        val frameArea = originalWidth * originalHeight
        return quadArea <= frameArea * 0.95
    }

    /**
     * Current detection parameters (CLAHE clip limit, Canny thresholds, brightness).
     * Returns null if the implementation doesn't track these metrics.
     */
    val detectionParams: StateFlow<DetectionParameters>?
        get() = null

    /**
     * Human-readable name of the detector (or the active inner detector for combined detectors).
     */
    val detectorName: String
        get() = javaClass.simpleName

    /**
     * Captures intermediate bitmap snapshots from the detector's internal mat bundle
     * after [preprocess] has completed. Each detector populates only the stages it produces.
     * Called before the pooled Mats are released so the intermediate results are still valid.
     *
     * @return snapshots of intermediate processing stages
     */
    fun captureIntermediateSnapshots(): IntermediateBitmaps = IntermediateBitmaps()

    /**
     * Captures snapshots that are only available after [detectQuad] has run.
     * Used by detectors like ONNX where the mask is produced inside detectQuad
     * (e.g., when ONNX runs as a fallback after classical detection fails).
     *
     * @return snapshots of post-detection stages, or empty if none
     */
    fun capturePostDetectionSnapshots(): IntermediateBitmaps = IntermediateBitmaps()

    /**
     * Releases all native resources held by this detector (pooled Mats, ONNX sessions, etc.).
     * Call once when the detector is no longer needed.
     */
    fun release()
}
