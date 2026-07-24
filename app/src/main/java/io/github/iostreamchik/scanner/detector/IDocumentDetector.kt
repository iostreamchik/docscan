package io.github.iostreamchik.scanner.detector

import android.graphics.Bitmap
import kotlinx.coroutines.flow.StateFlow
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import io.github.iostreamchik.scanner.opencv.PipelineParams

/**
 * Intermediate bitmap snapshots produced during preprocessing.
 * Each detector populates only the stages it actually produces.
 */
data class IntermediateSnapshots(
    val blur: Bitmap? = null,
    val clahe: Bitmap? = null,
    val morph: Bitmap? = null,
    val edges: Bitmap? = null,
    val mask: Bitmap? = null
)

/**
 * Document detection interface — abstracts the detection pipeline so
 * different implementations can be swapped via dependency injection.
 */
interface IDocumentDetector {

    /**
     * Runs the full image preprocessing pipeline (resize, grayscale, blur,
     * edge detection, etc.) and stores intermediate results in the matBundle.
     */
    fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat

    /**
     * Extracts document candidates from a preprocessed edge/morph Mat.
     * Returns the best quad in original image coordinates, or null if none found.
     */
    fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        rotation: Int = 0,
        params: PipelineParams = PipelineParams()
    ): MatOfPoint?

    /**
     * Validates that a detected quad doesn't fill the entire frame,
     * which indicates a likely false positive (background texture).
     */
    fun validateQuadSize(
        quad: MatOfPoint,
        originalWidth: Int,
        originalHeight: Int
    ): Boolean

    /**
     * Current detection parameters (CLAHE clip limit, Canny thresholds, brightness).
     * Returns null if the implementation doesn't track these metrics.
     */
    val detectionParams: StateFlow<DetectionParameters>?
        get() = null

    /**
     * Captures intermediate bitmap snapshots from the detector's internal mat bundle
     * after [preprocess] has completed. Each detector populates only the stages it produces.
     * Called before the pooled Mats are released so the intermediate results are still valid.
     *
     * @param rotation device rotation degrees for correcting bitmap orientation
     * @return snapshots of intermediate processing stages
     */
    fun captureIntermediateSnapshots(
        rotation: Int
    ): IntermediateSnapshots = IntermediateSnapshots()

    /**
     * Captures snapshots that are only available after [detectQuad] has run.
     * Used by detectors like ONNX where the mask is produced inside detectQuad
     * (e.g., when ONNX runs as a fallback after classical detection fails).
     *
     * @param rotation device rotation degrees for correcting bitmap orientation
     * @return snapshots of post-detection stages, or empty if none
     */
    fun capturePostDetectionSnapshots(
        rotation: Int
    ): IntermediateSnapshots = IntermediateSnapshots()
}
