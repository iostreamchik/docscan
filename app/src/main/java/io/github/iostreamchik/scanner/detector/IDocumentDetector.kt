package io.github.iostreamchik.scanner.detector

import kotlinx.coroutines.flow.StateFlow
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import io.github.iostreamchik.scanner.opencv.PipelineParams

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
}
