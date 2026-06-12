package io.github.iostreamchik.scanner.opencv

import io.github.iostreamchik.scanner.pipeline.PipelineType

/**
 * Data class holding all adjustable pipeline parameters.
 */
data class PipelineParams(
    // Blur
    val medianBlurKsize: Int = 3,
    val gaussianSigma: Double = 1.0,

    // CLAHE
    val claheClipLimit: Float = 0.8f,
    val claheTileSize: Int = 16,

    // Morph Close (pre-Canny)
    val morphCloseSize: Int = 9,

    // Canny
    val cannyLow: Float = 20f,
    val cannyHigh: Float = 60f,
    val cannyAutoDetect: Boolean = false,

    // Strong Closing (post-Canny)
    val strongCloseSize: Int = 5,

    // Directional Suppression
    val directionalKernelSize: Int = 15,

    // Contour Detection
    val approxPolyDPTolerance: Float = 0.015f,
    val minAreaFraction: Float = 0.025f,

    // Scoring weights
    val scoreAreaWeight: Float = 0.5f,
    val scoreCenterWeight: Float = 0.3f,
    val scoreAreaRatioWeight: Float = 0.2f,

    // Pipeline selection
    val pipelineType: PipelineType = PipelineType.CANNY_OTSU,

    // Adaptive thresholding parameters
    val adaptiveBlockSize: Int = 11,      // Must be odd, 3–51
    val adaptiveConstant: Float = 2.0f,   // C constant, 0–20
) {
    companion object {
        val Default = PipelineParams()
    }
}
