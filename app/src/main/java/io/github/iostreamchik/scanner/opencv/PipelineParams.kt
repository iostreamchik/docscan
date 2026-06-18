package io.github.iostreamchik.scanner.opencv

/**
 * Data class holding all adjustable pipeline parameters.
 */
data class PipelineParams(
    // Blur
    val medianBlurKsize: Int = 5,
    val gaussianSigma: Double = 1.0,

    // CLAHE
    val claheClipLimit: Float = 0.5f,
    val claheTileSize: Int = 8,

    // Morph Close (pre-Canny)
    val morphCloseSize: Int = 3,

    // Canny
    // 0f = auto mode (triggers Otsu+EMA threshold calculation)
    val cannyLow: Float = 0f,
    val cannyHigh: Float = 0f,
    val cannyAutoDetect: Boolean = true,

    // Strong Closing (post-Canny)
    val strongCloseSize: Int = 3,

    // Directional Suppression
    val directionalKernelSize: Int = 6,

    // Contour Detection
    val approxPolyDPTolerance: Float = 0.015f,
    val minAreaFraction: Float = 0.025f,

    // Scoring weights
    val scoreAreaWeight: Float = 0.5f,
    val scoreCenterWeight: Float = 0.3f,
    val scoreAreaRatioWeight: Float = 0.2f,
) {
    companion object {
        val Default = PipelineParams()
    }
}
