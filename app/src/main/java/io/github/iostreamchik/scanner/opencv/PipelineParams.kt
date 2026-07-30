package io.github.iostreamchik.scanner.opencv

data class PipelineParams(
    // Blur
    val medianBlurKsize: Int = 5,

    // CLAHE
    val claheClipLimit: Float = 1.5f,
    val claheTileSize: Int = 5,

    // Morph Close (pre-Canny)
    val morphCloseSize: Int = 5,

    // Canny
    val cannyLow: Float = 0f,
    val cannyHigh: Float = 0f,

    // Strong Closing (post-Canny)
    val strongCloseSize: Int = 5,

    // Directional Suppression
    val directionalKernelSize: Int = 6,

    // Contour Detection
    val approxPolyDPTolerance: Float = 0.025f,
    val minAreaFraction: Float = 0.025f,

    // Scoring weights
    val scoreAreaWeight: Float = 0.5f,
    val scoreCenterWeight: Float = 0.3f,
    val scoreAreaRatioWeight: Float = 0.2f,
)
