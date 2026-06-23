package io.github.iostreamchik.scanner.opencv

data class PipelineParams(
    val isAuto: Boolean = true,
    // Blur
    val medianBlurKsize: Int = 5,

    // CLAHE
    val claheClipLimit: Float = 0.5f,
    val claheTileSize: Int = 8,

    // Morph Close (pre-Canny)
    val morphCloseSize: Int = 3,

    // Canny
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
)
