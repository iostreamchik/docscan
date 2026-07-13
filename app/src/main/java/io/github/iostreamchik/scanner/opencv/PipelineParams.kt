package io.github.iostreamchik.scanner.opencv

data class PipelineParams(
    val isClaheAuto: Boolean = true,
    val isCannyAuto: Boolean = false,
    // Blur
    val medianBlurKsize: Int = 7,

    // CLAHE
    val isClaheEnabled: Boolean = true,
    val claheClipLimit: Float = 1.5f,
    val claheTileSize: Int = 8,

    // Morph Close (pre-Canny)
    val isMorphCloseEnabled: Boolean = true,
    val morphCloseSize: Int = 7,

    // Canny
    val cannyLow: Float = 0f,
    val cannyHigh: Float = 0f,
    val cannyAutoDetect: Boolean = true,

    // Strong Closing (post-Canny)
    val isStrongCloseEnabled: Boolean = true,
    val strongCloseSize: Int = 7,

    // Directional Suppression
    val isDirectionalSuppressionEnabled: Boolean = true,
    val directionalKernelSize: Int = 4,

    // Contour Detection
    val approxPolyDPTolerance: Float = 0.025f,
    val minAreaFraction: Float = 0.025f,

    // Scoring weights
    val scoreAreaWeight: Float = 0.5f,
    val scoreCenterWeight: Float = 0.3f,
    val scoreAreaRatioWeight: Float = 0.2f,
)
