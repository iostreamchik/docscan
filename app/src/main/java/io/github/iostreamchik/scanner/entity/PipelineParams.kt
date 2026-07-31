package io.github.iostreamchik.scanner.entity

data class PipelineParams(
    val medianBlurKsize: Int = 5,
    val claheClipLimit: Float = 1.5f,
    val claheTileSize: Int = 5,
    val morphCloseSize: Int = 5,
    val cannyLow: Float = 0f,
    val cannyHigh: Float = 0f,
    val strongCloseSize: Int = 5,
    val directionalKernelSize: Int = 6,
    val approxPolyDPTolerance: Float = 0.025f,
    val minAreaFraction: Float = 0.025f,
    val scoreAreaWeight: Float = 0.5f,
    val scoreCenterWeight: Float = 0.3f,
    val scoreAreaRatioWeight: Float = 0.2f,
)
