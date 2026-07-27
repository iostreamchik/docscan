package io.github.iostreamchik.scanner.detector

data class DetectionParameters(
    val claheClipLimit: String = "",
    val cannyHigh: String = "",
    val cannyLow: String = "",
    val brightness: String = "",
    val maskThreshold: String = "",
    val cornerScore: String = "",
    val cornerError: String = "",
)