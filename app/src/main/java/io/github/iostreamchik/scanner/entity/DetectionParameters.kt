package io.github.iostreamchik.scanner.entity

data class DetectionParameters(
    val claheClipLimit: String = "",
    val cannyHigh: String = "",
    val cannyLow: String = "",
    val brightness: String = "",
    val maskThreshold: String = "",
    val cornerScore: String = "",
    val cornerError: String = "",
)