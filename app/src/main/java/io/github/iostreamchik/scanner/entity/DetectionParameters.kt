package io.github.iostreamchik.scanner.entity

import androidx.compose.runtime.Immutable

@Immutable
data class DetectionParameters(
    val detectorName: String = "",
    val claheClipLimit: String = "",
    val cannyHigh: String = "",
    val cannyLow: String = "",
    val brightness: String = "",
    val maskThreshold: String = "",
    val cornerScore: String = "",
    val cornerError: String = "",
)