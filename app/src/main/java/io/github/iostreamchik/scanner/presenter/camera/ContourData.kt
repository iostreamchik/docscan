package io.github.iostreamchik.scanner.presenter.camera

import androidx.compose.runtime.Stable
import org.opencv.core.MatOfPoint

@Stable
data class ContourData(
    val contours: List<MatOfPoint>,
    val frameWidth: Int,
    val frameHeight: Int
) {
    fun release() {
        contours.forEach { it.release() }
    }
}