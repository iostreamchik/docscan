package io.github.iostreamchik.scanner.presenter.camera

import org.opencv.core.MatOfPoint

data class ContourData(
    val contours: List<MatOfPoint>,
    val frameWidth: Int,
    val frameHeight: Int,
    val rotation: Int
) {
    fun release() {
        contours.forEach { it.release() }
    }
}