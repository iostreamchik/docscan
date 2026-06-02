package io.github.iostreamchik.scanner.opencv

import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint2f

/**
 * No-op MatBundle for Compose preview. All Mats are empty (0x0) and
 * releaseAll() is a safe no-op.
 */
class MockMatBundle : IMatBundle {
    private val emptyMat = Mat()

    override var gray: Mat = emptyMat
    override var blurred: Mat = emptyMat
    override var enhanced: Mat = emptyMat
    override var morph: Mat = emptyMat
    override var temp: Mat = emptyMat
    override var edges: Mat = emptyMat
    override var morphAdd: Mat = emptyMat
    override var hierarchy: Mat = emptyMat

    override var mean: MatOfDouble = MatOfDouble()
    override var std: MatOfDouble = MatOfDouble()
    override var kernel: Mat = emptyMat
    override var kernel2: Mat = emptyMat
    override var hull: MatOfInt = MatOfInt()
    override var hullPoints: MatOfPoint2f = MatOfPoint2f()
    override var approx: MatOfPoint2f = MatOfPoint2f()

    override fun releaseAll() {
        // No-op for preview — empty Mats have no native resources
    }
}
