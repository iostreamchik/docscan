package io.github.iostreamchik.scanner.opencv

import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint2f

interface IMatBundle {
    var gray: Mat
    var blurred: Mat
    var enhanced: Mat
    var morph: Mat
    var temp: Mat
    var edges: Mat
    var morphAdd: Mat
    var hierarchy: Mat

    var mean: MatOfDouble
    var std: MatOfDouble
    var kernel: Mat
    var kernel2: Mat
    var hull: MatOfInt
    var hullPoints: MatOfPoint2f
    var approx: MatOfPoint2f

    fun releaseAll()
}

class MatBundle : IMatBundle {
    override var gray: Mat = Mat()
    override var blurred: Mat = Mat()
    override var enhanced: Mat = Mat()
    override var morph: Mat = Mat()
    override var temp: Mat = Mat()
    override var edges: Mat = Mat()
    override var morphAdd: Mat = Mat()
    override var hierarchy: Mat = Mat()

    // Pooled per-frame temporaries (reused, not reallocated each frame)
    override var mean: MatOfDouble = MatOfDouble()
    override var std: MatOfDouble = MatOfDouble()
    override var kernel: Mat = Mat()
    override var kernel2: Mat = Mat()
    override var hull: MatOfInt = MatOfInt()
    override var hullPoints: MatOfPoint2f = MatOfPoint2f()
    override var approx: MatOfPoint2f = MatOfPoint2f()

    override fun releaseAll() {
        gray.release()
        blurred.release()
        enhanced.release()
        morph.release()
        temp.release()
        edges.release()
        morphAdd.release()
        hierarchy.release()

        mean.release()
        std.release()
        kernel.release()
        kernel2.release()
        hull.release()
        hullPoints.release()
        approx.release()
    }
}
