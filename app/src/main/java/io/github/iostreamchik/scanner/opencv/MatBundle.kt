package io.github.iostreamchik.scanner.opencv

import org.opencv.core.Mat

class MatBundle {
    var gray: Mat = Mat()
    var blurred: Mat = Mat()
    var enhanced: Mat = Mat()
    var morph: Mat = Mat()
    var temp: Mat = Mat()
    var edges: Mat = Mat()
    var morphAdd: Mat = Mat()
    var hierarchy: Mat = Mat()

    fun releaseAll() {
        gray.release()
        blurred.release()
        enhanced.release()
        morph.release()
        temp.release()
        edges.release()
        morphAdd.release()
        hierarchy.release()
    }
}
