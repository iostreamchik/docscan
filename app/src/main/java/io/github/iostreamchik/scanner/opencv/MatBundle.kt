package io.github.iostreamchik.scanner.opencv

import org.opencv.core.Mat

class MatBundle {
    var gray: Mat? = null
        get() = field ?: Mat()
    var blurred: Mat? = null
        get() = field ?: Mat()
    var enhanced: Mat? = null
        get() = field ?: Mat()
    var morph: Mat? = null
        get() = field ?: Mat()
    var temp: Mat? = null
        get() = field ?: Mat()
    var edges: Mat? = null
        get() = field ?: Mat()
    var morphAdd: Mat? = null
        get() = field ?: Mat()
    var hierarchy: Mat? = null
        get() = field ?: Mat()

    fun releaseAll() {
        gray?.release()
        blurred?.release()
        enhanced?.release()
        morph?.release()
        temp?.release()
        edges?.release()
        morphAdd?.release()
        hierarchy?.release()
    }
}
