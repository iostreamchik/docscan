package io.github.iostreamchik.scanner.opencv

import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint2f

interface IMatBundle {
    fun getGray(): Mat
    fun getBlurred(): Mat
    fun getEnhanced(): Mat
    fun getMorph(): Mat
    fun getTemp(): Mat
    fun getEdges(): Mat
    fun getMorphAdd(): Mat
    fun getHierarchy(): Mat

    // Directional line suppression (bright environment)
    fun getGrayGaussian(): Mat
    fun getHorizontalClose(): Mat
    fun getVerticalClose(): Mat

    fun getMean(): MatOfDouble
    fun getStd(): MatOfDouble
    fun getKernel(): Mat
    fun getKernel2(): Mat
    fun getHorizontalKernel(): Mat
    fun getVerticalKernel(): Mat
    fun getHull(): MatOfInt
    fun getHullPoints(): MatOfPoint2f
    fun getApprox(): MatOfPoint2f

    // Adaptive threshold pipeline
    fun getAdaptiveBinary(): Mat

    fun releaseAll()
}

class MatBundle : IMatBundle {
    private var _gray: Mat = Mat()
    private var _blurred: Mat = Mat()
    private var _enhanced: Mat = Mat()
    private var _morph: Mat = Mat()
    private var _temp: Mat = Mat()
    private var _edges: Mat = Mat()
    private var _morphAdd: Mat = Mat()
    private var _hierarchy: Mat = Mat()

    // Directional line suppression (bright environment)
    private var _grayGaussian: Mat = Mat()
    private var _horizontalClose: Mat = Mat()
    private var _verticalClose: Mat = Mat()

    // Pooled per-frame temporaries (reused, not reallocated each frame)
    private var _mean: MatOfDouble = MatOfDouble()
    private var _std: MatOfDouble = MatOfDouble()
    private var _kernel: Mat = Mat()
    private var _kernel2: Mat = Mat()
    private var _horizontalKernel: Mat = Mat()
    private var _verticalKernel: Mat = Mat()
    private var _hull: MatOfInt = MatOfInt()
    private var _hullPoints: MatOfPoint2f = MatOfPoint2f()
    private var _approx: MatOfPoint2f = MatOfPoint2f()

    // Adaptive threshold pipeline
    private var _adaptiveBinary: Mat = Mat()

    override fun getGray(): Mat = _gray
    override fun getBlurred(): Mat = _blurred
    override fun getEnhanced(): Mat = _enhanced
    override fun getMorph(): Mat = _morph
    override fun getTemp(): Mat = _temp
    override fun getEdges(): Mat = _edges
    override fun getMorphAdd(): Mat = _morphAdd
    override fun getHierarchy(): Mat = _hierarchy

    override fun getMean(): MatOfDouble = _mean
    override fun getStd(): MatOfDouble = _std
    override fun getKernel(): Mat = _kernel
    override fun getKernel2(): Mat = _kernel2
    override fun getHorizontalKernel(): Mat = _horizontalKernel
    override fun getVerticalKernel(): Mat = _verticalKernel
    override fun getHull(): MatOfInt = _hull
    override fun getHullPoints(): MatOfPoint2f = _hullPoints
    override fun getApprox(): MatOfPoint2f = _approx

    override fun getAdaptiveBinary(): Mat = _adaptiveBinary

    override fun getGrayGaussian(): Mat = _grayGaussian
    override fun getHorizontalClose(): Mat = _horizontalClose
    override fun getVerticalClose(): Mat = _verticalClose

    override fun releaseAll() {
        _gray.release()
        _blurred.release()
        _enhanced.release()
        _morph.release()
        _temp.release()
        _edges.release()
        _morphAdd.release()
        _hierarchy.release()

        _grayGaussian.release()
        _horizontalClose.release()
        _verticalClose.release()

        _mean.release()
        _std.release()
        _kernel.release()
        _kernel2.release()
        _horizontalKernel.release()
        _verticalKernel.release()
        _hull.release()
        _hullPoints.release()
        _approx.release()

        _adaptiveBinary.release()
    }
}
