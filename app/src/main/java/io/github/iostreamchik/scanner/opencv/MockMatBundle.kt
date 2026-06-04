package io.github.iostreamchik.scanner.opencv

import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint2f

/**
 * No-op MatBundle for Compose preview. All Mats are empty (0x0) and
 * releaseAll() is a safe no-op. Lazily initialized to avoid
 * UnsatisfiedLinkError when OpenCV native libs aren't loaded (preview).
 */
class MockMatBundle : IMatBundle {
    private val emptyMat: Mat by lazy { Mat() }

    override fun getGray(): Mat = emptyMat
    override fun getBlurred(): Mat = emptyMat
    override fun getEnhanced(): Mat = emptyMat
    override fun getMorph(): Mat = emptyMat
    override fun getTemp(): Mat = emptyMat
    override fun getEdges(): Mat = emptyMat
    override fun getMorphAdd(): Mat = emptyMat
    override fun getHierarchy(): Mat = emptyMat

    override fun getMean(): MatOfDouble = MatOfDouble()
    override fun getStd(): MatOfDouble = MatOfDouble()
    override fun getKernel(): Mat = emptyMat
    override fun getKernel2(): Mat = emptyMat
    override fun getHorizontalKernel(): Mat = emptyMat
    override fun getVerticalKernel(): Mat = emptyMat
    override fun getHull(): MatOfInt = MatOfInt()
    override fun getHullPoints(): MatOfPoint2f = MatOfPoint2f()
    override fun getApprox(): MatOfPoint2f = MatOfPoint2f()

    override fun getGrayGaussian(): Mat = emptyMat
    override fun getHorizontalClose(): Mat = emptyMat
    override fun getVerticalClose(): Mat = emptyMat

    override fun releaseAll() {
        // No-op for preview — empty Mats have no native resources
    }
}
