package io.github.iostreamchik.scanner.opencv

import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

object OpenCVAdapter {

    fun resizeToGray(source: Mat, width: Int, height: Int, gray: Mat) {
        val smallMat = Mat()
        Imgproc.resize(source, smallMat, Size(width.toDouble(), height.toDouble()))
        Imgproc.cvtColor(smallMat, gray, Imgproc.COLOR_RGBA2GRAY)
        smallMat.release()
    }

    fun getAverageBrightness(image: Mat, bundle: IMatBundle): Double {
        Core.meanStdDev(image, bundle.getMean(), bundle.getStd())
        return bundle.getMean().toArray()[0]
    }

    fun getStdDev(image: Mat, bundle: IMatBundle): Double {
        Core.meanStdDev(image, bundle.getMean(), bundle.getStd())
        return bundle.getStd().toArray()[0]
    }

    fun applyClahe(source: Mat, dest: Mat, clipLimit: Double, tileSize: Double) {
        val clahe = Imgproc.createCLAHE(clipLimit, Size(tileSize, tileSize))
        clahe.apply(source, dest)
    }

    fun createRectKernel(size: Size, kernel: Mat) {
        Imgproc.getStructuringElement(Imgproc.MORPH_RECT, size).also { created ->
            kernel.release()
            created.copyTo(kernel)
        }
    }

    fun morphClose(source: Mat, dest: Mat, kernel: Mat) {
        Imgproc.morphologyEx(source, dest, Imgproc.MORPH_CLOSE, kernel)
    }

    fun findContours(image: Mat, hierarchy: Mat): List<MatOfPoint> {
        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(
            image,
            contours,
            hierarchy,
            Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        return contours
    }

    fun isRectangle(approx: MatOfPoint2f, toleranceDegrees: Double = 15.0): Boolean {
        return io.github.iostreamchik.scanner.isRectangle(approx, toleranceDegrees)
    }
}
