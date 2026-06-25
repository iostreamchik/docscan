package io.github.iostreamchik.scanner.opencv

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

class CannyThresholdCalculatorV3(private val matBundle: IMatBundle) : ICannyThresholdCalculator {

    private val gradX = Mat()
    private val gradY = Mat()

    private val medianSigma = 0.33

    fun computeGradientOtsu(grayMat: Mat): Double {
        Imgproc.Sobel(grayMat, gradX, CvType.CV_32F, 1, 0, 3)
        Imgproc.Sobel(grayMat, gradY, CvType.CV_32F, 0, 1, 3)

        Core.convertScaleAbs(gradX, matBundle.getOtsuThreshold(), 1.0, 0.0)
        Core.convertScaleAbs(gradY, matBundle.getTemp(), 1.0, 0.0)
        Core.add(matBundle.getOtsuThreshold(), matBundle.getTemp(), matBundle.getOtsuThreshold())

        return Imgproc.threshold(
            matBundle.getOtsuThreshold(),
            matBundle.getTemp(),
            0.0,
            255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )
    }

    fun computeGradientOtsuWithRatio(grayMat: Mat, lowHighRatio: Double): Pair<Double, Double> {
        val high = computeGradientOtsu(grayMat)
        return Pair(high * lowHighRatio, high) // Standardized to: Pair(lower, upper)
    }

    fun computeIntensityOtsu(grayMat: Mat): Double {
        return Imgproc.threshold(
            grayMat,
            matBundle.getTemp(),
            0.0,
            255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )
    }

    fun computeIntensityOtsuWithRatio(grayMat: Mat, lowHighRatio: Double): Pair<Double, Double> {
        val high = computeIntensityOtsu(grayMat)
        return Pair(high * lowHighRatio, high) // Standardized to: Pair(lower, upper)
    }

    override fun computeThreshold(grayMat: Mat): Pair<Double, Double> {
        return computeIntensityOtsuWithRatio(grayMat, 0.33)
    }

    override fun reset() {
        val zero = Scalar(0.0)
        gradX.setTo(zero)
        gradY.setTo(zero)
    }

    fun computeMedianBased(grayMat: Mat): Pair<Double, Double> {
        return computeMedianBasedWithSigma(grayMat, medianSigma)
    }

    fun computeMedianBasedWithSigma(grayMat: Mat, sigma: Double): Pair<Double, Double> {
        val median = computeMedian(grayMat)

        val lower = maxOf(0.0, (1.0 - sigma) * median)
        val upper = minOf(255.0, (1.0 + sigma) * median)

        return Pair(lower, upper) // Standardized to: Pair(lower, upper)
    }

    fun computeMedian(grayMat: Mat): Double {
        require(grayMat.channels() == 1) { "computeMedian requires a single-channel (CV_8UC1) Mat" }

        val totalPixels = grayMat.total().toInt()
        val histogram = IntArray(256)
        val buffer = ByteArray(totalPixels)

        // Single JNI transition for the entire Mat payload
        grayMat.get(0, 0, buffer)

        for (i in 0 until totalPixels) {
            histogram[buffer[i].toInt() and 0xFF]++
        }

        val half = totalPixels / 2
        var cumulative = 0
        for (value in 0..255) {
            cumulative += histogram[value]
            if (cumulative > half) {
                return value.toDouble()
            }
        }

        return 255.0
    }

    override fun release() {
        gradX.release()
        gradY.release()
    }
}