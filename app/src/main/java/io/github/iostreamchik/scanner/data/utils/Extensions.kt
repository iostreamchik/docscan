package io.github.iostreamchik.scanner.data.utils

import android.graphics.Bitmap
import android.graphics.ImageFormat
import androidx.camera.core.ImageProxy
import androidx.core.graphics.createBitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer

/**
 * Calculates the correct output dimensions for a perspective-transformed document.
 *
 * Takes the four sorted corners (top-left, top-right, bottom-right, bottom-left)
 * and computes width/height from the maximum edge distances. This handles perspective
 * distortion where the top edge may appear shorter than the bottom edge (or vice versa).
 *
 * @return Pair of (width, height) in pixels for the warped output image
 */

fun ImageProxy.toMatRGBA(): Mat {
    // Safety check ensuring correct format
    if (format != ImageFormat.YUV_420_888) {
        throw IllegalArgumentException("Unsupported image format: $format. Expected YUV_420_888")
    }

    val width = width
    val height = height

    val yPlane = planes[0]
    val uPlane = planes[1]
    val vPlane = planes[2]

    val yBuffer: ByteBuffer = yPlane.buffer
    val uBuffer: ByteBuffer = uPlane.buffer
    val vBuffer: ByteBuffer = vPlane.buffer

    // Clear buffer positions to guarantee reproducible reads
    yBuffer.rewind()
    uBuffer.rewind()
    vBuffer.rewind()

    val yRowStride = yPlane.rowStride
    val uRowStride = uPlane.rowStride
    val vRowStride = vPlane.rowStride
    val uvPixelStride = uPlane.pixelStride

    val rawYuvMat = Mat(height + height / 2, yRowStride, CvType.CV_8UC1)

    val yMatHeader = rawYuvMat.submat(0, height, 0, yRowStride)
    val uvMatHeader = rawYuvMat.submat(height, height + height / 2, 0, yRowStride)

    val ySize = yBuffer.remaining()
    val yByteArray = ByteArray(ySize)
    yBuffer.get(yByteArray)
    yMatHeader.put(0, 0, yByteArray)

    if (uvPixelStride == 2) {
        val uvSize = vBuffer.remaining()
        val uvByteArray = ByteArray(uvSize)
        vBuffer.get(uvByteArray)

        uvMatHeader.put(0, 0, uvByteArray)
    } else {
        val uByteArray = ByteArray(uBuffer.remaining())
        val vByteArray = ByteArray(vBuffer.remaining())
        uBuffer.get(uByteArray)
        vBuffer.get(vByteArray)

        val uMat = Mat(height / 2, uRowStride, CvType.CV_8UC1)
        uMat.put(0, 0, uByteArray)
        val vMat = Mat(height / 2, vRowStride, CvType.CV_8UC1)
        vMat.put(0, 0, vByteArray)

        val list = listOf(vMat, uMat)
        val mergedUV = Mat()
        Core.merge(list, mergedUV)

        mergedUV.copyTo(uvMatHeader.submat(0, height / 2, 0, mergedUV.cols()))

        uMat.release()
        vMat.release()
        mergedUV.release()
    }

    val paddedRgba = Mat()
    Imgproc.cvtColor(rawYuvMat, paddedRgba, Imgproc.COLOR_YUV2RGBA_NV21, 4)

    val finalRgba = if (yRowStride != width) {
        val roi = Rect(0, 0, width, height)
        Mat(paddedRgba, roi).clone()
    } else {
        paddedRgba
    }

    yMatHeader.release()
    uvMatHeader.release()
    rawYuvMat.release()
    if (yRowStride != width) {
        paddedRgba.release()
    }

    return finalRgba
}

fun Mat.fixRotation(rotationDegrees: Int): Mat {
    return when (rotationDegrees) {
        0 -> this
        90 -> this.rotate90Clockwise()
        270 -> this.rotate90CounterClockwise()
        180 -> {
            val tmp = Mat()
            Core.flip(this, tmp, -1)  // rotate 180
            tmp
        }

        else -> this.clone()
    }
}

fun Mat.toBitmap(): Bitmap {
    if (empty()) {
        return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
    }
    val source = when (this.channels()) {
        1 -> {
            val rgb = Mat()
            Imgproc.cvtColor(this, rgb, Imgproc.COLOR_GRAY2RGB)
            rgb
        }
        4 -> {
            val rgb = Mat()
            Imgproc.cvtColor(this, rgb, Imgproc.COLOR_RGBA2RGB)
            rgb
        }
        else -> this
    }
    val bmp = createBitmap(source.cols(), source.rows())
    Utils.matToBitmap(source, bmp)
    if (source !== this) source.release()
    return bmp
}

fun Mat.toBitmap(width: Int, height: Int): Bitmap {
    if (empty()) {
        return createBitmap(width, height)
    }
    val resized = Mat()
    Imgproc.resize(this, resized, Size(width.toDouble(), height.toDouble()))
    val result = resized.toBitmap()
    resized.release()
    return result
}

fun Mat.rotate90Clockwise(): Mat {
    val dst = Mat()
    Core.transpose(this, dst)
    Core.flip(dst, dst, 1)
    return dst
}

fun Mat.rotate90CounterClockwise(): Mat {
    val dst = Mat()
    Core.transpose(this, dst)
    Core.flip(dst, dst, 0)
    return dst
}

fun Mat.enhanceDocument(): Mat {
    val rgb = Mat()
    val lab = Mat()
    val result = Mat()
    val channels = ArrayList<Mat>()
    val illumination = Mat()

    try {
        // RGBA → RGB (remove alpha)
        Imgproc.cvtColor(this, rgb, Imgproc.COLOR_RGBA2RGB)

        // RGB → LAB
        Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)

        // Split channels
        Core.split(lab, channels)

        val l = channels[0]

        // Shadow removal (optional but powerful)
        Imgproc.GaussianBlur(l, illumination, Size(55.0, 55.0), 0.0)
        Core.divide(l, illumination, l, 255.0)

        // CLAHE (contrast)
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(l, l)

        // Merge back
        channels[0] = l
        Core.merge(channels, lab)

        // LAB → RGB
        Imgproc.cvtColor(lab, result, Imgproc.COLOR_Lab2RGB)

        // Optional sharpening
        val sharpened = result.sharpen()
        sharpened.copyTo(result)
        sharpened.release()
    } finally {
        channels.forEach { it.release() }
        rgb.release()
        lab.release()
        illumination.release()
    }

    return result
}

fun Mat.sharpen(): Mat {
    val blurred = Mat()
    Imgproc.GaussianBlur(this, blurred, Size(0.0, 0.0), 2.0)
    
    val sharpened = Mat()
    Core.addWeighted(this, 1.3, blurred, -0.3, 0.0, sharpened)
    
    blurred.release()
    return sharpened
}
