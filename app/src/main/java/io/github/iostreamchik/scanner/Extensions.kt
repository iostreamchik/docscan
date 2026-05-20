package io.github.iostreamchik.scanner

import android.graphics.Bitmap
import android.graphics.ImageFormat
import androidx.camera.core.ImageProxy
import androidx.core.graphics.createBitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer

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

    // Step 1: Allocate a single Master Mat to hold the raw YUV data layout matching rowStride
    // We allocation based on yRowStride instead of 'width' to completely absorb the hardware padding
    val rawYuvMat = Mat(height + height / 2, yRowStride, CvType.CV_8UC1)

    // Create view headers inside rawYuvMat for efficient zero-copy data insertions
    val yMatHeader = rawYuvMat.submat(0, height, 0, yRowStride)
    val uvMatHeader = rawYuvMat.submat(height, height + height / 2, 0, yRowStride)

    // Step 2: Fill the Y channel using bulk heap allocations (Bypasses loops)
    val ySize = yBuffer.remaining()
    val yByteArray = ByteArray(ySize)
    yBuffer.get(yByteArray)
    yMatHeader.put(0, 0, yByteArray)

    // Step 3: Parse and compact the UV Channel depending on the hardware architecture
    if (uvPixelStride == 2) {
        // NV21 or NV12 interleaved format.
        // Typically, the V buffer begins exactly 1 byte before or after the U buffer.
        val uvSize = vBuffer.remaining()
        val uvByteArray = ByteArray(uvSize)
        vBuffer.get(uvByteArray)

        // Put the raw interleaved block directly into the UV submat section
        uvMatHeader.put(0, 0, uvByteArray)
    } else {
        // Fallback for rare devices where planes are entirely planar/separated (uvPixelStride == 1)
        val uvSize = (yRowStride / 2) * (height / 2)
        val uByteArray = ByteArray(uBuffer.remaining())
        val vByteArray = ByteArray(vBuffer.remaining())
        uBuffer.get(uByteArray)
        vBuffer.get(vByteArray)

        // Manual reconstruction utilizing fast native matrix operations
        val uMat = Mat(height / 2, uRowStride, CvType.CV_8UC1)
        uMat.put(0, 0, uByteArray)
        val vMat = Mat(height / 2, vRowStride, CvType.CV_8UC1)
        vMat.put(0, 0, vByteArray)

        // Interleave via OpenCV's native channels merge (Highly optimized vs JVM loops)
        val list = listOf(vMat, uMat)
        val mergedUV = Mat()
        Core.merge(list, mergedUV)

        // Copy the cleanly merged native structure into our layout
        mergedUV.copyTo(uvMatHeader.submat(0, height / 2, 0, mergedUV.cols()))

        uMat.release()
        vMat.release()
        mergedUV.release()
    }

    // Step 4: Convert the padded YUV matrix to RGBA format
    val paddedRgba = Mat()
    Imgproc.cvtColor(rawYuvMat, paddedRgba, Imgproc.COLOR_YUV2RGBA_NV21, 4)

    // Step 5: Crop out the hardware padding to isolate pure pixels and eliminate the green border
    val finalRgba = if (yRowStride != width) {
        val roi = org.opencv.core.Rect(0, 0, width, height)
        Mat(paddedRgba, roi) // Submat crop out padding
    } else {
        paddedRgba
    }

    // Step 6: Native Memory Cleanup to avoid Out-Of-Memory (OOM) freezing
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
        90 -> this.rotate90Clockwise()
        270 -> this.rotate90CounterClockwise()
        180 -> {
            val tmp = Mat()
            Core.flip(this, tmp, -1)  // rotate 180
            tmp
        }

        else -> this
    }
}

fun Mat.toBitmap(): Bitmap {
    val bmp = createBitmap(this.cols(), this.rows())
    Utils.matToBitmap(this, bmp)
    return bmp
}

fun Mat.rotate90Clockwise(): Mat {
    val dst = Mat()
    Core.transpose(this, dst)  // transpose flips rows & cols
    Core.flip(dst, dst, 1)     // flip around y-axis
    return dst
}

fun Mat.rotate90CounterClockwise(): Mat {
    val dst = Mat()
    Core.transpose(this, dst)
    Core.flip(dst, dst, 0)     // flip around x-axis
    return dst
}

fun Mat.enhanceDocument(): Mat {
    val rgb = Mat()
    val lab = Mat()
    val result = Mat()
    val channels = ArrayList<Mat>()

    // 1️⃣ RGBA → RGB (remove alpha)
    Imgproc.cvtColor(this, rgb, Imgproc.COLOR_RGBA2RGB)

    // 2️⃣ RGB → LAB
    Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)

    // 3️⃣ Split channels
    Core.split(lab, channels)

    val l = channels[0]

    // 4️⃣ Shadow removal (optional but powerful)
    val illumination = Mat()
    Imgproc.GaussianBlur(l, illumination, Size(55.0, 55.0), 0.0)
    Core.divide(l, illumination, l, 255.0)

    // 5️⃣ CLAHE (contrast)
    val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
    clahe.apply(l, l)

    // 6️⃣ Merge back
    channels[0] = l
    Core.merge(channels, lab)

    // 7️⃣ LAB → RGB
    Imgproc.cvtColor(lab, result, Imgproc.COLOR_Lab2RGB)

    // 8️⃣ Optional sharpening
    val blurred = Mat()
    Imgproc.GaussianBlur(result, blurred, Size(0.0, 0.0), 2.0)

    Core.addWeighted(result, 1.3, blurred, -0.3, 0.0, result)

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