package io.github.iostreamchik.scanner

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import androidx.core.graphics.createBitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

fun Int.coerceOdd(): Int {
    return if (this % 2 == 0) this + 1 else this
}
fun ImageProxy.toMatRGBA(): Mat {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val nv21 = ByteArray(ySize + ySize / 2)

    val yRowStride = planes[0].rowStride
    val vRowStride = planes[2].rowStride
    val uRowStride = planes[1].rowStride
    val uvPixelStride = planes[1].pixelStride

    // Copy Y channel
    if (yRowStride == width) {
        yBuffer.get(nv21, 0, ySize)
    } else {
        for (row in 0 until height) {
            yBuffer.position(row * yRowStride)
            yBuffer.get(nv21, row * width, width)
        }
    }

    // Copy UV channels (interleaved NV21)
    val uvHeight = height / 2
    val uvWidth = width / 2
    var pos = ySize
    for (row in 0 until uvHeight) {
        for (col in 0 until uvWidth) {
            val v = vBuffer.get(row * vRowStride + col * uvPixelStride)
            val u = uBuffer.get(row * uRowStride + col * uvPixelStride)
            nv21[pos++] = v
            nv21[pos++] = u
        }
    }

    val yuv = Mat(height + height / 2, width, CvType.CV_8UC1)
    yuv.put(0, 0, nv21)

    val rgba = Mat()
    Imgproc.cvtColor(yuv, rgba, Imgproc.COLOR_YUV2RGBA_NV21, 4)
    yuv.release()
    return rgba
}

fun Mat.fixRotation(imageProxy: ImageProxy): Mat {
    val rotation = imageProxy.imageInfo.rotationDegrees
    return when (rotation) {
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