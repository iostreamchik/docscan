package io.github.iostreamchik.scanner

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import io.github.iostreamchik.scanner.opencv.IMatBundle
import io.github.iostreamchik.scanner.opencv.PipelineParams
import java.lang.Math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.sqrt

const val PROCESS_WIDTH_SIMPLE = 640

class DocumentDetectorOpenCV5(
    private val matBundle: IMatBundle
) : IDocumentDetector {

    private var _smoothedHigh = -1.0

    override fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat {
        Log.d("DocScan5", "=== preprocess START: ${scaledWidth}x${scaledHeight} ===")

        val smallMat = Mat()
        Imgproc.resize(rawMat, smallMat, Size(scaledWidth.toDouble(), scaledHeight.toDouble()))
        Imgproc.cvtColor(smallMat, matBundle.getGray(), Imgproc.COLOR_RGBA2GRAY)
        smallMat.release()

        val blurKsize = params.medianBlurKsize.coerceAtLeast(3)
        Imgproc.medianBlur(matBundle.getGray(), matBundle.getBlurred(), blurKsize)

        // --- CLAHE (auto when params is Auto, user-configured) ---
        val useAutoClahe = params.isClaheAuto
        val claheClipLimit: Double = if (useAutoClahe) {
            Core.meanStdDev(matBundle.getBlurred(), matBundle.getMean(), matBundle.getStd())
            val brightness = matBundle.getMean().toArray()[0].coerceIn(20.0, 200.0)
            val dimBoost = if (brightness < 80.0) {
                100.0 / (brightness + 10.0)
            } else {
                0.0
            }
            val brightBoost = if (brightness > 130.0) {
                (brightness - 130.0) / 30.0
            } else {
                0.0
            }
            (1.5 + dimBoost + brightBoost).coerceIn(1.0, 4.0)
        } else {
            params.claheClipLimit.toDouble().coerceIn(1.0, 4.0)
        }
        Log.d("DocScan5", "  CLAHE: clipLimit=${"%.2f".format(claheClipLimit)}, useAutoClahe=$useAutoClahe")
        val tileSize = params.claheTileSize.coerceAtLeast(8).toDouble()
        val clahe = Imgproc.createCLAHE(claheClipLimit, Size(tileSize, tileSize))
        clahe.apply(matBundle.getBlurred(), matBundle.getEnhanced())

        // Adaptive Canny thresholds via Sobel gradient + Otsu + EMA smoothing.
        // Sobel on the pre-Canny blurred image directly matches what Canny measures.
        Imgproc.Sobel(matBundle.getEnhanced(), matBundle.getSobelX(), CvType.CV_32F, 1, 0, 3)
        Imgproc.Sobel(matBundle.getEnhanced(), matBundle.getSobelY(), CvType.CV_32F, 0, 1, 3)

        Core.convertScaleAbs(matBundle.getSobelX(), matBundle.getOtsuThreshold(), 1.0, 0.0)
        Core.convertScaleAbs(matBundle.getSobelY(), matBundle.getTemp(), 1.0, 0.0)
        Core.add(matBundle.getOtsuThreshold(), matBundle.getTemp(), matBundle.getOtsuThreshold())

        val rawOtsu = Imgproc.threshold(
            matBundle.getOtsuThreshold(),
            matBundle.getTemp(),
            0.0, 255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )

        val emaAlpha = 0.15
        val thresholdFloor = 10.0
        val thresholdCeiling = 80.0
        val lowHighRatio = 0.25

        val smoothedHigh = if (_smoothedHigh < 0.0) {
            rawOtsu
        } else {
            (emaAlpha * rawOtsu) + ((1.0 - emaAlpha) * _smoothedHigh)
        }
        _smoothedHigh = smoothedHigh

        val cannyHigh = smoothedHigh.coerceIn(thresholdFloor, thresholdCeiling)
        val cannyLow = cannyHigh * lowHighRatio

        Log.d("DocScan5", "  Adaptive Canny: high=${"%.1f".format(cannyHigh)}, low=${"%.1f".format(cannyLow)}")

        Imgproc.Canny(matBundle.getEnhanced(), matBundle.getEdges(), cannyLow, cannyHigh)

        // Copy edges to morph so ViewModel can access it for bitmap conversion
        matBundle.getEdges().copyTo(matBundle.getMorph())

        Log.d("DocScan5", "=== preprocess END ===")
        return matBundle.getMorph()
    }

    override fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        params: PipelineParams
    ): MatOfPoint? {
        Log.d("DocScan5", "=== detectQuad START ===")

        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(
            morphImage,
            contours,
            matBundle.getHierarchy(),
            Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        Log.d("DocScan5", "  found ${contours.size} contours")

        val frameArea = scaledWidth * scaledHeight
        val minArea = frameArea * 0.01
        val candidates = mutableListOf<MatOfPoint>()
        val approx = matBundle.getApprox()

        for (contour in contours) {
            val area = abs(Geometry.contourArea(contour))
            if (area < minArea) continue
            if (contour.total() < 10) continue

            val pts2f = MatOfPoint2f(*contour.toArray().map { org.opencv.core.Point(it.x, it.y) }.toTypedArray())
            val peri = Geometry.arcLength(pts2f, true)
            Geometry.approxPolyDP(pts2f, approx, 0.02 * peri, true)

            if (approx.total() == 4L && isRectangle(approx)) {
                val scaleX = originalWidth.toDouble() / scaledWidth
                val scaleY = originalHeight.toDouble() / scaledHeight
                val scaledPoints = approx.toArray().map { Point(it.x * scaleX, it.y * scaleY) }
                candidates.add(MatOfPoint(*scaledPoints.toTypedArray()))
            }
        }

        val best = candidates.maxByOrNull { abs(Geometry.contourArea(it)) }
        val result = best?.let { MatOfPoint(*it.toArray()) }
        candidates.forEach { it.release() }

        Log.d("DocScan5", "  candidates=${candidates.size}, result=${if (result != null) "found" else "null"}")
        Log.d("DocScan5", "=== detectQuad END ===")
        return result
    }

    override fun validateQuadSize(
        quad: MatOfPoint,
        originalWidth: Int,
        originalHeight: Int
    ): Boolean {
        val rect = Geometry.boundingRect(quad)
        val quadArea = rect.width * rect.height
        val frameArea = originalWidth * originalHeight
        return quadArea <= frameArea * 0.95
    }

    companion object {
        fun computeAutoClaheClipLimit(avgBrightness: Double): Double {
            val brightness = avgBrightness.coerceIn(20.0, 200.0)
            val dimBoost = if (brightness < 80.0) {
                100.0 / (brightness + 10.0)
            } else {
                0.0
            }
            val brightBoost = if (brightness > 130.0) {
                (brightness - 130.0) / 30.0
            } else {
                0.0
            }
            return (1.5 + dimBoost + brightBoost).coerceIn(1.0, 4.0)
        }

        fun isRectangle(approx: MatOfPoint2f): Boolean {
            val pts = approx.toArray()
            var maxDeviation = 0.0
            for (i in 0..3) {
                val angle = computeAngle(
                    pts[(i + 1) % 4],
                    pts[(i + 3) % 4],
                    pts[i]
                )
                maxDeviation = max(maxDeviation, abs(90 - angle))
            }
            return maxDeviation < 20
        }

        fun computeAngle(p1: Point, p2: Point, center: Point): Double {
            val dx1 = p1.x - center.x
            val dy1 = p1.y - center.y
            val dx2 = p2.x - center.x
            val dy2 = p2.y - center.y
            val dot = dx1 * dx2 + dy1 * dy2
            val norm1 = sqrt(dx1 * dx1 + dy1 * dy1)
            val norm2 = sqrt(dx2 * dx2 + dy2 * dy2)
            return acos(dot / (norm1 * norm2)) * 180.0 / PI
        }
    }
}
