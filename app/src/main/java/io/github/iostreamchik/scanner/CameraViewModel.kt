// This one is the best
package io.github.iostreamchik.scanner

import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.lang.Math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.sqrt


class CameraViewModel : ViewModel() {

    private val quadHistory = ArrayDeque<MatOfPoint>()
    private var lastFrameSize: Size? = null
    private val MAX_HISTORY = 10

    private val _filteredBitmap = MutableStateFlow<Bitmap?>(null)
    val filteredBitmap = _filteredBitmap.asStateFlow()

    private val _resultBitmap = MutableStateFlow<Bitmap?>(null)
    val resultBitmap = _resultBitmap.asStateFlow()

    val gray = Mat()
    val blurred = Mat()
    val enhanced = Mat()
    val morph = Mat()
    val temp = Mat()
    val edges = Mat()
    val morphAdd = Mat()
    val hierarchy = Mat()

    fun processFrame(imageProxy: ImageProxy): List<MatOfPoint> {
        lastFrameSize = Size(
            imageProxy.width.toDouble(),
            imageProxy.height.toDouble()
        )
        val mat = imageProxy.toMatRGBA()

        try {
            // 1️⃣ Grayscale
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)

            // 2️⃣ Blur
            val scale = imageProxy.width / 640.0
            // Determine kernel size: must be at least 3, must be odd
            var ksize = (3.0 * scale).toInt()
            if (ksize % 2 == 0) ksize += 1
            Imgproc.medianBlur(gray, blurred, ksize)

            // Add CLAHE for contrast enhancement
            val clahe = Imgproc.createCLAHE(1.0, Size(1.0, 1.0))
            clahe.apply(blurred, enhanced)

            // 3️⃣ Adaptive Morph Close (scale-aware)
            val kernelSize = (5 * scale).toInt().coerceAtLeast(5)
            val kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT,
                Size(kernelSize.toDouble(), kernelSize.toDouble())
            )
            Imgproc.morphologyEx(enhanced, morph, Imgproc.MORPH_CLOSE, kernel)

            // 4️⃣ Otsu for auto Canny
            val otsu = Imgproc.threshold(
                morph,
                temp,
                0.0,
                255.0,
                Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
            )

            val high = otsu
            val low = otsu * 0.4

            // 5️⃣ Canny (correct order: low, high)
            Imgproc.Canny(morph, edges, low, high)

            // 6️⃣ Strong closing to connect document edges
            val size = Size(
                (5 * scale).coerceAtLeast(3.0),
                (5 * scale).coerceAtLeast(3.0)
            )
            Log.d("CameraViewModel", "Size: $size scale: $scale otsu: [$low-$high]")
            val kernel2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, size)
            Imgproc.morphologyEx(edges, morphAdd, Imgproc.MORPH_DILATE, kernel2)


            _filteredBitmap.value = morphAdd.fixRotation(imageProxy).toBitmap()

            // 7️⃣ Find contours
            val contours = mutableListOf<MatOfPoint>()

            Imgproc.findContours(
                morphAdd,
                contours,
                hierarchy,
                Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_SIMPLE
            )

            if (contours.isEmpty()) return emptyList()

            val frameArea = imageProxy.width * imageProxy.height.toDouble()
            val minArea = frameArea * 0.015

            val documentCandidates = mutableListOf<MatOfPoint>()

            for (contour in contours) {

                val area = Imgproc.contourArea(contour)
                if (area < minArea) continue

                // Convex hull
                val hull = MatOfInt()
                Imgproc.convexHull(contour, hull)

                val contourArray = contour.toArray()
                val hullPoints = MatOfPoint()
                hullPoints.fromList(hull.toArray().map { contourArray[it] })

                val peri = Imgproc.arcLength(MatOfPoint2f(*hullPoints.toArray()), true)
                val approx = MatOfPoint2f()
                Imgproc.approxPolyDP(
                    MatOfPoint2f(*hullPoints.toArray()),
                    approx,
                    0.02 * peri,
                    true
                )

                if (approx.total() != 4L) continue

                val quad = MatOfPoint(*approx.toArray())

                // 🔷 Angle validation
                if (!isRectangle(approx)) continue

                // 🔷 Solidity check
                val rect = Imgproc.boundingRect(quad)
                val solidity = area / (rect.width * rect.height).toDouble()
                if (solidity < 0.3) continue

                documentCandidates.add(quad)
            }

            if (documentCandidates.isEmpty()) return emptyList()

            // ✅ Choose best by score
            val best = documentCandidates.maxByOrNull {
                scoreContour(it, imageProxy.width, imageProxy.height)
            }

            val result = best?.let { listOf(it) } ?: emptyList()
            result.forEach { updateHistory(it) }

            if (isStable()) {
                val fusedQuad = getFusedQuad()
                return if (fusedQuad != null) {
                    val warped = warpDocument(mat, fusedQuad, imageProxy)
                    _resultBitmap.value = warped
                    listOf(fusedQuad)
                } else emptyList()
            } else return emptyList()
        } catch (e: Exception) {
            e.printStackTrace()
            return emptyList()
        } finally {
            mat.release()
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

    private fun updateHistory(quad: MatOfPoint) {
        if (quadHistory.size >= MAX_HISTORY) {
            quadHistory.removeFirst()
        }
        quadHistory.addLast(quad)
    }

    private fun isStable(): Boolean {
        if (quadHistory.size < MAX_HISTORY) return false

        val frameSize = lastFrameSize ?: return false
        val quads = quadHistory.toList()

        var totalMovement = 0.0

        for (i in 1 until quads.size) {
            totalMovement += quadDistance(
                quads[i - 1].toSortedQuad(),
                quads[i].toSortedQuad(),
                frameSize.width,
                frameSize.height
            )
        }

        return (totalMovement / (quads.size - 1)) < 0.02
    }

    private fun MatOfPoint.toSortedQuad(): List<Point> {
        return sortQuadPoints(this.toArray().toList())
    }

    private fun getFusedQuad(): MatOfPoint? {
        if (quadHistory.isEmpty()) return null

        val sortedQuads = quadHistory.map { sortQuadPoints(it.toArray().toList()) }

        val averaged = Array(4) { Point(0.0, 0.0) }

        for (i in 0..3) {
            for (quad in sortedQuads) {
                averaged[i].x += quad[i].x
                averaged[i].y += quad[i].y
            }
            averaged[i].x /= sortedQuads.size
            averaged[i].y /= sortedQuads.size
        }

        return MatOfPoint(*averaged)
    }

    fun sortQuadPoints(points: List<Point>): List<Point> {
        require(points.size == 4) { "Quad must have exactly 4 points" }

        // 1️⃣ Compute centroid
        val centerX = points.sumOf { it.x } / 4.0
        val centerY = points.sumOf { it.y } / 4.0

        // 2️⃣ Sort by angle around centroid (clockwise)
        val sortedByAngle = points.sortedBy {
            atan2(it.y - centerY, it.x - centerX)
        }

        // 3️⃣ Now ensure consistent starting point (top-left first)
        // Top-left = smallest (x + y)
        val topLeftIndex = sortedByAngle
            .mapIndexed { index, p -> index to (p.x + p.y) }
            .minBy { it.second }
            .first

        // Rotate list so top-left is first
        return List(4) { i ->
            sortedByAngle[(topLeftIndex + i) % 4]
        }
    }

    fun quadDistance(
        quad1: List<Point>,
        quad2: List<Point>,
        frameWidth: Double,
        frameHeight: Double
    ): Double {

        if (quad1.size != 4 || quad2.size != 4) return Double.MAX_VALUE

        val diagonal = sqrt(frameWidth * frameWidth + frameHeight * frameHeight)

        var totalDistance = 0.0

        for (i in 0 until 4) {
            val dx = quad1[i].x - quad2[i].x
            val dy = quad1[i].y - quad2[i].y
            totalDistance += sqrt(dx * dx + dy * dy)
        }

        // average corner shift normalized by frame diagonal
        return (totalDistance / 4.0) / diagonal
    }


    private fun scoreContour(
        contour: MatOfPoint,
        width: Int,
        height: Int
    ): Double {

        val area = Imgproc.contourArea(contour)

        val center = Imgproc.boundingRect(contour).let {
            Point(it.x + it.width / 2.0, it.y + it.height / 2.0)
        }

        val frameCenter = Point(width / 2.0, height / 2.0)
        val centerDist = hypot(
            center.x - frameCenter.x,
            center.y - frameCenter.y
        )

        val maxDist = hypot(width / 2.0, height / 2.0)

        val centerScore = 1.0 - (centerDist / maxDist)

        return area * 0.7 + centerScore * 0.3 * width * height
    }

    private fun isRectangle(approx: MatOfPoint2f): Boolean {
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

        return maxDeviation < 50
    }

    private fun computeAngle(p1: Point, p2: Point, center: Point): Double {
        val dx1 = p1.x - center.x
        val dy1 = p1.y - center.y
        val dx2 = p2.x - center.x
        val dy2 = p2.y - center.y

        val dot = dx1 * dx2 + dy1 * dy2
        val norm1 = sqrt(dx1 * dx1 + dy1 * dy1)
        val norm2 = sqrt(dx2 * dx2 + dy2 * dy2)

        return acos(dot / (norm1 * norm2)) * 180.0 / PI
    }

    private fun warpDocument(src: Mat, quad: MatOfPoint, imageProxy: ImageProxy): Bitmap? {
        val sorted = quad.toSortedQuad()

        val tl = sorted[0]
        val tr = sorted[1]
        val br = sorted[2]
        val bl = sorted[3]

        val scaleFactor = 2.0

        // Compute width
        val widthA = hypot(br.x - bl.x, br.y - bl.y)
        val widthB = hypot(tr.x - tl.x, tr.y - tl.y)
        val maxWidth = (max(widthA, widthB) * scaleFactor).toInt()

        // Compute height
        val heightA = hypot(tr.x - br.x, tr.y - br.y)
        val heightB = hypot(tl.x - bl.x, tl.y - bl.y)
        val maxHeight = (max(heightA, heightB) * scaleFactor).toInt()

        val srcPoints = MatOfPoint2f(
            tl,
            tr,
            br,
            bl
        )

        val dstPoints = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(maxWidth.toDouble(), 0.0),
            Point(maxWidth.toDouble(), maxHeight.toDouble()),
            Point(0.0, maxHeight.toDouble())
        )

        val transform = Imgproc.getPerspectiveTransform(srcPoints, dstPoints)

        val output = Mat()
        Imgproc.warpPerspective(
            src,
            output,
            transform,
            Size(maxWidth.toDouble(), maxHeight.toDouble())
        )

        return output.enhanceDocument().fixRotation(imageProxy).toBitmap()
    }

    private fun distance(p1: Point, p2: Point): Double {
        val dx = p1.x - p2.x
        val dy = p1.y - p2.y
        return sqrt(dx * dx + dy * dy)
    }

    private fun warpDocument(
        src: Mat,
        quad: MatOfPoint
    ): Mat {

        val points = sortQuadPoints(quad.toArray().toList())

        val tl = points[0]
        val tr = points[1]
        val br = points[2]
        val bl = points[3]

        // Compute width
        val widthA = distance(br, bl)
        val widthB = distance(tr, tl)
        val maxWidth = max(widthA, widthB).toInt()

        // Compute height
        val heightA = distance(tr, br)
        val heightB = distance(tl, bl)
        val maxHeight = max(heightA, heightB).toInt()

        val srcMat = MatOfPoint2f(tl, tr, br, bl)

        val dstMat = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(maxWidth - 1.0, 0.0),
            Point(maxWidth - 1.0, maxHeight - 1.0),
            Point(0.0, maxHeight - 1.0)
        )

        val transform = Imgproc.getPerspectiveTransform(srcMat, dstMat)

        val output = Mat()
        Imgproc.warpPerspective(src, output, transform, Size(maxWidth.toDouble(), maxHeight.toDouble()))

        return output
    }

    fun scaleQuad(
        quad: List<Point>,
        fromW: Double,
        fromH: Double,
        toW: Double,
        toH: Double
    ): List<Point> {

        val scaleX = toW / fromW
        val scaleY = toH / fromH

        return quad.map {
            Point(it.x * scaleX, it.y * scaleY)
        }
    }
}