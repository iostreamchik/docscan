package io.github.iostreamchik.scanner

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.core.Core
import org.opencv.core.MatOfDouble
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

    private val PROCESS_WIDTH = 640.0

    private val _filteredBitmap = MutableStateFlow<Bitmap?>(null)
    val filteredBitmap = _filteredBitmap.asStateFlow()

    private val _resultBitmap = MutableStateFlow<Bitmap?>(null)
    val resultBitmap = _resultBitmap.asStateFlow()

    private val _exposureStateFlow = MutableStateFlow("")
    val exposureStateFlow = _exposureStateFlow.asStateFlow()

    private val mats = io.github.iostreamchik.scanner.opencv.MatBundle()

    fun processFrame(imageProxy: ImageProxy): List<MatOfPoint> {
        val width = imageProxy.width
        val height = imageProxy.height
        lastFrameSize = Size(width.toDouble(), height.toDouble())

        val mat = imageProxy.toMatRGBA()
        val rotation = imageProxy.imageInfo.rotationDegrees

        val result = detectDocument(mat, rotation)

        if (result.isNotEmpty()) {
            result.forEach { updateHistory(it) }
            
            if (isStable()) {
                val fusedQuad = getFusedQuad()
                if (fusedQuad != null) {
                    val maxDimension = max(width, height)
                    val scale = PROCESS_WIDTH / maxDimension
                    val scaledWidth = width * scale
                    val scaledHeight = height * scale

                    val originalQuad = getOriginalResolutionQuad(
                        fusedQuad,
                        width,
                        height,
                        scaledWidth,
                        scaledHeight
                    )
                    Log.d("CameraViewModel", "\nFused Quad: ${fusedQuad.toArray().joinToString(", ")},\n Original Quad: ${originalQuad.toArray().joinToString(", ")}")
                    val warped = warpDocumentHighQuality(mat, fusedQuad, rotation)
                    _resultBitmap.value = warped
                    return listOf(fusedQuad)
                }
            }
        }
        return result
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


    private fun detectDocument(mat: Mat, rotation: Int): List<MatOfPoint> {
        val originalWidth = mat.cols()
        val originalHeight = mat.rows()
        val maxDimension = max(originalWidth, originalHeight)
        val scale = PROCESS_WIDTH / maxDimension
        val scaledWidth = (originalWidth * scale)
        val scaledHeight = (originalHeight * scale)

        try {
            val smallMat = Mat()
            Imgproc.resize(mat, smallMat, Size(scaledWidth, scaledHeight))

            // 1️⃣ Grayscale
            Imgproc.cvtColor(smallMat, mats.gray, Imgproc.COLOR_RGBA2GRAY)
            smallMat.release()

            val mean = MatOfDouble()
            val std = MatOfDouble()
            Core.meanStdDev(mats.gray, mean, std)
            val avgBrightness = mean.toArray()[0]
            val contrast = std.toArray()[0]
            mean.release()
            std.release()
            _exposureStateFlow.value = "${avgBrightness.toInt()}"

            // 2️⃣ Blur
            var ksize = (3.0 * scale).toInt()
            if (ksize % 2 == 0) ksize += 1
            Imgproc.medianBlur(mats.gray, mats.blurred, ksize)

            val clipLimit = when {
                avgBrightness < 40 -> 2.0
                avgBrightness < 80 -> 1.5
                avgBrightness < 120 -> 1.2
                else -> 1.0
            }
            val tileSize = when {
                avgBrightness < 80 -> 8.0
                avgBrightness < 120 -> 16.0
                else -> 1.0
            }
            val clahe = Imgproc.createCLAHE(clipLimit, Size(tileSize, tileSize))
            clahe.apply(mats.blurred, mats.enhanced)

            // 3️⃣ Adaptive Morph Close
            val kernelSize = when {
                avgBrightness < 70 -> 11.0
                avgBrightness < 100 -> 9.0
                else -> 5.0
            }
            val kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT,
                Size(kernelSize, kernelSize)
            )
            Imgproc.morphologyEx(mats.enhanced, mats.morph, Imgproc.MORPH_CLOSE, kernel)

            // 4️⃣ Otsu
            val otsu = Imgproc.threshold(
                mats.morph,
                mats.temp,
                0.0,
                255.0,
                Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
            )
            val high = otsu.coerceIn(70.0, 160.0)
            val low = (high * 0.45).coerceAtLeast(35.0)

            // 5️⃣ Canny
            Imgproc.Canny(mats.enhanced, mats.edges, low, high)

            // 6️⃣ Strong closing
            val size = Size(
                (5 * scale).coerceAtLeast(3.0),
                (5 * scale).coerceAtLeast(3.0)
            )
            val kernel2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, size)
            Imgproc.morphologyEx(mats.edges, mats.morph, Imgproc.MORPH_CLOSE, kernel2)
            kernel2.release()

            _filteredBitmap.value = mats.morph?.fixRotation(rotation)?.toBitmap()

            // 7️⃣ Find contours
            val contours = mutableListOf<MatOfPoint>()
            Imgproc.findContours(
                mats.morph,
                contours,
                mats.hierarchy,
                Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_SIMPLE
            )

            if (contours.isEmpty()) return emptyList()

            val frameArea = scaledWidth * scaledHeight
            val minArea = frameArea * 0.015
            val documentCandidates = mutableListOf<MatOfPoint>()

            for (contour in contours) {
                val area = Imgproc.contourArea(contour)
                if (area < minArea) continue

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

                val scaleX = originalWidth.toDouble() / scaledWidth
                val scaleY = originalHeight.toDouble() / scaledHeight
                val scaledPoints = approx.toArray().map { point ->
                    Point(point.x * scaleX, point.y * scaleY)
                }
                val quad = MatOfPoint(*scaledPoints.toTypedArray())

                if (!isRectangle(approx)) continue

                val scaledArea = area * (scaleX * scaleY)
                val rect = Imgproc.boundingRect(quad)
                val solidity = scaledArea / (rect.width * rect.height).toDouble()
                if (solidity < 0.3) continue

                documentCandidates.add(quad)
            }

            if (documentCandidates.isEmpty()) return emptyList()

            val best = documentCandidates.maxByOrNull {
                scoreContour(it, scaledWidth.toInt(), scaledHeight.toInt())
            }

            return best?.let { listOf(it) } ?: emptyList()

        } catch (e: Exception) {
            e.printStackTrace()
            return emptyList()
        } finally {
            mats.releaseAll()
        }
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

        return output.enhanceDocument().fixRotation(imageProxy.imageInfo.rotationDegrees).toBitmap()
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

    private fun warpDocumentHighQuality(src: Mat, quad: MatOfPoint, rotationDegrees: Int): Bitmap? {
        return try {
            val sorted = sortQuadPoints(quad.toArray().toList())
            val (tl, tr, br, bl) = sorted // Destructuring

            // Use original image dimensions for output size
            val outputWidth = src.cols().toDouble()
            val outputHeight = src.rows().toDouble()

            val srcPoints = MatOfPoint2f(tl, tr, br, bl)
            val dstPoints = MatOfPoint2f(
                Point(0.0, 0.0),
                Point(outputWidth, 0.0),
                Point(outputWidth, outputHeight),
                Point(0.0, outputHeight)
            )

            val transform = Imgproc.getPerspectiveTransform(srcPoints, dstPoints)
            val output = Mat()
            Imgproc.warpPerspective(src, output, transform, Size(outputWidth, outputHeight))
            // 7. Convert to Bitmap (applying your existing extensions for rotation/enhancement)
            // This ensures the "crop" looks like a real scanned document
            output
                .fixRotation(rotationDegrees)
                .sharpen()
                .toBitmap()
        } catch (e: Exception) {
            Log.e("CameraViewModel", "Warp error: ${e.message}")
            null
        }
    }

    private fun getOriginalResolutionQuad(
        scaledQuad: MatOfPoint,
        originalWidth: Int,
        originalHeight: Int,
        scaledWidth: Double,
        scaledHeight: Double
    ): MatOfPoint {

        // Calculate the ratio used to shrink the image
        val scaleX = originalWidth / scaledWidth
        val scaleY = originalHeight / scaledHeight

        // Map each point back to the original coordinate space
        val originalPoints = scaledQuad.toArray().map { point ->
            Point(point.x * scaleX, point.y * scaleY)
        }
        return MatOfPoint(*originalPoints.toTypedArray())
    }

    fun processPickedDocument(context: Context, uri: Uri) {
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, uri)) { decoder, _, _ ->
                        decoder.isMutableRequired = true
                    }
                } else {
                    MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
                }

                val mat = Mat()
                Utils.bitmapToMat(bitmap, mat)

                val originalWidth = mat.cols()
                val originalHeight = mat.rows()

                // Picked documents usually have 0 rotation
                val rotation = 0

                val result = detectDocument(mat, rotation)

                if (result.isNotEmpty()) {
                    val bestQuad = result.first()
                    val warped = warpDocumentHighQuality(mat, bestQuad, rotation)
                    _resultBitmap.value = warped ?: mat.enhanceDocument().toBitmap()
                } else {
                    _resultBitmap.value = mat.enhanceDocument().toBitmap()
                }

                mat.release()
                bitmap.recycle()

            } catch (e: Exception) {
                Log.e("CameraViewModel", "Error processing picked document", e)
            }
        }
    }

}
