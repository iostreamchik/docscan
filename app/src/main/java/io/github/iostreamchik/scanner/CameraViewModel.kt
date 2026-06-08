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
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.lang.Math.PI
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.sqrt
import io.github.iostreamchik.scanner.opencv.IMatBundle
import io.github.iostreamchik.scanner.opencv.MatBundle
import org.opencv.core.MatOfDouble
import kotlin.math.min

class CameraViewModel(
    private val matBundle: IMatBundle = MatBundle()
) : ViewModel() {

    val cameraExecutor = Executors.newSingleThreadExecutor()

    private val quadHistory = ArrayDeque<MatOfPoint>()
    private var lastFrameSize: Size? = null
    private val MAX_HISTORY = 10
    private var frameCounter = 0
    private val STABILITY_CHECK_INTERVAL = 3

    private val PROCESS_WIDTH = 640.0

    // Cache: skip re-warping when the same quad is detected repeatedly
    private var lastWarpedQuadHash: Long = 0
    private var lastWarpedBitmap: Bitmap? = null

    private val _filteredBitmap = MutableStateFlow<Bitmap?>(null)
    val filteredBitmap = _filteredBitmap.asStateFlow()

    private val _originalBitmap = MutableStateFlow<Bitmap?>(null)
    val originalBitmap = _originalBitmap.asStateFlow()

    private val _resultBitmap = MutableStateFlow<Bitmap?>(null)
    val resultBitmap = _resultBitmap.asStateFlow()

    private val _exposureStateFlow = MutableStateFlow("")
    val exposureStateFlow = _exposureStateFlow.asStateFlow()

    private val _errorState = MutableStateFlow<String?>(null)
    val errorState = _errorState.asStateFlow()

    private var lastUiUpdateTime = 0L
    private val UI_UPDATE_THROTTLE_MS = 100L

    fun setError(message: String?) {
        _errorState.value = message
    }

    fun processFrame(imageProxy: ImageProxy): List<MatOfPoint> {
        val width = imageProxy.width
        val height = imageProxy.height
        lastFrameSize = Size(width.toDouble(), height.toDouble())

        val mat = imageProxy.toMatRGBA()
        val rotation = imageProxy.imageInfo.rotationDegrees

        val result = detectDocument(mat, rotation)

        if (result.isNotEmpty()) {
            if (isStable()) {
                val fusedQuad = getFusedQuad()
                if (fusedQuad != null) {
                    // Always add detection results to history — never add the fused quad
                    // itself, which would saturate the history with near-duplicate values
                    // and prevent getFusedQuad() from responding to new detections.
                    result.forEach { updateHistory(it) }
                    val quadHash = quadHash(fusedQuad)
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
                    Log.d(
                        "CameraViewModel",
                        "\nFused Quad: ${
                            fusedQuad.toArray().joinToString(", ")
                        },\n Original Quad: ${originalQuad.toArray().joinToString(", ")}"
                    )
                    // Warp only if the quad has changed — skip expensive op when hash matches
                    val warped = if (quadHash != lastWarpedQuadHash) {
                        warpDocumentHighQuality(mat, fusedQuad, rotation).also {
                            lastWarpedBitmap?.recycle()
                            lastWarpedBitmap = it
                            lastWarpedQuadHash = quadHash
                        }
                    } else {
                        lastWarpedBitmap
                    }
                    // Clone the bitmap before emitting to state flow so Compose gets
                    // its own independent copy that won't be affected by recycling.
                    _resultBitmap.value =
                        warped?.copy(android.graphics.Bitmap.Config.ARGB_8888, false)
                    // Clone: fusedQuad lives in quadHistory, caller owns the clone
                    return listOf(MatOfPoint(*fusedQuad.toArray()))
                }
            } else {
                // Accumulate during pre-stability bootstrapping
                result.forEach { updateHistory(it) }
            }
        }
        // Clone each result: objects also live in quadHistory, caller owns the clones
        return result.map { MatOfPoint(*it.toArray()) }
    }

    private fun updateHistory(quad: MatOfPoint) {
        if (quadHistory.size >= MAX_HISTORY) {
            quadHistory.removeFirst().release()
        }
        quadHistory.addLast(quad)
    }

    private fun isStable(): Boolean {
        if (quadHistory.size < MAX_HISTORY) return false
        if (++frameCounter % STABILITY_CHECK_INTERVAL != 0) return true

        val frameSize = lastFrameSize ?: return false
        val quads = quadHistory.toList()

        var totalMovement = 0.0
        var validPairs = 0

        for (i in 1 until quads.size) {
            val prevSorted = quads[i - 1].toSortedQuad()
            val currSorted = quads[i].toSortedQuad()

            // Skip if either quad is invalid
            if (prevSorted.isEmpty() || currSorted.isEmpty()) continue

            totalMovement += quadDistance(
                prevSorted,
                currSorted,
                frameSize.width,
                frameSize.height
            )
            validPairs++
        }

        // Need at least one valid pair to calculate stability
        return validPairs > 0 && (totalMovement / validPairs) < 0.02
    }

    private fun MatOfPoint.toSortedQuad(): List<Point> {
        val points = this.toArray().toList()
        if (points.size != 4) {
            Log.w(
                "CameraViewModel",
                "toSortedQuad: Invalid quad with ${points.size} points, returning empty list"
            )
            return emptyList()
        }
        return sortQuadPoints(points)
    }

    private fun getFusedQuad(): MatOfPoint? {
        if (quadHistory.isEmpty()) return null

        val validSortedQuads = quadHistory.mapNotNull { matOfPoint ->
            val points = matOfPoint.toArray().toList()
            if (points.size == 4) sortQuadPoints(points) else null
        }

        // Need at least some valid quads to fuse
        if (validSortedQuads.isEmpty()) return null

        val averaged = Array(4) { Point(0.0, 0.0) }

        for (i in 0..3) {
            for (quad in validSortedQuads) {
                averaged[i].x += quad[i].x
                averaged[i].y += quad[i].y
            }
            averaged[i].x /= validSortedQuads.size
            averaged[i].y /= validSortedQuads.size
        }

        return MatOfPoint(*averaged)
    }

    fun sortQuadPoints(points: List<Point>): List<Point> {
        if (points.size != 4) {
            Log.w(
                "CameraViewModel",
                "sortQuadPoints: Invalid quad with ${points.size} points, returning empty list"
            )
            return emptyList()
        }

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
            Imgproc.cvtColor(smallMat, matBundle.getGray(), Imgproc.COLOR_RGBA2GRAY)
            smallMat.release()

            Core.meanStdDev(matBundle.getGray(), matBundle.getMean(), matBundle.getStd())
            val avgBrightness = matBundle.getMean().toArray()[0]
            val contrast = matBundle.getStd().toArray()[0]
            Log.d(
                "Pipeline",
                "--- Pipeline start: brightness=$avgBrightness contrast=$contrast scale=$scale (${scaledWidth.toInt()}x${scaledHeight.toInt()}) ---"
            )

            val exposureTime = System.currentTimeMillis()
            if (exposureTime - lastUiUpdateTime >= UI_UPDATE_THROTTLE_MS) {
                _exposureStateFlow.value = "br: ${avgBrightness.toInt()} ct: ${contrast.toInt()}"
                lastUiUpdateTime = exposureTime
            }

            // 2️⃣ Median Blur (fixed ksize=5)
            Imgproc.medianBlur(matBundle.getGray(), matBundle.getBlurred(), 5)
            Log.d("Pipeline", "medianBlur: ksize=5")

            // 3️⃣ CLAHE (clipLimit=0.1, tileSize=8)
            val clahe = Imgproc.createCLAHE(0.1, Size(8.0, 8.0))
            clahe.apply(matBundle.getBlurred(), matBundle.getEnhanced())
            Log.d("Pipeline", "CLAHE: clipLimit=0.1 tileSize=8")

            // 4️⃣ Morph Close (kernelSize=3)
            Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT,
                Size(3.0, 3.0)
            ).also { kernel ->
                matBundle.getKernel().release()
                kernel.copyTo(matBundle.getKernel())
            }
            Imgproc.morphologyEx(matBundle.getEnhanced(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, matBundle.getKernel())

            // 4️⃣ Automatic Canny thresholds via Otsu/σ-based method
            // Adapted from PyImageSearch: zero-parameter automatic Canny edge detection
            // Uses image intensity + σ multiplier instead of hardcoded brightness breakpoints
            val sigma = 0.33
            val cannyHigh = min(255.0, max(30.0, avgBrightness * (1.0 + sigma)))
            val cannyLow = max(10.0, cannyHigh * 0.5)
            Log.d("Pipeline", "Auto Canny: High=$cannyHigh, Low=$cannyLow intensity=$avgBrightness sigma: $sigma")
            // Apply Canny to enhanced image (not heavily-blurred morph) to preserve edges
            Imgproc.Canny(matBundle.getEnhanced(), matBundle.getEdges(), cannyLow, cannyHigh)

            // 6️⃣ Strong Closing (kernelSize=7)
            var closeKsize = (7 * scale).coerceAtLeast(3.0).toInt()
            if (closeKsize % 2 == 0) closeKsize++
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(closeKsize.toDouble(), closeKsize.toDouble())).also { kernel2 ->
                matBundle.getKernel2().release()
                kernel2.copyTo(matBundle.getKernel2())
            }
            Imgproc.morphologyEx(matBundle.getEdges(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, matBundle.getKernel2())
            Log.d("Pipeline", "Strong Close: kernel=$closeKsize")

            // 7️⃣ Directional Suppression (kernelSize=10)
            val dirKsize = (10 * scale).coerceAtLeast(3.0).toInt()
            Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT,
                Size(dirKsize.toDouble(), 1.0)
            ).also { kernel ->
                matBundle.getHorizontalKernel().release()
                kernel.copyTo(matBundle.getHorizontalKernel())
            }
            Imgproc.morphologyEx(
                matBundle.getMorph(),
                matBundle.getHorizontalClose(),
                Imgproc.MORPH_CLOSE,
                matBundle.getHorizontalKernel()
            )

            Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT,
                Size(1.0, dirKsize.toDouble())
            ).also { kernel ->
                matBundle.getVerticalKernel().release()
                kernel.copyTo(matBundle.getVerticalKernel())
            }
            Imgproc.morphologyEx(
                matBundle.getHorizontalClose(),
                matBundle.getVerticalClose(),
                Imgproc.MORPH_CLOSE,
                matBundle.getVerticalKernel()
            )

            matBundle.getVerticalClose().copyTo(matBundle.getMorph())
            Log.d("Pipeline", "DirectionalSuppression: kernel=$dirKsize")

            // Always set filtered bitmap — camera screen doesn't display it, so no throttle needed.
            // File scan screen needs it and the shared throttle blocks it when exposure update runs first.
            _filteredBitmap.value = matBundle.getMorph().fixRotation(rotation).toBitmap()

            // 8️⃣ Find contours
            val contours = mutableListOf<MatOfPoint>()
            Imgproc.findContours(
                matBundle.getMorph(),
                contours,
                matBundle.getHierarchy(),
                Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_SIMPLE
            )

            if (contours.isEmpty()) return emptyList()

            val frameArea = scaledWidth * scaledHeight
            val minAreaFraction = 0.025
            val minArea = frameArea * minAreaFraction
            val documentCandidates = mutableListOf<MatOfPoint>()

            for (contour in contours) {
                val area = Imgproc.contourArea(contour)
                if (area < minArea) continue

                val hull = matBundle.getHull()
                matBundle.getHullPoints().release()
                matBundle.getHullPoints().create(0, 1, CvType.CV_32FC2)
                val approx = matBundle.getApprox()
                Imgproc.convexHull(contour, hull)
                val contourArray = contour.toArray()
                val hullPointList = hull.toArray().map { contourArray[it] }
                matBundle.getHullPoints().fromList(hullPointList.map { Point(it.x, it.y) })

                val peri = Imgproc.arcLength(matBundle.getHullPoints(), true)
                Imgproc.approxPolyDP(
                    matBundle.getHullPoints(),
                    approx,
                    0.015 * peri,
                    true
                )

                if (approx.total() != 4L) continue

                val scaleX = originalWidth.toDouble() / scaledWidth
                val scaleY = originalHeight.toDouble() / scaledHeight
                val scaledPoints = approx.toArray().map { point ->
                    Point(point.x * scaleX, point.y * scaleY)
                }
                val quad = MatOfPoint(*scaledPoints.toTypedArray())

                if (!isRectangle(approx)) {
                    quad.release()
                    continue
                }

                val scaledArea = area * (scaleX * scaleY)
                val rect = Imgproc.boundingRect(quad)
                val solidity = scaledArea / (rect.width * rect.height).toDouble()
                if (solidity < 0.3) {
                    quad.release()
                    continue
                }

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
            matBundle.releaseAll()
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

        // Area ratio: penalize if contour fills too much of the frame
        // (likely background texture rather than a document)
        val frameArea = width * height.toDouble()
        val areaRatio = area / frameArea
        val areaRatioScore = if (areaRatio > 0.5) 0.2 else 1.0

        return area * 0.5 + centerScore * 0.3 * width * height + areaRatioScore * 0.2 * width * height
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

    private fun distance(p1: Point, p2: Point): Double {
        val dx = p1.x - p2.x
        val dy = p1.y - p2.y
        return sqrt(dx * dx + dy * dy)
    }

    private fun quadHash(quad: MatOfPoint): Long {
        val points = quad.toArray()
        var hash: Long = 1
        for (p in points) {
            hash = 31 * hash + p.x.toInt()
            hash = 31 * hash + p.y.toInt()
        }
        return hash
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

    fun processPickedDocument(context: Context, uri: Uri, onScanComplete: () -> Unit) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                // Clear stale filtered bitmap from camera scanner
                _filteredBitmap.value?.recycle()
                _filteredBitmap.value = null

                val sourceBitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    ImageDecoder.decodeBitmap(
                        ImageDecoder.createSource(
                            context.contentResolver,
                            uri
                        )
                    ) { decoder, _, _ ->
                        decoder.isMutableRequired = true
                    }
                } else {
                    MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
                }

                // Capture original before processing
                _originalBitmap.value?.recycle()
                _originalBitmap.value = sourceBitmap.copy(Bitmap.Config.ARGB_8888, false)

                val mat = Mat()
                Utils.bitmapToMat(sourceBitmap, mat)

                // Picked documents usually have 0 rotation
                val rotation = 0

                // Clone mat for detectDocument since it modifies the input in-place.
                // The original mat must stay intact for warpDocumentHighQuality.
                val matForDetection = mat.clone()
                val result = detectDocument(matForDetection, rotation)

                if (result.isNotEmpty()) {
                    val bestQuad = result.first()
                    val warped = warpDocumentHighQuality(mat, bestQuad, rotation)
                    _resultBitmap.value = warped ?: mat.enhanceDocument().toBitmap()
                } else {
                    _resultBitmap.value = mat.enhanceDocument().toBitmap()
                }

                matForDetection.release()
                mat.release()
                sourceBitmap.recycle()

                // Notify UI that scan is complete — navigation is handled by callback
                onScanComplete()

            } catch (e: Exception) {
                Log.e("CameraViewModel", "Error processing picked document", e)
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        cameraExecutor.shutdown()
        cameraExecutor.awaitTermination(5, TimeUnit.SECONDS)
        clearBitmaps()
        lastWarpedQuadHash = 0
    }

    private fun clearBitmaps() {
        _filteredBitmap.value?.recycle()
        _filteredBitmap.value = null
        _originalBitmap.value?.recycle()
        _originalBitmap.value = null
        _resultBitmap.value?.recycle()
        _resultBitmap.value = null
        lastWarpedBitmap?.recycle()
        lastWarpedBitmap = null
    }

}
