package io.github.iostreamchik.scanner.camera

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
import io.github.iostreamchik.scanner.drawQuadOverlay
import io.github.iostreamchik.scanner.enhanceDocument
import io.github.iostreamchik.scanner.fixRotation
import io.github.iostreamchik.scanner.opencv.CannyThresholdCalculator
import io.github.iostreamchik.scanner.opencv.ICannyThresholdCalculator
import io.github.iostreamchik.scanner.opencv.IMatBundle
import io.github.iostreamchik.scanner.opencv.MatBundle
import io.github.iostreamchik.scanner.opencv.PipelineParams
import io.github.iostreamchik.scanner.sharpen
import io.github.iostreamchik.scanner.toBitmap
import io.github.iostreamchik.scanner.toMatRGBA
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
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

class CameraViewModel(
    private val matBundle: IMatBundle = MatBundle(),
    private val thresholdCalculator: ICannyThresholdCalculator = CannyThresholdCalculator(matBundle)
) : ViewModel() {

    val cameraExecutor = Executors.newSingleThreadExecutor()

    private val quadHistory = ArrayDeque<MatOfPoint>()
    private var lastFrameSize: Size? = null
    private val MAX_HISTORY = 10
    private var frameCounter = 0
    private val STABILITY_CHECK_INTERVAL = 3

    private val PROCESS_WIDTH = 640.0

    // Debug: manual canny high override (0f = auto)
    private val _manualCannyHigh = MutableStateFlow<Float>(0f)
    val manualCannyHigh: StateFlow<Float> = _manualCannyHigh.asStateFlow()

    fun setManualCannyHigh(value: Float) {
        _manualCannyHigh.value = value
    }

    // Cache: skip re-warping when the same quad is detected repeatedly
    private var lastWarpedQuadHash: Long = 0
    private var lastWarpedBitmap: Bitmap? = null

    private val _filteredBitmap = MutableStateFlow<Bitmap?>(null)
    val filteredBitmap = _filteredBitmap.asStateFlow()

    private val _originalBitmap = MutableStateFlow<Bitmap?>(null)
    val originalBitmap = _originalBitmap.asStateFlow()

    private val _resultBitmap = MutableStateFlow<Bitmap?>(null)
    val resultBitmap = _resultBitmap.asStateFlow()

    private val _quadOverlayBitmap = MutableStateFlow<Bitmap?>(null)
    val quadOverlayBitmap = _quadOverlayBitmap.asStateFlow()

    private val _exposureStateFlow = MutableStateFlow("")
    val exposureStateFlow = _exposureStateFlow.asStateFlow()

    private val _errorState = MutableStateFlow<String?>(null)
    val errorState = _errorState.asStateFlow()

    private var lastUiUpdateTime = 0L
    private val UI_UPDATE_THROTTLE_MS = 100L

    // Store last picked URI for reprocessing
    private var lastPickedUri: Uri? = null

    fun setError(message: String?) {
        _errorState.value = message
    }

    // Pipeline parameters — reads from pipelineConfigurationManager
    private val _currentParams = MutableStateFlow(PipelineParams.Default)
    val currentParams: StateFlow<PipelineParams> = _currentParams.asStateFlow()

    /**
     * Update pipeline parameters — called from FileScanResultScreen when parameters change.
     */
    fun updateParams(newParams: PipelineParams) {
        _currentParams.value = newParams
    }

    /**
     * Reset all pipeline parameters to defaults.
     */
    fun resetToDefaultParams() {
        _currentParams.value = PipelineParams.Default
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
                        warped?.copy(Bitmap.Config.ARGB_8888, false)
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

            // 4️⃣ Automatic Canny thresholds via Otsu + EMA
            // Manual override: 0f = auto, any other value = manual threshold
            val (cannyHigh, cannyLow) = if (_manualCannyHigh.value == 0f) {
                thresholdCalculator.computeThreshold(matBundle.getGray())
            } else {
                val h = _manualCannyHigh.value.toDouble()
                Pair(h, h * 0.5)
            }
            val exposureTime = System.currentTimeMillis()
            if (exposureTime - lastUiUpdateTime >= UI_UPDATE_THROTTLE_MS) {
                val modeSuffix = if (_manualCannyHigh.value != 0f) " (MAN)" else ""
                _exposureStateFlow.value = "br: ${avgBrightness.toInt()} ct: ${contrast.toInt()}"
                lastUiUpdateTime = exposureTime
            }
            Log.d("Pipeline", "Auto Canny: High=$cannyHigh, Low=$cannyLow intensity=$avgBrightness")
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
                val hullIndices = IntArray(hull.rows().toInt())
                hull.get(0, 0, hullIndices)
                val hullPointList = hullIndices.map { contourArray[it] }
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

            if (best == null) return emptyList()

            // Validate document size — a quad filling the entire frame is likely a false positive
            val bestRect = Imgproc.boundingRect(best)
            val bestArea = bestRect.width * bestRect.height
            val frameOriginalArea = originalWidth * originalHeight
            if (bestArea > frameOriginalArea * 0.95) {
                // Detected "document" equals the full image — treat as no detection
                best.release()
                return emptyList()
            }

            return listOf(best)

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

        return maxDeviation < 15
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
                lastPickedUri = uri
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

                // Use 0 rotation for picked images (no camera rotation)
                val rotation = 0

                // Clone mat for detectDocument since it modifies the input in-place.
                // The original mat must stay intact for warpDocumentHighQuality.
                val matForDetection = mat.clone()
                val params = _currentParams.value
                val result = detectDocumentWithParams(matForDetection, rotation, params)

                if (result.isNotEmpty()) {
                    // Use the best quad directly (no fusion needed for single image)
                    // Unlike live camera, we don't have multiple frames to average
                    val bestQuad = result.first()
                    
                    // Warp using the detected quad
                    val warped = warpDocumentHighQuality(mat, bestQuad, rotation)
                    // Clone before emitting to StateFlow so Compose gets its own
                    // independent copy that won't be affected by recycling in
                    // onCleared() or a subsequent processPickedDocument call.
                    _resultBitmap.value = (warped ?: mat.enhanceDocument().toBitmap())
                        .copy(Bitmap.Config.ARGB_8888, false)

                    // Create quad overlay bitmap for FileScanResultScreen
                    val originalMat = Mat()
                    Utils.bitmapToMat(_originalBitmap.value ?: sourceBitmap, originalMat)
                    val quadPoints = bestQuad.toList().map { Point(it.x.toDouble(), it.y.toDouble()) }
                    _quadOverlayBitmap.value = originalMat.drawQuadOverlay(quadPoints, thickness = 4)
                    originalMat.release()
                } else {
                    // No document detected — show enhanced original
                    // Clone before emitting to StateFlow.
                    _resultBitmap.value = mat.enhanceDocument().toBitmap()
                        .copy(Bitmap.Config.ARGB_8888, false)
                    _quadOverlayBitmap.value = null
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

    /**
     * Reprocess the last picked document with current pipeline parameters.
     * Called from FileScanResultScreen when parameters change.
     */
    fun reprocessPickedDocument(context: Context) {
        val uri = lastPickedUri ?: return
        processPickedDocument(context, uri) {}
    }

    /**
     * Compute auto Canny thresholds and reprocess the picked document with them.
     * Updates currentParams flow so UI picks up new values automatically.
     */
    suspend fun enableCannyAuto(context: Context) {
        val uri = lastPickedUri ?: return
        withContext(Dispatchers.IO) {
            try {
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

                val mat = Mat()
                Utils.bitmapToMat(sourceBitmap, mat)

                val originalWidth = mat.cols()
                val originalHeight = mat.rows()
                val maxDimension = max(originalWidth, originalHeight)
                val scale = PROCESS_WIDTH / maxDimension
                val scaledWidth = (originalWidth * scale)
                val scaledHeight = (originalHeight * scale)

                val smallMat = Mat()
                Imgproc.resize(mat, smallMat, Size(scaledWidth, scaledHeight))
                Imgproc.cvtColor(smallMat, matBundle.getGray(), Imgproc.COLOR_RGBA2GRAY)
                smallMat.release()
                mat.release()
                sourceBitmap.recycle()

                val (cannyHigh, cannyLow) = thresholdCalculator.computeThreshold(matBundle.getGray())
                updateParams(_currentParams.value.copy(
                    cannyLow = cannyLow.toFloat(),
                    cannyHigh = cannyHigh.toFloat(),
                    cannyAutoDetect = true
                ))
                  // Reprocess with newly computed auto-detect thresholds
                 processPickedDocument(context, uri) {}
            } catch (e: Exception) {
                Log.e("CameraViewModel", "Error in enableCannyAuto: ${e.message}")
            }
        }
    }

    /**
     * Disable auto Canny detection — set the flag to false so camera pipeline
     * uses the manual thresholds instead.
     */
    fun disableCannyAuto() {
        _currentParams.value = _currentParams.value.copy(
            cannyAutoDetect = false
        )
    }

    /**
     * Detect document using configurable pipeline parameters.
     * Similar to detectDocument() but uses params for all adjustable values.
     */
    private fun detectDocumentWithParams(mat: Mat, rotation: Int, params: PipelineParams): List<MatOfPoint> {
        val originalWidth = mat.cols()
        val originalHeight = mat.rows()
        val maxDimension = max(originalWidth, originalHeight)
        val scale = PROCESS_WIDTH / maxDimension
        val scaledWidth = (originalWidth * scale)
        val scaledHeight = (originalHeight * scale)

        Log.d("CameraViewModel", "=== detectDocumentWithParams START ===")
        Log.d("CameraViewModel", "Input: ${originalWidth}x${originalHeight}, maxDim=$maxDimension, scale=${"%.4f".format(scale)}, scaled=${scaledWidth.toInt()}x${scaledHeight.toInt()}")
        Log.d("CameraViewModel", "Params: medianBlur=${params.medianBlurKsize}, claheClip=${params.claheClipLimit}, claheTile=${params.claheTileSize}, morphClose=${params.morphCloseSize}, cannyLow=${params.cannyLow}, cannyHigh=${params.cannyHigh}, strongClose=${params.strongCloseSize}, dirKernel=${params.directionalKernelSize}, approxTol=${params.approxPolyDPTolerance}, minAreaFrac=${params.minAreaFraction}")

        try {
            val smallMat = Mat()
            Imgproc.resize(mat, smallMat, Size(scaledWidth, scaledHeight))

            // 1️⃣ Grayscale
            Imgproc.cvtColor(smallMat, matBundle.getGray(), Imgproc.COLOR_RGBA2GRAY)
            smallMat.release()

            // 2️⃣ Median Blur (configurable kernel size)
            val blurKsize = params.medianBlurKsize.coerceAtLeast(3)
            Imgproc.medianBlur(matBundle.getGray(), matBundle.getBlurred(), blurKsize)
            Log.d("CameraViewModel", "Step2 MedianBlur: ksize=$blurKsize")

            // 3️⃣ CLAHE (configurable clip limit and tile size)
            val clahe = Imgproc.createCLAHE(params.claheClipLimit.toDouble(), Size(params.claheTileSize.toDouble(), params.claheTileSize.toDouble()))
            clahe.apply(matBundle.getBlurred(), matBundle.getEnhanced())
            Log.d("CameraViewModel", "Step3 CLAHE: clipLimit=${params.claheClipLimit}, tileSize=${params.claheTileSize}")

            // 4️⃣ Morph Close (configurable kernel size, gated by contrast)
            // For low-contrast images, morph close washes out faint document edges.
            // Skip it when the enhanced image std dev is below threshold.
            Core.meanStdDev(matBundle.getEnhanced(), matBundle.getMean(), matBundle.getStd())
            val enhancedContrast = matBundle.getStd().toArray()[0]
            val skipMorphClose = enhancedContrast < 25.0

            if (skipMorphClose) {
                matBundle.getEnhanced().copyTo(matBundle.getMorph())
                Log.d("CameraViewModel", "Step4 MorphClose: SKIPPED (contrast=$enhancedContrast < 25)")
            } else {
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_RECT,
                    Size(params.morphCloseSize.toDouble(), params.morphCloseSize.toDouble())
                ).also { kernel ->
                    matBundle.getKernel().release()
                    kernel.copyTo(matBundle.getKernel())
                }
                Imgproc.morphologyEx(matBundle.getEnhanced(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, matBundle.getKernel())
                Log.d("CameraViewModel", "Step4 MorphClose: ksize=${params.morphCloseSize} (contrast=$enhancedContrast)")
            }

            // 5️⃣ Canny thresholds (auto or manual)
            var (cannyHigh, cannyLow) = if (params.cannyAutoDetect) {
                thresholdCalculator.computeThreshold(matBundle.getGray())
            } else {
                Pair(params.cannyHigh.toDouble(), params.cannyLow.toDouble())
            }

            // Auto-fallback: if Otsu produced thresholds > 100, the document edges
            // are likely too faint for Canny at those levels. Scale down to a
            // range that captures weaker edges. Only applies in auto mode.
            if (params.cannyAutoDetect && cannyHigh > 100.0) {
                val fallbackHigh = 50.0
                val fallbackLow = fallbackHigh * 0.5
                Log.d("CameraViewModel", "Step5 Canny: Otsu=$cannyHigh/$cannyLow → FALLBACK to $fallbackLow/$fallbackHigh")
                cannyHigh = fallbackHigh
                cannyLow = fallbackLow
            }

            Imgproc.Canny(matBundle.getEnhanced(), matBundle.getEdges(), cannyLow, cannyHigh)
            Log.d("CameraViewModel", "Step5 Canny: low=$cannyLow, high=$cannyHigh (auto=${params.cannyAutoDetect})")

            // Push auto-computed thresholds back into params so the UI can display them
            if (params.cannyAutoDetect) {
                viewModelScope.launch(Dispatchers.Main) {
                    updateParams(_currentParams.value.copy(
                        cannyLow = cannyLow.toFloat(),
                        cannyHigh = cannyHigh.toFloat()
                    ))
                }
            }

            // 6️⃣ Strong Closing (configurable kernel size, scaled)
            var closeKsize = (params.strongCloseSize * scale).coerceAtLeast(3.0).toInt()
            if (closeKsize % 2 == 0) closeKsize++
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(closeKsize.toDouble(), closeKsize.toDouble())).also { kernel2 ->
                matBundle.getKernel2().release()
                kernel2.copyTo(matBundle.getKernel2())
            }
            Imgproc.morphologyEx(matBundle.getEdges(), matBundle.getMorph(), Imgproc.MORPH_CLOSE, matBundle.getKernel2())
            Log.d("CameraViewModel", "Step6 StrongClose: ksize=$closeKsize (params=${params.strongCloseSize}, scale=${"%.4f".format(scale)})")

            // 7️⃣ Directional Suppression (configurable kernel size, scaled)
            val dirKsize = (params.directionalKernelSize * scale).coerceAtLeast(3.0).toInt()
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
            Log.d("CameraViewModel", "Step7 DirSuppression: ksize=$dirKsize (params=${params.directionalKernelSize})")

            // Set filtered bitmap from morph result
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

            Log.d("CameraViewModel", "Step8 Contours: total=${contours.size}")

            if (contours.isEmpty()) {
                Log.d("CameraViewModel", "=== detectDocumentWithParams NO DETECTION: no contours ===")
                return emptyList()
            }

            val frameArea = scaledWidth * scaledHeight
            val minArea = frameArea * params.minAreaFraction
            Log.d("CameraViewModel", "Step8 MinArea: $minArea (frameArea=$frameArea * minAreaFrac=${params.minAreaFraction})")
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
                val hullIndices = IntArray(hull.rows().toInt())
                hull.get(0, 0, hullIndices)
                val hullPointList = hullIndices.map { contourArray[it] }
                matBundle.getHullPoints().fromList(hullPointList.map { Point(it.x, it.y) })

                val peri = Imgproc.arcLength(matBundle.getHullPoints(), true)
                Imgproc.approxPolyDP(
                    matBundle.getHullPoints(),
                    approx,
                    params.approxPolyDPTolerance * peri,
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

            Log.d("CameraViewModel", "Step8 Candidates: ${documentCandidates.size} quads passed filters")

            if (documentCandidates.isEmpty()) {
                Log.d("CameraViewModel", "=== detectDocumentWithParams NO DETECTION: no candidates ===")
                return emptyList()
            }

            val best = documentCandidates.maxByOrNull {
                val score = scoreContourWithParams(it, scaledWidth.toInt(), scaledHeight.toInt(), params)
                Log.d("CameraViewModel", "  Candidate score: area=${Imgproc.contourArea(it)}, score=$score")
                score
            }

            Log.d("CameraViewModel", "Step8 Best detected: ${best != null}")

            if (best == null) {
                Log.d("CameraViewModel", "=== detectDocumentWithParams NO DETECTION: no best ===")
                return emptyList()
            }

            // Validate document size — a quad filling the entire frame is likely a false positive
            val bestRect = Imgproc.boundingRect(best)
            val bestArea = bestRect.width * bestRect.height
            val frameOriginalArea = originalWidth * originalHeight
            Log.d("CameraViewModel", "Step8 SizeValidation: bestArea=$bestArea, frameArea=$frameOriginalArea, ratio=${"%.4f".format(bestArea.toDouble() / frameOriginalArea)}")
            if (bestArea > frameOriginalArea * 0.95) {
                Log.d("CameraViewModel", "Step8 SizeValidation: REJECTED (fills entire frame)")
                best.release()
                return emptyList()
            }

            Log.d("CameraViewModel", "=== detectDocumentWithParams SUCCESS: detected quad ===")
            return listOf(best)

        } catch (e: Exception) {
            Log.e("CameraViewModel", "Error in detectDocumentWithParams: ${e.message}")
            e.printStackTrace()
            return emptyList()
        } finally {
            matBundle.releaseAll()
        }
    }

    private fun scoreContourWithParams(
        contour: MatOfPoint,
        width: Int,
        height: Int,
        params: PipelineParams
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

        val frameArea = width * height.toDouble()
        val areaRatio = area / frameArea
        // Smooth interpolation: 1.0 at areaRatio <= 0.02, linearly down to 0.2 at areaRatio >= 0.5
        val areaRatioScore = when {
            areaRatio <= 0.02f -> 1.0
            areaRatio >= 0.5  -> 0.2
            else               -> 1.0 - ((areaRatio - 0.02) / 0.48) * 0.8
        }

        return area * params.scoreAreaWeight +
            centerScore * params.scoreCenterWeight * width * height +
            areaRatioScore * params.scoreAreaRatioWeight * width * height
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
