package io.github.iostreamchik.scanner.detector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import io.github.iostreamchik.scanner.opencv.IMatBundle
import io.github.iostreamchik.scanner.opencv.PipelineParams
import io.github.iostreamchik.scanner.scoreContourWithParams
import io.github.iostreamchik.scanner.sortQuadPoints
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.collections.copy
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class OnnxDocumentDetector(
    private val context: Context,
    private val matBundle: IMatBundle,
    private val modelPath: String = "onnx/deeplabv3_mbv3_docseg.onnx",
    private val useCustomNormalization: Boolean = true
) : IDocumentDetector {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()

    var maskThreshold: Float = 0.5f
        set(value) {
            field = value.coerceIn(0.1f, 0.7f)
            _detectionParams.value = _detectionParams.value.copy(
                maskThreshold = "%.2f".format(field)
            )
        }

    private val _detectionParams = MutableStateFlow(DetectionParameters())
    override val detectionParams = _detectionParams.asStateFlow()

    private var session: OrtSession? = null
    private var inputName: String? = null
    private var isDeepLabV3 = false
    private var isInit = false

    // Persist the ONNX mask between preprocess() and detectQuad() so we don't
    // depend on matBundle.getMorph() surviving the intermediate bitmap captures
    // in CameraViewModel.runDetection().
    private var cachedMask: Mat? = null

    private val inputBuffer = FloatArray(INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS)
    private val outputBuffer = FloatArray(INPUT_SIZE * INPUT_SIZE * OUTPUT_CHANNELS)

    private fun initModel() {
        isDeepLabV3 = modelPath.contains("deeplabv3")
        try {
            sessionOptions.setIntraOpNumThreads(1)
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)

            val inputStream: InputStream = context.assets.open(modelPath)
            val modelBytes = inputStream.readBytes()
            inputStream.close()

            session = env.createSession(modelBytes, sessionOptions)
            inputName = session!!.inputNames.iterator().next()
            Log.d(TAG, "Model loaded: $modelPath, input=$inputName, size=${INPUT_SIZE}x$INPUT_SIZE, type=${if (isDeepLabV3) "DeepLabV3" else "BiRefNet"}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load ONNX model", e)
        } finally {
            sessionOptions.close()
        }
    }

    override fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat {
        if (isInit.not()) initModel()
        isInit = true
        // Clear stale mask from previous frame — error paths below return early,
        // so detectQuad() must not use a cached mask from a prior successful frame.
        cachedMask?.release()
        cachedMask = null
        val sess = session ?: return matBundle.getMorph()

        Log.d(TAG, "preprocess: rawMat=${rawMat.rows()}x${rawMat.cols()}, type=${rawMat.type()}, channels=${rawMat.channels()}")

        // Resize maintaining aspect ratio, then pad to INPUT_SIZE x INPUT_SIZE
        val scale = INPUT_SIZE.toDouble() / maxOf(rawMat.cols(), rawMat.rows())
        val resizedW = (rawMat.cols() * scale).toInt()
        val resizedH = (rawMat.rows() * scale).toInt()

        val resized = Mat()
        Imgproc.resize(rawMat, resized, Size(resizedW.toDouble(), resizedH.toDouble()))

        val rgb = Mat()
        Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_RGBA2RGB)
        resized.release()

        // Pad RGB to square with neutral gray borders
        val rgbPadded = Mat(INPUT_SIZE, INPUT_SIZE, CvType.CV_8UC3)
        rgbPadded.setTo(Scalar(128.0, 128.0, 128.0))
        val roi = rgbPadded.submat(0, rgb.rows(), 0, rgb.cols())
        rgb.copyTo(roi)
        roi.release()
        rgb.release()

        Log.d(TAG, "preprocess: padded to ${INPUT_SIZE}x${INPUT_SIZE} from ${resizedW}x${resizedH}")

        // Fill input buffer in NCHW layout
        if (isDeepLabV3) {
            for (c in 0 until INPUT_CHANNELS) {
                val channelOffset = c * INPUT_SIZE * INPUT_SIZE
                val meanVal = if (useCustomNormalization) CUSTOM_MEAN[c] else 0f
                val stdVal = if (useCustomNormalization) CUSTOM_STD[c] else 1f
                for (y in 0 until INPUT_SIZE) {
                    for (x in 0 until INPUT_SIZE) {
                        val pixel = rgbPadded.get(y, x)
                        inputBuffer[channelOffset + y * INPUT_SIZE + x] =
                            (((pixel[c] / 255.0) - meanVal) / stdVal).toFloat()
                    }
                }
            }
        } else {
            for (c in 0 until INPUT_CHANNELS) {
                val channelOffset = c * INPUT_SIZE * INPUT_SIZE
                for (y in 0 until INPUT_SIZE) {
                    for (x in 0 until INPUT_SIZE) {
                        val pixel = rgbPadded.get(y, x)
                        inputBuffer[channelOffset + y * INPUT_SIZE + x] =
                            (((pixel[c] / 255.0) - IMAGE_MEAN[c]) / IMAGE_STD[c]).toFloat()
                    }
                }
            }
        }

        val directBuffer = ByteBuffer.allocateDirect(inputBuffer.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        directBuffer.put(inputBuffer)
        directBuffer.flip()

        val inputTensor = OnnxTensor.createTensor(
            env,
            directBuffer,
            longArrayOf(1, INPUT_CHANNELS.toLong(), INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )

        Log.d(TAG, "preprocess: running model inference (input=${inputName})")
        val output: OrtSession.Result = try {
            sess.run(mapOf(inputName to inputTensor))
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            inputTensor.close()
            rgbPadded.release()
            return matBundle.getMorph()
        } finally {
            inputTensor.close()
        }
        Log.d(TAG, "preprocess: inference complete")

        output.use {
            val outputValue = output.get(0).value
            if (outputValue is OnnxTensor) {
                val floatBuf = outputValue.floatBuffer
                floatBuf.rewind()
                floatBuf.get(outputBuffer)
            } else if (outputValue is Array<out Any?>) {
                @Suppress("UNCHECKED_CAST")
                val arr = outputValue as Array<Array<Array<FloatArray>>>
                var flatIdx = 0
                for (c in arr) {
                    for (z in c) {
                        for (row in z) {
                            for (v in row) {
                                if (flatIdx < outputBuffer.size) {
                                    outputBuffer[flatIdx++] = v
                                }
                            }
                        }
                    }
                }
            } else {
                Log.e(TAG, "Unexpected output type: ${outputValue.javaClass.name}")
                rgbPadded.release()
                return matBundle.getMorph()
            }

            var outMin = Float.MAX_VALUE; var outMax = Float.MIN_VALUE
            var outMean = 0f
            for (i in outputBuffer.indices) {
                outMean += outputBuffer[i]
                if (outputBuffer[i] < outMin) outMin = outputBuffer[i]
                if (outputBuffer[i] > outMax) outMax = outputBuffer[i]
            }
            outMean /= outputBuffer.size
            Log.d(TAG, "preprocess: output tensor stats: min=${"%.4f".format(outMin)}, max=${"%.4f".format(outMax)}, mean=${"%.4f".format(outMean)}, shape=[1,${OUTPUT_CHANNELS},${INPUT_SIZE},${INPUT_SIZE}]")

            var below0 = 0; var below05 = 0; var above05 = 0; var above08 = 0
            for (v in outputBuffer) {
                if (v < 0f) below0++
                if (v < 0.5f) below05++
                if (v > 0.5f) above05++
                if (v > 0.8f) above08++
            }
            Log.d(TAG, "preprocess: output distribution: <0=$below0, <0.5=$below05, >0.5=$above05, >0.8=$above08, total=${outputBuffer.size}")
        }

        val probMap = Mat(INPUT_SIZE, INPUT_SIZE, CvType.CV_32FC1)

        if (isDeepLabV3) {
            for (y in 0 until INPUT_SIZE) {
                for (x in 0 until INPUT_SIZE) {
                    val idx = y * INPUT_SIZE + x
                    val bgLogit = outputBuffer[idx]
                    val fgLogit = outputBuffer[INPUT_SIZE * INPUT_SIZE + idx]
                    val documentProb = 1f / (1f + exp(-(fgLogit - bgLogit)))
                    probMap.put(y, x, documentProb.toDouble())
                }
            }
        } else {
            val totalPixels = INPUT_SIZE * INPUT_SIZE
            for (i in 0 until totalPixels) {
                probMap.put(i / INPUT_SIZE, i % INPUT_SIZE, outputBuffer[i].toDouble())
            }
        }

        // Zero out padding region — the model was trained on real content, not gray borders.
        // Leaving padding untouched lets the model hallucinate edges at the borders.
        val padTop = 0
        val padBottom = INPUT_SIZE - resizedH
        val padLeft = 0
        val padRight = INPUT_SIZE - resizedW
        if (padBottom > 0) probMap.submat(resizedH, INPUT_SIZE, 0, INPUT_SIZE).setTo(Scalar(0.0))
        if (padRight > 0) probMap.submat(0, resizedH, resizedW, INPUT_SIZE).setTo(Scalar(0.0))

        Imgproc.GaussianBlur(probMap, probMap, Size(5.0, 5.0), 1.5)

        // Convert to 8-bit [0,255] before threshold — prevents CV_32FC1 leak
        // through Imgproc.threshold which would break connectedComponentsWithStats.
        val prob8 = Mat(INPUT_SIZE, INPUT_SIZE, CvType.CV_8UC1)
        probMap.convertTo(prob8, CvType.CV_8UC1, 255.0)
        probMap.release()

        // Adaptive threshold: try Otsu first on the content region only (no padding).
        // Otsu finds the natural bimodal split between document and background pixels.
        // Falls back to manual threshold if Otsu produces nothing (flat histogram).
        val contentProb = prob8.submat(padTop, resizedH, padLeft, resizedW)
        val otsuMask = Mat(resizedH, resizedW, CvType.CV_8UC1)
        val otsuVal = Imgproc.threshold(contentProb, otsuMask, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
        Log.d(TAG, "preprocess: Otsu threshold=${"%.1f".format(otsuVal)} (manual=${"%.2f".format(maskThreshold * 255.0)})")

        val fullMask = if (Core.countNonZero(otsuMask) > INPUT_SIZE * INPUT_SIZE * 0.0005) {
            // Otsu found a valid split — compose full-size mask with padding zeroed
            val m = Mat.zeros(INPUT_SIZE, INPUT_SIZE, CvType.CV_8UC1)
            otsuMask.copyTo(m.submat(padTop, resizedH, padLeft, resizedW))
            otsuMask.release()
            m
        } else {
            // Fallback: manual threshold on full image
            otsuMask.release()
            val m = Mat(INPUT_SIZE, INPUT_SIZE, CvType.CV_8UC1)
            Imgproc.threshold(prob8, m, maskThreshold * 255.0, 255.0, Imgproc.THRESH_BINARY)
            m
        }
        prob8.release()

        var maskNonzero = Core.countNonZero(fullMask)
        Log.d(TAG, "preprocess: fullMask nonzero=$maskNonzero / ${fullMask.total()} (${"%.1f".format(100.0 * maskNonzero / fullMask.total())}%)")

        // Estimate document size from mask to scale morphological kernels proportionally.
        // Fixed 3x3/5x5 kernels are ineffective on 384x384 — they don't bridge real gaps.
        val tmpLabels = Mat(); val tmpStats = Mat(); val tmpCentroids = Mat()
        val tmpNLabels = Imgproc.connectedComponentsWithStats(fullMask, tmpLabels, tmpStats, tmpCentroids, 8, CvType.CV_32S)
        var docArea = 0
        for (l in 1 until tmpNLabels) {
            val a = tmpStats.get(l, Imgproc.CC_STAT_AREA)[0].toInt()
            if (a > docArea) docArea = a
        }
        tmpLabels.release(); tmpStats.release(); tmpCentroids.release()

        // Kernel size scales with largest blob: sqrt(area) gives a linear measure,
        // clamped so kernels are 5–21 (always odd). Small fragments get light cleanup;
        // large document masks get strong gap-bridging closes.
        val docLinear = sqrt(docArea.toDouble()).coerceIn(10.0, 200.0)
        val kernelCloseK = ((docLinear / 6.0).toInt().coerceIn(5, 21)).takeIf { it % 2 == 1 } ?: ((docLinear / 6.0).toInt().coerceIn(5, 21) - 1)
        val kernelOpenK = ((docLinear / 12.0).toInt().coerceIn(3, 9)).takeIf { it % 2 == 1 } ?: ((docLinear / 12.0).toInt().coerceIn(3, 9) - 1)

        val kernelClose = Mat(kernelCloseK, kernelCloseK, CvType.CV_8UC1, Scalar.all(1.0))
        val kernelOpen = Mat(kernelOpenK, kernelOpenK, CvType.CV_8UC1, Scalar.all(1.0))
        Log.d(TAG, "preprocess: morph kernels close=${kernelCloseK}x${kernelCloseK}, open=${kernelOpenK}x${kernelOpenK}, docArea=$docArea")

        Imgproc.morphologyEx(fullMask, fullMask, Imgproc.MORPH_CLOSE, kernelClose)
        Imgproc.morphologyEx(fullMask, fullMask, Imgproc.MORPH_OPEN, kernelOpen)
        Imgproc.morphologyEx(fullMask, fullMask, Imgproc.MORPH_CLOSE, kernelOpen)
        kernelClose.release()
        kernelOpen.release()

        // Connected component filtering — keep only the largest blob (the document).
        // Noise, page numbers, watermarks, and table fragments are always smaller.
        Log.d(TAG, "preprocess: fullMask before CCW: rows=${fullMask.rows()}, cols=${fullMask.cols()}, type=${fullMask.type()}, depth=${fullMask.depth()}, channels=${fullMask.channels()}, empty=${fullMask.empty()}")
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val nLabels = Imgproc.connectedComponentsWithStats(fullMask, labels, stats, centroids, 8, CvType.CV_32S)

        // Find largest component (skip label 0 = background)
        var largestLabel = 0
        var largestArea = 0
        for (label in 1 until nLabels) {
            val area = stats.get(label, Imgproc.CC_STAT_AREA)[0].toInt()
            if (area > largestArea) {
                largestArea = area
                largestLabel = label
            }
        }

        val minBlobArea = (INPUT_SIZE * INPUT_SIZE * 0.0005).toInt()
        val cleanedMask = Mat.zeros(fullMask.size(), CvType.CV_8UC1)
        if (largestArea >= minBlobArea) {
            Log.d(TAG, "preprocess: largest CC label=$largestLabel area=$largestArea — keeping")
            val compMask = Mat()
            Core.compare(labels, Scalar(largestLabel.toDouble()), compMask, Core.CMP_EQ)
            compMask.copyTo(cleanedMask)
            compMask.release()
        } else {
            Log.d(TAG, "preprocess: largest CC area=$largestArea below min=$minBlobArea — discarding all")
        }
        labels.release()
        stats.release()
        centroids.release()
        fullMask.release()

        // Crop to unpadded region
        val croppedMask = cleanedMask.submat(0, resizedH, 0, resizedW).clone()
        cleanedMask.release()
        rgbPadded.release()

        val morph = matBundle.getMorph()
        // Use INTER_NEAREST for binary mask — preserves sharp edges, no interpolation artifacts
        Imgproc.resize(croppedMask, morph,
            Size(scaledWidth.toDouble(), scaledHeight.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
        croppedMask.release()

        // Cache the mask for detectQuad() — matBundle.getMorph() may be corrupted
        // by intermediate bitmap captures in runDetection() before detectQuad() runs.
        cachedMask = morph.clone()

        val nonzeroCount = Core.countNonZero(morph)
        val morphTotal = morph.total()
        Log.d(TAG, "preprocess: resized mask nonzero=$nonzeroCount / $morphTotal (${"%.1f".format(100.0 * nonzeroCount / morphTotal)}%) -> ${scaledWidth}x${scaledHeight}")

        _detectionParams.value = _detectionParams.value.copy(
            brightness = "ONNX",
            cannyHigh = "N/A",
            cannyLow = "N/A"
        )

        Log.d(TAG, "preprocess done: mask nonzero=$nonzeroCount / $morphTotal, scaled to ${scaledWidth}x${scaledHeight}")
        return morph
    }

    override fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        params: PipelineParams
    ): MatOfPoint? {
        // Use cached ONNX mask if available — the matBundle.getMorph() passed as
        // morphImage is often empty because intermediate bitmap captures in
        // runDetection() corrupt/clear the shared pooled mat.
        val useCached = cachedMask != null && !cachedMask!!.empty()
        val workingMask = if (useCached) cachedMask!! else morphImage

        Log.d(TAG, "=== detectQuad START: morph=${workingMask.rows()}x${workingMask.cols()}, scaled=${scaledWidth}x${scaledHeight}, source=${if (useCached) "cached" else "param"} ===")

        Log.d(TAG, "  workingMask: type=${workingMask.type()}, channels=${workingMask.channels()}, total=${workingMask.total()}, depth=${workingMask.depth()}")
        val morphNonZero = Core.countNonZero(workingMask)
        Log.d(TAG, "  workingMask nonzero pixels: $morphNonZero / ${workingMask.total()}")

        var hasNonZero = false
        val sampleN = min(5000, workingMask.total().toInt())
        for (i in 0 until sampleN) {
            val r = i / workingMask.cols()
            val c = i % workingMask.cols()
            if (r < workingMask.rows() && c < workingMask.cols()) {
                val v = workingMask.get(r, c)
                if (v != null && v[0] > 0.0) {
                    hasNonZero = true
                    break
                }
            }
        }
        Log.d(TAG, "  workingMask hasNonZero (sampled): $hasNonZero")

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = matBundle.getHierarchy()
        Log.d(TAG, "  findContours: calling with RETR_LIST + CHAIN_APPROX_SIMPLE, hierarchy=${hierarchy.rows()}x${hierarchy.cols()}, type=${hierarchy.type()}")
        Imgproc.findContours(
            workingMask,
            contours,
            hierarchy,
            Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        Log.d(TAG, "  findContours: found ${contours.size} contours")

        if (contours.isEmpty() && workingMask.total() > 0) {
            Log.d(TAG, "  [DEBUG] workingMask pixel samples (top-left 10x10):")
            for (r in 0 until min(10, workingMask.rows())) {
                val rowStr = mutableListOf<String>()
                for (c in 0 until min(10, workingMask.cols())) {
                    val v = workingMask.get(r, c)
                    rowStr.add(if (v != null) "${"%.0f".format(v[0])}" else "?")
                }
                Log.d(TAG, "    row $r: ${rowStr.joinToString(" ")}")
            }
        }

        val frameArea = scaledWidth * scaledHeight
        val minArea = frameArea * params.minAreaFraction
        Log.d(TAG, "  frameArea=$frameArea, minArea=$minArea, minAreaFraction=${params.minAreaFraction}")

        val areaStats = contours.map { abs(Geometry.contourArea(it)) }.sortedDescending().take(5)
        Log.d(TAG, "  Top contour areas: ${areaStats.joinToString(", ") { "%.0f".format(it) }}")

        val candidates = mutableListOf<MatOfPoint>()
        val approx = matBundle.getApprox()
        val hullPoints = matBundle.getHullPoints()
        hullPoints.release()
        hullPoints.create(0, 1, CvType.CV_32FC2)
        var skippedArea = 0
        var skippedPoints = 0
        var skippedNot4 = 0
        var skippedNotRect = 0
        var skippedSolidity = 0

        for (contour in contours) {
            val area = abs(Geometry.contourArea(contour))
            if (area < minArea) {
                skippedArea++
                continue
            }

            if (contour.total() < 10) {
                skippedPoints++
                continue
            }

            val hull = matBundle.getHull()
            Geometry.convexHull(contour, hull)
            val hullCount = hull.rows() * hull.cols()
            val hullData = IntArray(hullCount * hull.channels())
            hull.get(0, 0, hullData)
            val hullPointList = mutableListOf<Point>()
            for (i in hullData.indices step 2) {
                if (i + 1 < hullData.size) {
                    hullPointList.add(Point(hullData[i].toDouble(), hullData[i + 1].toDouble()))
                }
            }
            hullPoints.fromList(hullPointList)

            val hullPtCount = hullPointList.size
            val peri = Geometry.arcLength(hullPoints, true)
            var foundQuad = false
            val epsilons = listOf(0.015, 0.025, 0.04, 0.06, 0.10)

            for (tol in epsilons) {
                val epsilon = tol * peri
                Geometry.approxPolyDP(hullPoints, approx, epsilon, true)
                if (approx.total() == 4L && isRectangle(approx)) {
                    Log.d(TAG, "    approxPolyDP: tol=$tol, peri=${"%.0f".format(peri)}, hullPts=$hullPtCount -> QUAD at epsilon=${"%.2f".format(epsilon)}")
                    foundQuad = true
                    break
                }
            }

            if (!foundQuad) {
                // approxPolyDP couldn't reduce the hull to 4 points — use diagonal
                // extremes of the contour (min/max x+y, min/max y-x) as corners.
                val pts = contour.toArray()
                var tlIdx = 0; var brIdx = 0; var trIdx = 0; var blIdx = 0
                var tlSum = Long.MAX_VALUE; var brSum = Long.MIN_VALUE
                var trDiff = Long.MAX_VALUE; var blDiff = Long.MIN_VALUE
                for (i in pts.indices) {
                    val x = pts[i].x.toLong(); val y = pts[i].y.toLong()
                    val sum = x + y; val diff = y - x
                    if (sum < tlSum) { tlSum = sum; tlIdx = i }
                    if (sum > brSum) { brSum = sum; brIdx = i }
                    if (diff < trDiff) { trDiff = diff; trIdx = i }
                    if (diff > blDiff) { blDiff = diff; blIdx = i }
                }
                val fallbackPts = arrayOf(pts[tlIdx], pts[trIdx], pts[brIdx], pts[blIdx])
                val fallbackApprox = matBundle.getApprox()
                fallbackApprox.fromArray(*fallbackPts)
                val rectCheck = Geometry.boundingRect(fallbackApprox)
                val rectAreaCheck = rectCheck.width * rectCheck.height
                val solidityCheck = area / rectAreaCheck.toDouble()
                Log.d(TAG, "    fallback diagonal extremes: solidity=${"%.2f".format(solidityCheck)}")
                if (solidityCheck >= 0.15) {
                    val scaleX = originalWidth.toDouble() / scaledWidth
                    val scaleY = originalHeight.toDouble() / scaledHeight
                    val scaledFallback = fallbackPts.map { Point(it.x * scaleX, it.y * scaleY) }
                    val quadFallback = MatOfPoint(*scaledFallback.toTypedArray())
                    candidates.add(quadFallback)
                    Log.d(TAG, "    fallback: ADDED quad (solidity=${"%.2f".format(solidityCheck)})")
                } else {
                    skippedSolidity++
                }
                continue
            }

            if (!isRectangle(approx)) {
                skippedNotRect++
                continue
            }

            val rect = Geometry.boundingRect(approx)
            val rectArea = rect.width * rect.height
            val solidity = area / rectArea.toDouble()
            if (solidity < 0.3) {
                skippedSolidity++
                continue
            }

            val scaleX = originalWidth.toDouble() / scaledWidth
            val scaleY = originalHeight.toDouble() / scaledHeight
            val scaledPoints = approx.toArray().map { Point(it.x * scaleX, it.y * scaleY) }
            val quad = MatOfPoint(*scaledPoints.toTypedArray())

            Log.d(TAG, "  CANDIDATE: area=$area, solidity=${"%.2f".format(solidity)}, rect=${rect.width}x${rect.height}")
            candidates.add(quad)
        }

        Log.d(TAG, "  detectQuad summary: contours=${contours.size}, candidates=${candidates.size}, skippedArea=$skippedArea, skippedPoints=$skippedPoints, skippedNot4=$skippedNot4, skippedNotRect=$skippedNotRect, skippedSolidity=$skippedSolidity")

        val best = candidates.maxByOrNull {
            scoreContourWithParams(
                it,
                originalWidth,
                originalHeight,
                params
            )
        }
        val result = best?.let { quad ->
            val sorted = sortQuadPoints(quad.toArray().toList())
            if (sorted.size == 4) MatOfPoint(*sorted.toTypedArray()) else quad
        }
        candidates.forEach { it.release() }
        Log.d(TAG, "  detectQuad END: result=${if (result != null) "found (${result.total()} pts)" else "null"}")
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

    fun release() {
        session?.close()
        session = null
        cachedMask?.release()
        cachedMask = null
        try {
            env.close()
        } catch (_: Exception) {
            // Ignore cleanup errors
        }
    }

    companion object {
        const val INPUT_SIZE = 384
        const val INPUT_CHANNELS = 3
        const val OUTPUT_CHANNELS = 2

        private val IMAGE_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGE_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        // LearnOpenCV custom normalization — matches training pipeline in
        // https://learnopencv.com/deep-learning-based-document-segmentation-using-semantic-segmentation-deeplabv3-on-custom-dataset/
        private val CUSTOM_MEAN = floatArrayOf(0.4611f, 0.4359f, 0.3905f)
        private val CUSTOM_STD = floatArrayOf(0.2193f, 0.2150f, 0.2109f)

        private const val TAG = "OnnxDetector"

        fun isRectangle(approx: MatOfPoint2f): Boolean {
            val pts = approx.toArray()
            val deviations = mutableListOf<Float>()
            var maxDeviation = 0.0
            for (i in 0..3) {
                val angle = computeAngle(
                    pts[(i + 1) % 4],
                    pts[(i + 3) % 4],
                    pts[i]
                )
                val deviation = abs(90.0 - angle)
                deviations.add(deviation.toFloat())
                maxDeviation = max(maxDeviation, deviation)
            }
            Log.d(TAG, "    isRectangle: angles=[${"%.1f".format(pts[0].x)},${"%.1f".format(pts[0].y)};${"%.1f".format(pts[1].x)},${"%.1f".format(pts[1].y)};${"%.1f".format(pts[2].x)},${"%.1f".format(pts[2].y)};${"%.1f".format(pts[3].x)},${"%.1f".format(pts[3].y)}], deviations=[${deviations.joinToString(", ") { "%.1f".format(it) }}]°, max=${"%.1f".format(maxDeviation)}° -> ${if (maxDeviation < 25) "PASS" else "FAIL"}")
            return maxDeviation < 25
        }

        fun computeAngle(p1: Point, p2: Point, center: Point): Double {
            val dx1 = p1.x - center.x
            val dy1 = p1.y - center.y
            val dx2 = p2.x - center.x
            val dy2 = p2.y - center.y
            val dot = dx1 * dx2 + dy1 * dy2
            val norm1 = sqrt(dx1 * dx1 + dy1 * dy1)
            val norm2 = sqrt(dx2 * dx2 + dy2 * dy2)
            return acos(dot / (norm1 * norm2)) * 180.0 / Math.PI
        }
    }
}