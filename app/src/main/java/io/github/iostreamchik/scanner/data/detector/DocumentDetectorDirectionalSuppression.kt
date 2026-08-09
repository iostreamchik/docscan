package io.github.iostreamchik.scanner.data.detector

import io.github.iostreamchik.scanner.entity.DetectionParameters
import io.github.iostreamchik.scanner.entity.IntermediateBitmaps
import io.github.iostreamchik.scanner.entity.PipelineParams
import io.github.iostreamchik.scanner.data.opencv.IMatBundle
import io.github.iostreamchik.scanner.data.opencv.OpenCVAdapter
import io.github.iostreamchik.scanner.data.opencv.PreprocessingConfig
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Size
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import kotlin.math.abs

class DocumentDetectorDirectionalSuppression(
    internal val matBundle: IMatBundle
) : IDocumentDetector {

    private val _detectionParams = MutableStateFlow(DetectionParameters())
    override val detectionParams = _detectionParams.asStateFlow()

    private val directionalConfig = PreprocessingConfig(
        dimBoostDivisor = 8.0,
        brightBoostDivisor = 100.0,
        brightnessFormat = "%.1f",
        claheFormat = "%.2f",
        cannyFormat = "%d"
    )

    override fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat {
        OpenCVAdapter.preprocessClassical(
            rawMat, scaledWidth, scaledHeight, params, directionalConfig, matBundle,
            { brightness, claheClipLimit, cannyHigh, cannyLow ->
                _detectionParams.value = _detectionParams.value.copy(
                    brightness = "%.1f".format(brightness),
                    claheClipLimit = "%.2f".format(claheClipLimit),
                    cannyHigh = cannyHigh.toInt().toString(),
                    cannyLow = cannyLow.toInt().toString()
                )
            }
        )

        var closeKsize = params.strongCloseSize.coerceIn(3, 5)
        if (closeKsize % 2 == 0) closeKsize++
        OpenCVAdapter.createRectKernel(
            Size(closeKsize.toDouble(), closeKsize.toDouble()),
            matBundle.getKernel2()
        )
        OpenCVAdapter.morphClose(matBundle.getEdges(), matBundle.getMorph(), matBundle.getKernel2())

        val dirKsize = params.directionalKernelSize.coerceIn(1, 5)
        OpenCVAdapter.createRectKernel(
            Size(dirKsize.toDouble(), 1.0),
            matBundle.getHorizontalKernel()
        )
        Imgproc.morphologyEx(
            matBundle.getMorph(),
            matBundle.getHorizontalClose(),
            Imgproc.MORPH_CLOSE,
            matBundle.getHorizontalKernel()
        )

        OpenCVAdapter.createRectKernel(
            Size(1.0, dirKsize.toDouble()),
            matBundle.getVerticalKernel()
        )
        Imgproc.morphologyEx(
            matBundle.getHorizontalClose(),
            matBundle.getVerticalClose(),
            Imgproc.MORPH_CLOSE,
            matBundle.getVerticalKernel()
        )

        matBundle.getVerticalClose().copyTo(matBundle.getMorph())

        return matBundle.getMorph()
    }

    override fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        rotation: Int,
        params: PipelineParams
    ): MatOfPoint? {
        return OpenCVAdapter.findBestQuad(
            morphImage, matBundle,
            scaledWidth, scaledHeight,
            originalWidth, originalHeight,
            params.minAreaFraction.toDouble(),
            approxEpsilon = 0.015,
            selector = { candidates ->
                candidates.maxByOrNull { abs(Geometry.contourArea(it)) }
            }
        )
    }

    override fun captureIntermediateSnapshots(rotation: Int): IntermediateBitmaps =
        matBundle.captureClassicalSnapshots(rotation)

    override fun release() {
        matBundle.releaseAll()
    }
}