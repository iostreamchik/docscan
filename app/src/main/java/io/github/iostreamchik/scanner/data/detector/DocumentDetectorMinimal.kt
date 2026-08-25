package io.github.iostreamchik.scanner.data.detector

import io.github.iostreamchik.scanner.entity.DetectionParameters
import io.github.iostreamchik.scanner.entity.IntermediateBitmaps
import io.github.iostreamchik.scanner.entity.PipelineParams
import io.github.iostreamchik.scanner.data.opencv.IMatBundle
import io.github.iostreamchik.scanner.data.opencv.OpenCVAdapter
import io.github.iostreamchik.scanner.data.opencv.PreprocessingConfig
import io.github.iostreamchik.scanner.data.utils.scoreContourWithParams
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class DocumentDetectorMinimal(
    internal val matBundle: IMatBundle
) : IDocumentDetector {

    private val _detectionParams = MutableStateFlow(
        DetectionParameters(detectorName = AsyncDetectorSource.MINIMAL.detectionParamsName)
    )
    override val detectionParams: StateFlow<DetectionParameters> = _detectionParams.asStateFlow()

    override suspend fun preprocess(
        rawMat: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        params: PipelineParams
    ): Mat {
        OpenCVAdapter.preprocessClassical(
            rawMat, scaledWidth, scaledHeight, params, PreprocessingConfig(), matBundle,
            { brightness, claheClipLimit, cannyHigh, cannyLow ->
                _detectionParams.value = _detectionParams.value.copy(
                    claheClipLimit = String.format("%.1f", claheClipLimit),
                    cannyHigh = String.format("%.0f", cannyHigh),
                    cannyLow = String.format("%.0f", cannyLow),
                    brightness = String.format("%.0f", brightness)
                )
            }
        )

        OpenCVAdapter.createRectKernel(Size(7.0, 7.0), matBundle.getKernel())
        OpenCVAdapter.morphClose(matBundle.getEdges(), matBundle.getMorph(), matBundle.getKernel())

        return matBundle.getMorph()
    }

    override suspend fun detectQuad(
        morphImage: Mat,
        scaledWidth: Int,
        scaledHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        params: PipelineParams
    ): MatOfPoint? {
        return OpenCVAdapter.findBestQuad(
            morphImage, matBundle,
            scaledWidth, scaledHeight,
            originalWidth, originalHeight,
            params.minAreaFraction.toDouble(),
            params.approxPolyDPTolerance.toDouble(),
            rectangleTolerance = 20.0,
            selector = { candidates ->
                candidates.maxByOrNull { contour ->
                    scoreContourWithParams(contour, originalWidth, originalHeight, params)
                }
            }
        )
    }

    override fun captureIntermediateSnapshots(): IntermediateBitmaps =
        matBundle.captureClassicalSnapshots()

    override fun release() {
        matBundle.releaseAll()
    }
}