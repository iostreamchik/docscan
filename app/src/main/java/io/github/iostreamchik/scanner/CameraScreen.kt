package io.github.iostreamchik.scanner

import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shadow
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalInspectionMode
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import org.opencv.android.OpenCVLoader
import org.opencv.core.MatOfPoint
import java.util.concurrent.Executors
import kotlin.math.max

@Composable
fun CameraScreen(
    modifier: Modifier = Modifier,
    viewModel: CameraViewModel = viewModel()
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val previewView = remember { PreviewView(context) }

    data class ContourData(
        val contours: List<MatOfPoint>,
        val frameWidth: Int,
        val frameHeight: Int,
        val rotation: Int
    )

    val contourState = remember { mutableStateOf<ContourData?>(null) }

    Box(modifier = modifier) {

        val isPreview = LocalInspectionMode.current
        if (isPreview.not()) {
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { previewView },
            )
        } else {
            Box(modifier = Modifier.fillMaxSize())
        }
        val cornerRadius = rememberDeviceCornerRadiusDp()
        Column(
            modifier = Modifier
                .padding(horizontal = 16.dp)
                .align(Alignment.BottomStart)
        ) {
            val exposure by viewModel.exposureStateFlow.collectAsStateWithLifecycle()
            Text(
                text = exposure, fontWeight = FontWeight.Bold, style = TextStyle(
                    fontSize = 30.sp,
                    shadow = Shadow(
                        color = Color.Black.copy(alpha = 0.5f),
                        offset = Offset(5f, 5f),
                        blurRadius = 8f
                    )
                )
            )
            Row(
                Modifier
                    .navigationBarsPadding(), horizontalArrangement = Arrangement.SpaceBetween
            ) {
                val bmp by viewModel.filteredBitmap.collectAsStateWithLifecycle()
                bmp?.let {
                    Card(
                        shape = RoundedCornerShape(cornerRadius),
                        elevation = CardDefaults.cardElevation(8.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = Color.Transparent,
                        ),
                    ) {
                        Image(
                            bitmap = it.asImageBitmap(),
                            contentDescription = null,
                        )
                    }
                }
                Spacer(modifier = Modifier.width(16.dp))
                val resultBmp by viewModel.resultBitmap.collectAsStateWithLifecycle()
                resultBmp?.let {
                    Card(
                        shape = RoundedCornerShape(4.dp),
                        elevation = CardDefaults.cardElevation(8.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = Color.Transparent,
                        ),
                    ) {
                        Image(
                            bitmap = it.asImageBitmap(),
                            contentDescription = null,
                        )
                    }
                }

            }
        }
        Canvas(modifier = Modifier.matchParentSize()) {

            val data = contourState.value ?: return@Canvas

            val originalW = data.frameWidth.toFloat()
            val originalH = data.frameHeight.toFloat()

            val previewW = size.width
            val previewH = size.height

            // Determine rotated dimensions
            val rotatedW =
                if (data.rotation == 90 || data.rotation == 270) originalH else originalW

            val rotatedH =
                if (data.rotation == 90 || data.rotation == 270) originalW else originalH

            val scale = max(previewW / rotatedW, previewH / rotatedH)

            val scaledW = rotatedW * scale
            val scaledH = rotatedH * scale

            val dx = (previewW - scaledW) / 2f
            val dy = (previewH - scaledH) / 2f

            fun rotatePoint(x: Float, y: Float): Offset {
                return when (data.rotation) {
                    90 -> Offset(originalH - y, x)
                    180 -> Offset(originalW - x, originalH - y)
                    270 -> Offset(y, originalW - x)
                    else -> Offset(x, y)
                }
            }
            data.contours.forEach { contour ->
                val points = contour.toArray()

                for (i in points.indices) {

                    val p1 = rotatePoint(
                        points[i].x.toFloat(),
                        points[i].y.toFloat()
                    )

                    val p2 = rotatePoint(
                        points[(i + 1) % points.size].x.toFloat(),
                        points[(i + 1) % points.size].y.toFloat()
                    )

                    drawLine(
                        color = Color.Red.copy(alpha = 0.5f),
                        start = Offset(
                            p1.x * scale + dx,
                            p1.y * scale + dy
                        ),
                        end = Offset(
                            p2.x * scale + dx,
                            p2.y * scale + dy
                        ),
                        strokeWidth = 8f
                    )
                }
            }
        }
        val context = LocalContext.current
        LaunchedEffect(Unit) {
            val cameraProvider = ProcessCameraProvider
                .getInstance(context)
                .get()

            // 1. Define the Resolution Selector
            // This tells CameraX to prefer the highest resolution possible
            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(
                        // Setting a very high resolution acts as a hint to pick the max available
                        Size(1000, 1000),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                    )
                )
                .build()

            // 2. Apply the selector to the Preview
            val preview = Preview.Builder()
                .setResolutionSelector(resolutionSelector)
                .build()
                .also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

            // 3. Apply the SAME selector to the ImageAnalyzer
            // IMPORTANT: If you don't do this, the analyzer might use a low-res
            // stream even if the preview is high-res.
            val imageAnalyzer = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalyzer.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                val contours = viewModel.processFrame(imageProxy)

                contourState.value = ContourData(
                    contours = contours,
                    frameWidth = imageProxy.width,
                    frameHeight = imageProxy.height,
                    rotation = imageProxy.imageInfo.rotationDegrees
                )
                imageProxy.close()
            }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e("CameraX", "Use case binding failed", e)
            }
        }
    }

}

@androidx.compose.ui.tooling.preview.Preview
@Composable
private fun CameraScreenPreview() {
    Surface() {
        CameraScreen(
            viewModel = viewModel() { CameraViewModel() }
        )
    }
}
