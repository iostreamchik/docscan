package io.github.iostreamchik.scanner

import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalInspectionMode
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
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
        Row(
            Modifier
                .align(Alignment.BottomStart)
                .padding(horizontal = 16.dp)
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
                        color = Color.Green.copy(alpha = 0.5f),
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

            val preview = Preview.Builder().build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(
                    ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                )
                .build()

            imageAnalyzer.setAnalyzer(
                Executors.newSingleThreadExecutor()
            ) { imageProxy ->

                val contours = viewModel.processFrame(imageProxy)

                contourState.value = ContourData(
                    contours = contours,
                    frameWidth = imageProxy.width,
                    frameHeight = imageProxy.height,
                    rotation = imageProxy.imageInfo.rotationDegrees
                )

                imageProxy.close()
            }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalyzer
            )
        }
    }

}

@androidx.compose.ui.tooling.preview.Preview
@Composable
private fun CameraScreenPreview() {
    Surface() {
        CameraScreen()
    }
}
