package io.github.iostreamchik.scanner.camera

import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
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
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Image
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.FloatingActionButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shadow
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
import io.github.iostreamchik.scanner.BitmapCard
import io.github.iostreamchik.scanner.ContourCanvas
import io.github.iostreamchik.scanner.opencv.MockMatBundle
import io.github.iostreamchik.scanner.rememberDeviceCornerRadiusDp

@Composable
fun CameraScreen(
    modifier: Modifier = Modifier,
    viewModel: CameraViewModel,
    toScanFromFile: () -> Unit = {},
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember { PreviewView(context) }

    val contourState =
        remember { mutableStateOf<ContourData?>(null) }

    val exposure by viewModel.exposureStateFlow.collectAsStateWithLifecycle()
    val filteredBitmap by viewModel.filteredBitmap.collectAsStateWithLifecycle()
    val resultBitmap by viewModel.resultBitmap.collectAsStateWithLifecycle()
    val errorState by viewModel.errorState.collectAsStateWithLifecycle()
    val manualCannyHigh by viewModel.manualCannyHigh.collectAsStateWithLifecycle()
    val cornerRadius = rememberDeviceCornerRadiusDp()

    // Throttle for contourState updates (matches ViewModel's UI_UPDATE_THROTTLE_MS)
    val lastContourUpdateTime = remember { mutableLongStateOf(0L) }
    val CONTOUR_UPDATE_THROTTLE_MS = 30L

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

        errorState?.let { state ->
            Surface(
                color = Color.Red.copy(alpha = 0.8f),
                modifier = Modifier
                    .align(Alignment.Center)
                    .padding(16.dp)
            ) {
                Text(
                    text = state,
                    color = Color.White,
                    modifier = Modifier.padding(8.dp)
                )
            }
        }

        Column(
            modifier = Modifier
                .padding(horizontal = 16.dp)
                .align(Alignment.BottomStart)
        ) {
            Text(
                text = exposure,
                fontWeight = FontWeight.Bold, style = TextStyle(
                    color = Color.White,
                    fontSize = 20.sp,
                    shadow = Shadow(
                        color = Color.Black.copy(alpha = 0.5f),
                        offset = Offset(0f, 0f),
                        blurRadius = 8f
                    )
                )
            )
            
            // Debug: Canny High threshold slider (hidden when auto = 0f)
            if (manualCannyHigh != 0f) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 8.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Canny High",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Medium,
                            color = Color.White.copy(alpha = 0.7f)
                        )
                        Text(
                            text = "${manualCannyHigh.toInt()}",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                    }
                    Slider(
                        value = manualCannyHigh,
                        onValueChange = {
                            viewModel.setManualCannyHigh(it)
                        },
                        valueRange = 1f..255f,
                        steps = 22,
                        modifier = Modifier.fillMaxWidth(),
                        colors = SliderDefaults.colors()
                    )
                }
            }
            
            Row(
                Modifier
                    .navigationBarsPadding()
                    .fillMaxWidth()
                    .height(260.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                BitmapCard(
                    bitmap = filteredBitmap,
                    shape = RoundedCornerShape(cornerRadius)
                )
                Spacer(modifier = Modifier.width(16.dp))
                BitmapCard(
                    bitmap = resultBitmap,
                    shape = RoundedCornerShape(4.dp)
                )

            }
        }
        FloatingActionButton(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp),
            onClick = toScanFromFile,
            elevation = FloatingActionButtonDefaults.elevation(8.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Image,
                contentDescription = "Scan from file"
            )
        }
        ContourCanvas(
            contourState = contourState,
            modifier = Modifier.matchParentSize()
        )
        LaunchedEffect(Unit) {
            val cameraProvider = ProcessCameraProvider
                .getInstance(context)
                .get()

            // 1. Define the Resolution Selector
            // This tells CameraX to prefer the highest resolution possible
            val size = 1000
            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(
                        // Setting a very high resolution acts as a hint to pick the max available
                        Size(size, size),
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

            imageAnalyzer.setAnalyzer(viewModel.cameraExecutor) { imageProxy ->
                val contours = viewModel.processFrame(imageProxy)

                val now = System.currentTimeMillis()
                if (now - lastContourUpdateTime.longValue >= CONTOUR_UPDATE_THROTTLE_MS) {
                    contourState.value?.release()
                    contourState.value = ContourData(
                        contours = contours,
                        frameWidth = imageProxy.width,
                        frameHeight = imageProxy.height,
                        rotation = imageProxy.imageInfo.rotationDegrees
                    )
                    lastContourUpdateTime.longValue = now
                } else {
                    contours.forEach { it.release() }
                }

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
                // Release any existing contour data to prevent memory leak on error
                contourState.value?.release()
                viewModel.setError("Camera initialization failed")
            }
        }

        // Release contour data when the composable is disposed to prevent memory leaks
        DisposableEffect(Unit) {
            onDispose {
                contourState.value?.release()
            }
        }
    }

}

@androidx.compose.ui.tooling.preview.Preview
@Composable
private fun CameraScreenPreview() {
    Surface() {
        CameraScreen(
            viewModel = viewModel() { CameraViewModel(MockMatBundle()) }
        )
    }
}
