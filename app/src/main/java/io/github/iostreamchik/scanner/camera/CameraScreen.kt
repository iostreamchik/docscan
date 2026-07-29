package io.github.iostreamchik.scanner.camera

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.provider.Settings
import android.util.Log
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FlashlightOn
import androidx.compose.material.icons.filled.Image
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.FloatingActionButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalInspectionMode
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.github.iostreamchik.scanner.BitmapCard
import io.github.iostreamchik.scanner.ContourCanvas
import io.github.iostreamchik.scanner.detector.MockDocumentDetector
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
    val detectionParams by viewModel.detectionParams.collectAsStateWithLifecycle()
    val filteredBitmap by viewModel.filteredBitmap.collectAsStateWithLifecycle()
    val onnxMaskBitmap by viewModel.onnxMaskBitmap.collectAsStateWithLifecycle()
    val resultBitmap by viewModel.resultBitmap.collectAsStateWithLifecycle()
    val errorState by viewModel.errorState.collectAsStateWithLifecycle()
    val torchOn by viewModel.torchOn.collectAsStateWithLifecycle()
    val cornerRadius = rememberDeviceCornerRadiusDp()

    // Store the bound Camera reference for torch state observation
    val boundCamera = remember { mutableStateOf<androidx.camera.core.Camera?>(null) }

    // Camera permission handling — per Android docs: request in-context using RequestPermission contract
    val (permissionGranted, setPermissionGranted) = remember { mutableStateOf(false) }
    val (showRationale, setShowRationale) = remember { mutableStateOf(false) }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { granted ->
            setPermissionGranted(granted)
            if (!granted) {
                val activity = context as? ComponentActivity
                setShowRationale(activity?.shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) ?: false)
            }
        }
    )
    val settingsLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartActivityForResult(),
        onResult = {
            setPermissionGranted(
                ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
            )
        }
    )

    LaunchedEffect(Unit) {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            setPermissionGranted(true)
        } else {
            val activity = context as? ComponentActivity
            val shouldShow = activity?.shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) ?: false
            setShowRationale(shouldShow)
            if (!shouldShow) {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    // Throttle for contourState updates (matches ViewModel's UI_UPDATE_THROTTLE_MS)
    val lastContourUpdateTime = remember { mutableLongStateOf(0L) }
    val CONTOUR_UPDATE_THROTTLE_MS = 30L

    Box(modifier = modifier) {

        Column {

            // Camera preview container - 70% height with rounded bottom corners
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .fillMaxHeight(0.7f)
                    .clip(RoundedCornerShape(bottomStart = 8.dp, bottomEnd = 8.dp))
            ) {
                if (!permissionGranted && !LocalInspectionMode.current) {
                    // Permission denied overlay
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(Color.Black.copy(alpha = 0.85f))
                            .padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        Text(
                            text = "Camera Access Required",
                            style = MaterialTheme.typography.headlineSmall,
                            color = Color.White
                        )
                        Text(
                            text = if (showRationale) {
                                "This app needs camera access to detect and scan documents in real time."
                            } else {
                                "Camera access was denied. Please enable it in your app settings to use the scanner."
                            },
                            style = MaterialTheme.typography.bodyMedium,
                            color = Color.White.copy(alpha = 0.7f),
                            modifier = Modifier.padding(top = 8.dp, bottom = 24.dp)
                        )
                        Box(
                            modifier = Modifier
                                .clickable {
                                    if (showRationale) {
                                        cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                                    } else {
                                        val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                                            setData(android.net.Uri.parse("package:${context.packageName}"))
                                        }
                                        settingsLauncher.launch(intent)
                                    }
                                }
                                .background(
                                    color = MaterialTheme.colorScheme.primary,
                                    shape = RoundedCornerShape(24.dp)
                                )
                        ) {
                            Text(
                                text = if (showRationale) "Grant Camera Access" else "Open Settings",
                                modifier = Modifier.padding(horizontal = 24.dp, vertical = 12.dp),
                                style = MaterialTheme.typography.bodyLarge,
                                color = MaterialTheme.colorScheme.onPrimary
                            )
                        }
                    }
                } else {
                    val isPreview = LocalInspectionMode.current
                    if (isPreview.not()) {
                        AndroidView(
                            modifier = Modifier.fillMaxSize(),
                            factory = { previewView },
                        )

                        LaunchedEffect(previewView) {
                            val cameraProvider = ProcessCameraProvider
                                .getInstance(context)
                                .get()

                            val size = 2000
                            val resolutionSelector = ResolutionSelector.Builder()
                                .setResolutionStrategy(
                                    ResolutionStrategy(
                                        Size(size, size),
                                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                                    )
                                )
                                .build()

                            val preview = Preview.Builder()
                                .setResolutionSelector(resolutionSelector)
                                .build()
                                .also {
                                    it.surfaceProvider = previewView.surfaceProvider
                                }

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
                                boundCamera.value = cameraProvider.bindToLifecycle(
                                    lifecycleOwner,
                                    CameraSelector.DEFAULT_BACK_CAMERA,
                                    preview,
                                    imageAnalyzer
                                )
                            } catch (e: Exception) {
                                Log.e("CameraX", "Use case binding failed", e)
                                contourState.value?.release()
                                viewModel.setError("Camera initialization failed")
                            }
                        }
                    } else {
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(Color.Gray)
                        )
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

                ContourCanvas(
                    contourState = contourState,
                    modifier = Modifier.fillMaxSize()
                )

                // Torch toggle button - top right corner
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .statusBarsPadding()
                        .padding(horizontal = 16.dp)
                ) {
                    IconButton(
                        modifier = Modifier.align(Alignment.CenterEnd),
                        onClick = { viewModel.toggleTorch() }
                    ) {
                        Icon(
                            imageVector = Icons.Default.FlashlightOn,
                            contentDescription = "Toggle torch",
                            tint = if (torchOn) Color.Black else Color.Black.copy(alpha = 0.5f)
                        )
                    }
                }

                // Detection params info box - bottom of preview
                Surface(
                    modifier = Modifier
                        .align(Alignment.BottomStart)
                        .padding(start = 8.dp, bottom = 8.dp),
                    shape = RoundedCornerShape(8.dp),
                    color = Color.Black.copy(alpha = 0.5f)
                ) {
                    Text(
                        modifier = Modifier.padding(8.dp),
                        text = "CLAHE: ${detectionParams.claheClipLimit}" +
                                "\nCanny: ${detectionParams.cannyLow}/${detectionParams.cannyHigh}" +
                                "\nBright: ${detectionParams.brightness}",
                        color = Color.White.copy(alpha = 0.8f),
                        fontSize = 10.sp,
                    )
                }
                }
            }

            // Bottom section - preview bitmaps
            Row(
                modifier = Modifier
                    .fillMaxHeight()
                    .padding(horizontal = 8.dp)
                    .padding(top = 8.dp)
                    .navigationBarsPadding(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Surface(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxSize(),
                    shape = RoundedCornerShape(
                        topStart = 8.dp,
                        topEnd = 8.dp,
                        bottomStart = cornerRadius,
                        bottomEnd = 8.dp
                    ),
                    color = MaterialTheme.colorScheme.surfaceVariant
                ) {
                    val isPreview = LocalInspectionMode.current
                    if (isPreview) {
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(Color.Gray)
                        )
                    } else {
                        BitmapCard(
                            bitmap = onnxMaskBitmap ?: filteredBitmap,
                            modifier = Modifier.fillMaxSize()
                        )
                    }
                }
                Surface(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxSize(),
                    shape = RoundedCornerShape(
                        topStart = 8.dp,
                        topEnd = 8.dp,
                        bottomStart = 8.dp,
                        bottomEnd = cornerRadius
                    ),
                    color = MaterialTheme.colorScheme.surfaceVariant
                ) {
                    val isPreview = LocalInspectionMode.current
                    if (isPreview) {
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(Color.Gray)
                        )
                    } else {
                        BitmapCard(
                            bitmap = resultBitmap,
                            modifier = Modifier.fillMaxSize()
                        )
                    }
                }
            }
        }

        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .navigationBarsPadding()
        ) {
            FloatingActionButton(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .clip(RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp))
                    .background(MaterialTheme.colorScheme.background)
                    .padding(top = 8.dp, start = 8.dp, end = 8.dp),
                onClick = toScanFromFile,
                elevation = FloatingActionButtonDefaults.elevation(0.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.Image,
                    contentDescription = "Scan from file"
                )
            }
                }

        // Release contour data when the composable is disposed to prevent memory leaks
        DisposableEffect(Unit) {
            onDispose {
                contourState.value?.release()
            }
        }

        // Observe torch state from CameraX and sync to ViewModel
        LaunchedEffect(boundCamera.value) {
            boundCamera.value?.cameraInfo?.torchState?.observe(lifecycleOwner) { torchState ->
                viewModel.setTorchOpposite(torchState == androidx.camera.core.TorchState.ON)
            }
        }

        // Apply torch state changes from ViewModel to CameraX
        LaunchedEffect(torchOn, boundCamera.value) {
            boundCamera.value?.cameraControl?.apply {
                enableTorch(torchOn)
            }
        }
    }

}

@androidx.compose.ui.tooling.preview.Preview
@Composable
private fun CameraScreenPreview() {
    Surface() {
        CameraScreen(
            modifier = Modifier.background(MaterialTheme.colorScheme.background),
            viewModel = viewModel {
                CameraViewModel(
                    matBundle = MockMatBundle(),
                    detector = MockDocumentDetector()
                )
            }
        )
    }
}
