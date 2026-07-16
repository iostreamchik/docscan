package io.github.iostreamchik.scanner.local_files

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia.ImageOnly
import androidx.compose.animation.AnimatedContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.ImageSearch
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.github.iostreamchik.scanner.BitmapCard
import io.github.iostreamchik.scanner.old_detectors.DetectionParameters
import io.github.iostreamchik.scanner.detector.MockDocumentDetector
import io.github.iostreamchik.scanner.camera.CameraViewModel
import io.github.iostreamchik.scanner.opencv.MockMatBundle
import io.github.iostreamchik.scanner.opencv.PipelineParams
import io.github.iostreamchik.scanner.pipeline.ParameterSlider
import kotlinx.coroutines.flow.StateFlow

/**
 * Displays the original image with contour overlay, filtered preview, and warped result
 * after a file-based scan completes.
 * Also shows configurable pipeline parameters without previews.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FileScanResultScreen(
    modifier: Modifier = Modifier,
    onBack: () -> Unit,
    viewModel: CameraViewModel,
) {
    val context = LocalContext.current

    val currentParams by viewModel.pipelineParams.collectAsStateWithLifecycle()
    val error by viewModel.errorState.collectAsStateWithLifecycle()
    val isProcessing by viewModel.isProcessing.collectAsStateWithLifecycle()

    val pickMediaLauncher = rememberLauncherForActivityResult(
        contract = PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            viewModel.processPickedDocument(context, uri) {}
        }
    }

    Scaffold(
        modifier = modifier.fillMaxSize(),
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = "Scan Result",
                        fontWeight = FontWeight.Bold,
                        fontSize = 20.sp
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Default.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                }
            )
        },
        floatingActionButton = {
            FloatingActionButton(
                onClick = {
                    pickMediaLauncher.launch(PickVisualMediaRequest(ImageOnly))
                },
            ) {
                Icon(
                    imageVector = Icons.Default.ImageSearch,
                    contentDescription = "Scan from file"
                )
            }
        }
    ) { innerPadding ->
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.Center
        ) {
            // Error banner
            error?.let { msg ->
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 16.dp)
                ) {
                    Text(
                        text = msg,
                        color = Color.Red,
                        modifier = Modifier
                            .background(Color.Red.copy(alpha = 0.1f), RoundedCornerShape(8.dp))
                            .padding(12.dp)
                    )
                }
            }

            val blurBitmap by viewModel.blurBitmap.collectAsStateWithLifecycle()
            val claheBitmap by viewModel.claheBitmap.collectAsStateWithLifecycle()
            val morphBitmap by viewModel.morphBitmap.collectAsStateWithLifecycle()
            val filteredBitmap by viewModel.filteredBitmap.collectAsStateWithLifecycle()
            val originalBitmap by viewModel.originalBitmap.collectAsStateWithLifecycle()
            val resultBitmap by viewModel.resultBitmap.collectAsStateWithLifecycle()
            val hasImage = originalBitmap != null
            AnimatedContent(
                targetState = hasImage,
                label = "file_scan_content"
            ) { hasLoaded ->
                if (!hasLoaded) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .weight(1f),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center,
                    ) {
                        Icon(
                            imageVector = Icons.Default.ImageSearch,
                            contentDescription = null,
                            modifier = Modifier.size(80.dp),
                            tint = Color(0xFFBDBDBD)
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = if (isProcessing) "Processing..." else "Select a file",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Medium,
                            color = Color(0xFF757575)
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Button(
                            onClick = {
                                pickMediaLauncher.launch(PickVisualMediaRequest(ImageOnly))
                            },
                        ) {
                            Icon(
                                imageVector = Icons.Default.ImageSearch,
                                contentDescription = null,
                                modifier = Modifier.padding(end = 8.dp)
                            )
                            Text("Choose File")
                        }
                    }
                } else {
                    Column {
                        // Intermediate pipeline stages
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            // Median Blur
                            Column(
                                modifier = Modifier.weight(1f),
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Text(
                                    text = "Blur",
                                    fontSize = 14.sp,
                                    fontWeight = FontWeight.Medium,
                                )
                                BitmapCard(
                                    modifier = Modifier
                                        .height(120.dp)
                                        .clip(RoundedCornerShape(8.dp)),
                                    bitmap = blurBitmap,
                                    animated = true
                                )
                            }

                            // CLAHE
                            Column(
                                modifier = Modifier.weight(1f),
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Text(
                                    text = "CLAHE",
                                    fontSize = 14.sp,
                                    fontWeight = FontWeight.Medium,
                                )
                                BitmapCard(
                                    modifier = Modifier
                                        .height(120.dp)
                                        .clip(RoundedCornerShape(8.dp)),
                                    bitmap = claheBitmap,
                                    animated = true
                                )
                            }

                            // Morph Close
                            Column(
                                modifier = Modifier.weight(1f),
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Text(
                                    text = "Morph",
                                    fontSize = 14.sp,
                                    fontWeight = FontWeight.Medium,
                                )
                                BitmapCard(
                                    modifier = Modifier
                                        .height(120.dp)
                                        .clip(RoundedCornerShape(8.dp)),
                                    bitmap = morphBitmap,
                                    animated = true
                                )
                            }
                        }
                        Spacer(modifier = Modifier.height(8.dp))

                        // Final results row
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            // Filtered image
                            Column(
                                modifier = Modifier.weight(1f),
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Text(
                                    text = "Filtered",
                                    fontSize = 14.sp,
                                    fontWeight = FontWeight.Medium,
                                )
                                BitmapCard(
                                    modifier = Modifier
                                        .height(120.dp)
                                        .clip(RoundedCornerShape(8.dp)),
                                    bitmap = filteredBitmap,
                                    animated = true
                                )
                            }

                            // Original image
                            Column(
                                modifier = Modifier.weight(1f),
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Text(
                                    text = "Original",
                                    fontSize = 14.sp,
                                    fontWeight = FontWeight.Medium,
                                )
                                BitmapCard(
                                    modifier = Modifier
                                        .height(120.dp)
                                        .clip(RoundedCornerShape(8.dp)),
                                    bitmap = originalBitmap,
                                    animated = true
                                )
                            }

                            // Detected (warped) result
                            Column(
                                modifier = Modifier.weight(1f),
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Text(
                                    text = "Detected",
                                    fontSize = 14.sp,
                                    fontWeight = FontWeight.Medium,
                                )
                                BitmapCard(
                                    modifier = Modifier
                                        .height(120.dp)
                                        .clip(RoundedCornerShape(8.dp)),
                                    bitmap = resultBitmap,
                                    animated = true
                                )
                            }
                        }
                        Spacer(modifier = Modifier.height(16.dp))

                        // Pipeline parameters — fills remaining space, scrolls independently
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .weight(1f)
                                .verticalScroll(rememberScrollState()),
                        ) {
                            Text(
                                text = "Pipeline Parameters",
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier.padding(bottom = 12.dp)
                            )

                            PipelineParametersSection(
                                params = currentParams,
                                detectionParams = viewModel.detectionParams,
                                onParamsChange = { newParams ->
                                    viewModel.updateParams(newParams)
                                    viewModel.reprocessPickedDocument(context)
                                },
                                onEnableCannyAuto = {
                                    viewModel.enableCannyAuto()
                                    viewModel.reprocessPickedDocument(context)
                                },
                                onDisableCannyAuto = {
                                    viewModel.disableCannyAuto()
                                    viewModel.reprocessPickedDocument(context)
                                },
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun PipelineParametersSection(
    params: PipelineParams,
    detectionParams: StateFlow<DetectionParameters>,
    onParamsChange: (PipelineParams) -> Unit,
    onEnableCannyAuto: () -> Unit,
    onDisableCannyAuto: () -> Unit,
) {

    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // Median Blur
        PipelineStageControls(
            stageName = "Median Blur",
        ) {
            var kernelSize by remember { mutableIntStateOf(params.medianBlurKsize) }
            ParameterSlider(
                label = "Kernel Size",
                value = kernelSize.toFloat(),
                valueRange = 3f..21f,
                step = 2f,
                valueFormatter = { "${it.toInt()}" },
                onValueChange = {
                    val v = it.toInt().coerceIn(3, 21)
                    if (v % 2 == 0) kernelSize = v + 1 else kernelSize = v
                    onParamsChange(params.copy(medianBlurKsize = kernelSize))
                }
            )
        }

        // CLAHE
        PipelineStageControls(
            stageName = "CLAHE",
        ) {
            val autoParams by detectionParams.collectAsStateWithLifecycle()
            var modeAuto by remember { mutableStateOf(params.isClaheAuto) }
            var claheClip by remember { mutableFloatStateOf(params.claheClipLimit) }
            var claheTile by remember { mutableIntStateOf(params.claheTileSize) }

            // Auto/Manual toggle
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = if (modeAuto) "Auto (brightness-adaptive)" else "Manual",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium
                )
                Switch(
                    checked = modeAuto,
                    onCheckedChange = { newValue ->
                        modeAuto = newValue
                        if (newValue) {
                            onParamsChange(params.copy(isClaheAuto = true))
                        } else {
                            claheClip = params.claheClipLimit
                            claheTile = params.claheTileSize
                            onParamsChange(
                                params.copy(
                                    isClaheAuto = false,
                                    claheClipLimit = params.claheClipLimit,
                                    claheTileSize = params.claheTileSize
                                )
                            )
                        }
                    }
                )
            }

            // Show auto-computed CLAHE values when in auto mode
            if (modeAuto) {
                val clipText = if (autoParams.claheClipLimit.isNotBlank()) {
                    autoParams.claheClipLimit
                } else {
                    "Computing..."
                }
                Text(
                    text = "Auto: clipLimit=$clipText, tileSize=${params.claheTileSize}",
                    fontSize = 12.sp,
                    color = Color(0xFF2196F3),
                    modifier = Modifier.padding(bottom = 4.dp)
                )
            }

            if (!modeAuto) {
                Spacer(modifier = Modifier.height(4.dp))
                ParameterSlider(
                    label = "Clip Limit",
                    value = claheClip,
                    valueRange = 0.5f..8.0f,
                    step = 0.1f,
                    valueFormatter = { "%.1f".format(it) },
                    onValueChange = {
                        claheClip = it
                        onParamsChange(
                            params.copy(
                                isClaheAuto = false,
                                claheClipLimit = it,
                                claheTileSize = claheTile
                            )
                        )
                    }
                )
                ParameterSlider(
                    label = "Tile Size",
                    value = claheTile.toFloat(),
                    valueRange = 8f..64f,
                    step = 8f,
                    valueFormatter = { "${it.toInt()}" },
                    onValueChange = {
                        claheTile = it.toInt().coerceIn(8, 64)
                        onParamsChange(
                            params.copy(
                                isClaheAuto = false,
                                claheClipLimit = claheClip,
                                claheTileSize = claheTile
                            )
                        )
                    }
                )
            }
        }

        // Morph Close
        PipelineStageControls(
            stageName = "Morph Close",
        ) {
            var kernelSize by remember { mutableIntStateOf(params.morphCloseSize) }
            ParameterSlider(
                label = "Kernel Size",
                value = kernelSize.toFloat(),
                valueRange = 3f..21f,
                step = 2f,
                valueFormatter = { "${it.toInt()}" },
                onValueChange = {
                    val v = it.toInt().coerceIn(3, 21)
                    if (v % 2 == 0) kernelSize = v + 1 else kernelSize = v
                    onParamsChange(params.copy(morphCloseSize = kernelSize))
                }
            )
        }

        // Canny Edges
        PipelineStageControls(
            stageName = "Canny Edges",
        ) {
            val autoParams by detectionParams.collectAsStateWithLifecycle()
            var lowThreshold by remember { mutableFloatStateOf(params.cannyLow) }
            var highThreshold by remember { mutableFloatStateOf(params.cannyHigh) }

            if (params.cannyAutoDetect) {
                val lowText =
                    if (autoParams.cannyLow.isNotBlank()) autoParams.cannyLow else "Computing..."
                val highText =
                    if (autoParams.cannyHigh.isNotBlank()) autoParams.cannyHigh else "Computing..."
                Text(
                    text = "Auto: $lowText / $highText",
                    fontSize = 12.sp,
                    color = Color(0xFF2196F3),
                    modifier = Modifier.padding(bottom = 4.dp)
                )
            } else {
                Text(
                    text = "Manual thresholds",
                    fontSize = 12.sp,
                    color = Color.Gray,
                    modifier = Modifier.padding(bottom = 4.dp)
                )
            }

            ParameterSlider(
                label = "Low Threshold",
                value = lowThreshold,
                valueRange = 10f..100f,
                step = 5f,
                valueFormatter = { "%.0f".format(it) },
                onValueChange = {
                    lowThreshold = it
                    onParamsChange(
                        params.copy(
                            isCannyAuto = false,
                            cannyAutoDetect = false,
                            cannyLow = it,
                            cannyHigh = highThreshold
                        )
                    )
                }
            )
            ParameterSlider(
                label = "High Threshold",
                value = highThreshold,
                valueRange = 30f..300f,
                step = 10f,
                valueFormatter = { "%.0f".format(it) },
                onValueChange = {
                    highThreshold = it
                    onParamsChange(
                        params.copy(
                            isCannyAuto = false,
                            cannyAutoDetect = false,
                            cannyLow = lowThreshold,
                            cannyHigh = it
                        )
                    )
                }
            )

            // Auto-detect switch
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 8.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Auto Detect",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Medium,
                    color = Color.Gray
                )
                Switch(
                    checked = params.cannyAutoDetect,
                    onCheckedChange = { newValue ->
                        if (newValue) {
                            onEnableCannyAuto()
                        } else {
                            onDisableCannyAuto()
                        }
                    }
                )
            }
        }

        // Strong Close
        PipelineStageControls(
            stageName = "Strong Close",
        ) {
            var kernelSize by remember { mutableIntStateOf(params.strongCloseSize) }
            var enabled by remember { mutableStateOf(params.isStrongCloseEnabled) }

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = if (enabled) "Strong Closing (post-Canny)" else "Strong Close disabled — detection may degrade",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium
                )
                Switch(
                    checked = enabled,
                    onCheckedChange = { newValue ->
                        enabled = newValue
                        if (newValue) {
                            onParamsChange(params.copy(isStrongCloseEnabled = true, strongCloseSize = kernelSize))
                        } else {
                            onParamsChange(params.copy(isStrongCloseEnabled = false))
                        }
                    }
                )
            }

            if (!enabled) {
                Text(
                    text = "Closes gaps in detected edges. Disabling may cause fragmented contours.",
                    fontSize = 11.sp,
                    color = Color(0xFFFF9800),
                    modifier = Modifier.padding(top = 4.dp)
                )
            }

            if (enabled) {
                Spacer(modifier = Modifier.height(4.dp))
                ParameterSlider(
                    label = "Kernel Size",
                    value = kernelSize.toFloat(),
                    valueRange = 3f..15f,
                    step = 2f,
                    valueFormatter = { "${it.toInt()}" },
                    onValueChange = {
                        val v = it.toInt().coerceIn(3, 15)
                        if (v % 2 == 0) kernelSize = v + 1 else kernelSize = v
                        onParamsChange(params.copy(isStrongCloseEnabled = true, strongCloseSize = kernelSize))
                    }
                )
            }
        }

        // Directional Suppression
        PipelineStageControls(
            stageName = "Directional Suppression",
        ) {
            var kernelSize by remember { mutableIntStateOf(params.directionalKernelSize) }
            var enabled by remember { mutableStateOf(params.isDirectionalSuppressionEnabled) }

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = if (enabled) "Directional Suppression" else "Directional Suppression disabled — detection may degrade",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium
                )
                Switch(
                    checked = enabled,
                    onCheckedChange = { newValue ->
                        enabled = newValue
                        if (newValue) {
                            onParamsChange(params.copy(isDirectionalSuppressionEnabled = true, directionalKernelSize = kernelSize))
                        } else {
                            onParamsChange(params.copy(isDirectionalSuppressionEnabled = false))
                        }
                    }
                )
            }

            if (!enabled) {
                Text(
                    text = "Suppresses non-directional noise. Disabling may cause false contours.",
                    fontSize = 11.sp,
                    color = Color(0xFFFF9800),
                    modifier = Modifier.padding(top = 4.dp)
                )
            }

            if (enabled) {
                Spacer(modifier = Modifier.height(4.dp))
                ParameterSlider(
                    label = "Kernel Size",
                    value = kernelSize.toFloat(),
                    valueRange = 1f..31f,
                    step = 2f,
                    valueFormatter = { "${it.toInt()}" },
                    onValueChange = {
                        kernelSize = it.toInt().coerceIn(1, 31)
                        onParamsChange(params.copy(isDirectionalSuppressionEnabled = true, directionalKernelSize = kernelSize))
                    }
                )
            }
        }
    }
}

@Composable
private fun PipelineStageControls(
    stageName: String,
    content: @Composable () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp)
        ) {
            Text(
                text = stageName,
                fontSize = 14.sp,
                fontWeight = FontWeight.Medium,
                color = Color.Gray
            )
            Spacer(modifier = Modifier.height(8.dp))
            content()
        }
    }
}

@Preview
@Composable
@OptIn(ExperimentalMaterial3Api::class)
private fun FileScanResultPreview() {
    Surface {
        FileScanResultScreen(
            viewModel = viewModel {
                CameraViewModel(
                    matBundle = MockMatBundle(),
                    detector = MockDocumentDetector()
                )
            },
            onBack = {}
        )
    }
}
