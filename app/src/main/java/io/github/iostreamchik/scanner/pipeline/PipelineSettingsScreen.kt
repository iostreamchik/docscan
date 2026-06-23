package io.github.iostreamchik.scanner.pipeline

import android.content.Context
import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia.ImageOnly
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CloudUpload
import androidx.compose.material.icons.filled.ImageSearch
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.github.iostreamchik.scanner.opencv.MockMatBundle
import io.github.iostreamchik.scanner.opencv.PipelineParams
import io.github.iostreamchik.scanner.opencv.*

/**
 * Pipeline Settings screen — pick an image, tweak detection parameters,
 * and see real-time previews at every stage of the OpenCV pipeline.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PipelineSettingsScreen(
    modifier: Modifier = Modifier,
    onBack: () -> Unit,
    viewModel: PipelineSettingsViewModel,
) {
    val context = LocalContext.current

    // Collect state
    val originalBitmap by viewModel.originalBitmap.collectAsStateWithLifecycle()
    val previewBitmaps by viewModel.previewBitmaps.collectAsStateWithLifecycle()
    val resultBitmap by viewModel.resultBitmap.collectAsStateWithLifecycle()
    val isProcessing by viewModel.isProcessing.collectAsStateWithLifecycle()
    val error by viewModel.error.collectAsStateWithLifecycle()
    val detectedQuad by viewModel.detectedQuad.collectAsStateWithLifecycle()
    val hasDetectedDocument by viewModel.hasDetectedDocument.collectAsStateWithLifecycle()
    val currentParams by viewModel.currentParams.collectAsStateWithLifecycle()
    val avgBrightness by viewModel.avgBrightness.collectAsStateWithLifecycle()
    val contrast by viewModel.contrast.collectAsStateWithLifecycle()

    // File picker
    val pickMediaLauncher = rememberLauncherForActivityResult(
        contract = PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            viewModel.loadImage(context, uri)
        }
    }

    Scaffold(
        modifier = modifier.fillMaxSize(),
        topBar = {
            TopAppBar(
                title = {
                    Row(
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = "Pipeline Settings",
                            fontWeight = FontWeight.Bold,
                            fontSize = 20.sp
                        )
                        if (isProcessing) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(20.dp),
                                strokeWidth = 2.dp
                            )
                        }
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Default.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                },
                actions = {
                    if (hasDetectedDocument) {
                        if (resultBitmap != null) {
                            Box(
                                modifier = Modifier
                                    .size(24.dp)
                                    .padding(2.dp)
                            ) {
                                Image(
                                    bitmap = resultBitmap!!.asImageBitmap(),
                                    contentDescription = "Recognized document",
                                    modifier = Modifier.fillMaxSize()
                                )
                            }
                        }
                        Spacer(modifier = Modifier.width(4.dp))
                    }
                    IconButton(onClick = {
                        viewModel.resetParams(context)
                    }) {
                        Icon(
                            imageVector = Icons.Default.Refresh,
                            contentDescription = "Reset to defaults"
                        )
                    }
                }
            )
        },
        floatingActionButton = {
            FloatingActionButton(
                onClick = {
                    pickMediaLauncher.launch(PickVisualMediaRequest(ImageOnly))
                }
            ) {
                Icon(
                    imageVector = Icons.Default.ImageSearch,
                    contentDescription = "Pick another image"
                )
            }
        }
    ) { innerPadding ->
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (originalBitmap == null) {
                // Empty state — no image loaded yet
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center,
                ) {
                    Icon(
                        imageVector = Icons.Default.CloudUpload,
                        contentDescription = null,
                        modifier = Modifier.size(80.dp),
                        tint = Color(0xFF9E9E9E)
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "Select a file",
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

                // Brightness & Contrast info row
                if (avgBrightness != null && contrast != null) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 8.dp),
                        horizontalArrangement = Arrangement.SpaceEvenly,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        InfoChip(label = "Brightness", value = avgBrightness!!.toInt().toString())
                        InfoChip(label = "Contrast", value = "${contrast!!.toInt()}")
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                }

                // Original image
                originalBitmap?.let { bmp ->
                    PipelineStageCard(
                        title = "Original Image",
                        bitmap = bmp,
                        isReadonly = true
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                }

                // Quad overlay — original image with detected quad, shown as a prominent card
                previewBitmaps["Quad"]?.let { bmp ->
                    QuadCard(
                        quadBitmap = bmp,
                        resultBitmap = resultBitmap,
                        hasDocument = hasDetectedDocument,
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                }

                // Pipeline stage previews with parameter controls
                val stageOrder = listOf(
                    "Grayscale",
                    "Median Blur",
                    "CLAHE",
                    "Morph Close",
                    "Canny Edges",
                    "Strong Close",
                    "Directional Suppression"
                )

                stageOrder.forEach { stageName ->
                    previewBitmaps[stageName]?.let { bmp ->
                        val isReadonly = stageName in setOf(
                            "Contour Map",
                            "Detected Quad"
                        )
                        PipelineStageCard(
                            title = stageName,
                            bitmap = bmp,
                            isReadonly = isReadonly,
                            parameters = getParametersForStage(stageName, currentParams, viewModel, context)
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                    }
                }

                // Reprocess button — shown inside QuadCard when processing
                if (isProcessing) {
                    Button(
                        onClick = {
                            viewModel.resetParams(context)
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(
                            imageVector = Icons.Default.Refresh,
                            contentDescription = null,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Reprocess")
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                }

                // Bottom padding for FAB
                Spacer(modifier = Modifier.height(80.dp))
            }
        }
    }
}

/**
 * Returns the parameter controls for a given pipeline stage.
 * Each control uses its own local state and debounces to the ViewModel.
 */
@Composable
private fun getParametersForStage(
    stageName: String,
    currentParams: PipelineParams,
    viewModel: PipelineSettingsViewModel,
    context: Context
): @Composable () -> Unit {
    val p = currentParams
    return {
        when (stageName) {
            "Grayscale" -> {}

            "Median Blur" -> {
                var kernelSize by remember { mutableIntStateOf(p.medianBlurKsize) }
                val debouncedKernel = debounceInt(kernelSize, 300)
                LaunchedEffect(debouncedKernel) {
                    if (debouncedKernel != p.medianBlurKsize) {
                        viewModel.updateParamSafely(
                            p.copy(
                                isAuto = false,
                                medianBlurKsize = debouncedKernel
                            )
                        ) { context }
                    }
                }
                ParameterSlider(
                    label = "Kernel Size",
                    value = kernelSize.toFloat(),
                    valueRange = 3f..21f,
                    step = 2f,
                    valueFormatter = { "${it.toInt()}" },
                    onValueChange = {
                        val v = it.toInt().coerceIn(3, 21)
                        if (v % 2 == 0) kernelSize = v + 1 else kernelSize = v
                    }
                )
            }

            "CLAHE" -> {
                val isAuto = currentParams.isAuto
                var clipLimit by remember { mutableFloatStateOf(p.claheClipLimit) }
                var tileSize by remember { mutableIntStateOf(p.claheTileSize) }
                var modeAuto by remember { mutableStateOf(isAuto) }

                val debouncedClip = debounceFloat(clipLimit, 300)
                val debouncedTile = debounceInt(tileSize, 300)
                LaunchedEffect(modeAuto, debouncedClip, debouncedTile) {
                    if (modeAuto) {
                        viewModel.updateParamSafely(PipelineParams()) { context }
                    } else if (debouncedClip != p.claheClipLimit ||
                        debouncedTile != p.claheTileSize
                    ) {
                        viewModel.updateParamSafely(
                            p.copy(
                                isAuto = false,
                                claheClipLimit = debouncedClip,
                                claheTileSize = debouncedTile
                            )
                        ) { context }
                    }
                }

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
                        onCheckedChange = { modeAuto = it }
                    )
                }

                if (!modeAuto) {
                    Spacer(modifier = Modifier.height(4.dp))
                    ParameterSlider(
                        label = "Clip Limit",
                        value = clipLimit,
                        valueRange = 0.5f..8.0f,
                        step = 0.1f,
                        valueFormatter = { "%.1f".format(it) },
                        onValueChange = { clipLimit = it }
                    )
                    ParameterSlider(
                        label = "Tile Size",
                        value = tileSize.toFloat(),
                        valueRange = 8f..64f,
                        step = 8f,
                        valueFormatter = { "${it.toInt()}" },
                        onValueChange = { tileSize = it.toInt().coerceIn(8, 64) }
                    )
                }
            }

            "Morph Close" -> {
                var kernelSize by remember { mutableIntStateOf(p.morphCloseSize) }
                val debouncedKernel = debounceInt(kernelSize, 300)
                LaunchedEffect(debouncedKernel) {
                    if (debouncedKernel != p.morphCloseSize) {
                        viewModel.updateParamSafely(
                            p.copy(
                                isAuto = false,
                                morphCloseSize = debouncedKernel)
                        ) { context }
                    }
                }
                ParameterSlider(
                    label = "Kernel Size",
                    value = kernelSize.toFloat(),
                    valueRange = 3f..21f,
                    step = 2f,
                    valueFormatter = { "${it.toInt()}" },
                    onValueChange = {
                        val v = it.toInt().coerceIn(3, 21)
                        if (v % 2 == 0) kernelSize = v + 1 else kernelSize = v
                    }
                )
            }

            "Canny Edges" -> {
                var lowThreshold by remember { mutableFloatStateOf(p.cannyLow) }
                var highThreshold by remember { mutableFloatStateOf(p.cannyHigh) }

                val debouncedLow = debounceFloat(lowThreshold, 300)
                val debouncedHigh = debounceFloat(highThreshold, 300)
                LaunchedEffect(debouncedLow, debouncedHigh) {
                    if (debouncedLow != p.cannyLow ||
                        debouncedHigh != p.cannyHigh
                    ) {
                        viewModel.updateParamSafely(
                            p.copy(
                                isAuto = false,
                                cannyLow = debouncedLow,
                                cannyHigh = debouncedHigh
                            )
                        ) { context }
                    }
                }

                ParameterSlider(
                    label = "Low Threshold",
                    value = lowThreshold,
                    valueRange = 10f..100f,
                    step = 5f,
                    valueFormatter = { "%.0f".format(it) },
                    onValueChange = { lowThreshold = it }
                )
                ParameterSlider(
                    label = "High Threshold",
                    value = highThreshold,
                    valueRange = 30f..300f,
                    step = 10f,
                    valueFormatter = { "%.0f".format(it) },
                    onValueChange = { highThreshold = it }
                )
            }

            "Strong Close" -> {
                var kernelSize by remember { mutableIntStateOf(p.strongCloseSize) }
                val debouncedKernel = debounceInt(kernelSize, 300)
                LaunchedEffect(debouncedKernel) {
                    if (debouncedKernel != p.strongCloseSize) {
                        viewModel.updateParamSafely(
                            p.copy(
                                isAuto = false,
                                strongCloseSize = debouncedKernel)
                        ) { context }
                    }
                }
                ParameterSlider(
                    label = "Kernel Size",
                    value = kernelSize.toFloat(),
                    valueRange = 3f..15f,
                    step = 2f,
                    valueFormatter = { "${it.toInt()}" },
                    onValueChange = {
                        val v = it.toInt().coerceIn(3, 15)
                        if (v % 2 == 0) kernelSize = v + 1 else kernelSize = v
                    }
                )
            }

            "Directional Suppression" -> {
                var kernelSize by remember { mutableIntStateOf(p.directionalKernelSize) }

                val debouncedKernel = debounceInt(kernelSize, 300)
                LaunchedEffect(debouncedKernel) {
                    if (debouncedKernel != p.directionalKernelSize) {
                        viewModel.updateParamSafely(
                            p.copy(
                                isAuto = false,
                                directionalKernelSize = debouncedKernel
                            )
                        ) { context }
                    }
                }

                ParameterSlider(
                    label = "Kernel Size",
                    value = kernelSize.toFloat(),
                    valueRange = 1f..31f,
                    step = 2f,
                    valueFormatter = { "${it.toInt()}" },
                    onValueChange = { kernelSize = it.toInt().coerceIn(1, 31) }
                )
            }

            "Quad" -> {}
        }
    }
}

/**
 * Prominent card showing the original image with the detected quad overlaid,
 * alongside the recognized (cropped) document preview.
 */
@Composable
private fun QuadCard(
    quadBitmap: Bitmap,
    resultBitmap: Bitmap?,
    hasDocument: Boolean,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        elevation = CardDefaults.cardElevation(0.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp)
        ) {
            // Header with status
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.ImageSearch,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(20.dp)
                    )
                    Text(
                        text = "Result",
                        fontWeight = FontWeight.Bold,
                        fontSize = 16.sp
                    )
                }
                if (hasDocument) {
                    Icon(
                        imageVector = Icons.Default.CheckCircle,
                        contentDescription = "Document recognized",
                        tint = MaterialTheme.colorScheme.primary
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Two-column layout: Quad overlay + Recognized document
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // Quad overlay
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Text(
                        text = "Quad",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Medium,
                        color = Color.Gray
                    )
                    Image(
                        bitmap = quadBitmap.asImageBitmap(),
                        contentDescription = "Detected quad overlay",
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(160.dp)
                            .background(Color.Black.copy(alpha = 0.05f), RoundedCornerShape(8.dp))
                    )
                }

                // Recognized document
                if (resultBitmap != null) {
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Text(
                            text = "Recognized Document",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Medium,
                            color = Color.Gray
                        )
                        Image(
                            bitmap = resultBitmap.asImageBitmap(),
                            contentDescription = "Recognized document preview",
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(160.dp)
                                .background(Color.Black.copy(alpha = 0.05f), RoundedCornerShape(8.dp))
                        )
                    }
                }
            }
        }
    }
}

/**
 * A single pipeline stage card showing a preview bitmap and optional parameter controls.
 * Collapsible: tap the header to expand/collapse.
 */
@Composable
private fun PipelineStageCard(
    title: String,
    bitmap: Bitmap,
    isReadonly: Boolean = false,
    parameters: @Composable () -> Unit = {},
) {
    var isExpanded by remember { mutableStateOf(!isReadonly) }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        elevation = CardDefaults.cardElevation(0.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
        )
    ) {
        Column {
            // Collapsible header
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { isExpanded = !isExpanded }
                    .padding(horizontal = 12.dp, vertical = 10.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = title,
                    fontWeight = FontWeight.Medium,
                    fontSize = 16.sp,
                    modifier = Modifier.weight(1f)
                )
                Icon(
                    imageVector = Icons.Default.KeyboardArrowDown,
                    contentDescription = if (isExpanded) "Collapse $title" else "Expand $title",
                    modifier = Modifier
                        .size(24.dp)
                        .graphicsLayer { rotationZ = if (isExpanded) 180f else 0f }
                )
            }

            // Expandable content
            if (isExpanded) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 4.dp)
                ) {
                    // Preview bitmap
                    // Don't use remember — the Android framework caches ImageBitmap internally.
                    // Using remember(bitmap) with a Bitmap object as key is unsafe: if the
                    // underlying Bitmap is recycled (e.g., setPreviewBitmaps recycles old
                    // bitmaps), the stale ImageBitmap wrapper causes "Canvas: trying to use
                    // a recycled bitmap". Creating a fresh ImageBitmap each composition
                    // avoids this race condition.
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = "$title preview",
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(200.dp)
                            .background(Color.Black.copy(alpha = 0.05f), RoundedCornerShape(8.dp))
                    )

                    // Parameter controls
                    Spacer(modifier = Modifier.height(8.dp))
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        parameters()
                    }
                }
            }
        }
    }
}

/**
 * Prominent card showing the recognized (cropped/straightened) document.
 * Always expanded, with a reprocess button when processing is active.
 */
@Composable
private fun RecognizedDocumentCard(
    bitmap: Bitmap?,
    isProcessing: Boolean,
    onReprocess: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        elevation = CardDefaults.cardElevation(6.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp)
        ) {
            // Header with status
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.ImageSearch,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(20.dp)
                    )
                    Text(
                        text = "Recognized Document",
                        fontWeight = FontWeight.Bold,
                        fontSize = 16.sp
                    )
                }
                if (isProcessing) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        strokeWidth = 2.dp
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Preview bitmap
            // Don't use remember — see BitmapCard comment for rationale.
            bitmap?.let { bmp ->
                Image(
                    bitmap = bmp.asImageBitmap(),
                    contentDescription = "Recognized document preview",
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(200.dp)
                        .background(Color.Black.copy(alpha = 0.05f), RoundedCornerShape(8.dp))
                )
            }

            // Reprocess button
            if (isProcessing) {
                Spacer(modifier = Modifier.height(8.dp))
                Button(
                    onClick = onReprocess,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(
                        imageVector = Icons.Default.Refresh,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Reprocess")
                }
            }
        }
    }
}

@Preview
@Composable
private fun PipelineSettingsPreview() {
    Surface {
        PipelineSettingsScreen(
            onBack = {},
            viewModel = viewModel { PipelineSettingsViewModel(
                matBundle = MockMatBundle()
            ) }
        )
    }
}
