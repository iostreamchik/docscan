package io.github.iostreamchik.scanner.local_files

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia.ImageOnly
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
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CloudUpload
import androidx.compose.material.icons.filled.ImageSearch
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.Button
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.material3.ExperimentalMaterial3Api
import kotlinx.coroutines.launch
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.github.iostreamchik.scanner.BitmapCard
import io.github.iostreamchik.scanner.camera.CameraViewModel
import io.github.iostreamchik.scanner.opencv.MockMatBundle
import io.github.iostreamchik.scanner.pipeline.ParameterSlider
import io.github.iostreamchik.scanner.pipeline.debounceFloat
import io.github.iostreamchik.scanner.pipeline.debounceInt
import androidx.compose.runtime.rememberCoroutineScope

/**
 * Displays the original image with contour overlay, filtered preview, and warped result
 * after a file-based scan completes.
 * Also shows configurable pipeline parameters without previews.
 */
@Composable
fun FileScanResultScreen(
    modifier: Modifier = Modifier,
    onBack: () -> Unit,
    viewModel: CameraViewModel,
) {
    val context = LocalContext.current

     // Collect state from ViewModel — all pipeline params come from the flow
    val filteredBitmap by viewModel.filteredBitmap.collectAsStateWithLifecycle()
    val originalBitmap by viewModel.originalBitmap.collectAsStateWithLifecycle()
    val resultBitmap by viewModel.resultBitmap.collectAsStateWithLifecycle()
    val currentParams by viewModel.currentParams.collectAsStateWithLifecycle()
    val coroutineScope = rememberCoroutineScope()

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
            Box(
                modifier = Modifier
                     .fillMaxWidth()
                     .systemBarsPadding()
             ) {
                IconButton(
                    onClick = onBack,
                    modifier = Modifier.align(Alignment.CenterStart)
                 ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Default.ArrowBack,
                        contentDescription = "Back to camera"
                     )
                 }
                Text(
                    text = "Scan Result",
                    fontWeight = FontWeight.Bold,
                    fontSize = 20.sp,
                    modifier = Modifier.align(Alignment.Center)
                 )
             }
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
                 .padding(16.dp),
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
                // Fixed preview images (not scrollable)
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
                        filteredBitmap?.let { bmp ->
                            BitmapCard(
                                bitmap = bmp,
                                shape = RoundedCornerShape(8.dp),
                                modifier = Modifier.fillMaxWidth()
                             )
                         }
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
                        originalBitmap?.let { bmp ->
                            BitmapCard(
                                bitmap = bmp,
                                shape = RoundedCornerShape(8.dp),
                                modifier = Modifier.fillMaxWidth()
                             )
                         }
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
                        resultBitmap?.let { bmp ->
                            BitmapCard(
                                bitmap = bmp,
                                shape = RoundedCornerShape(8.dp),
                                modifier = Modifier.fillMaxWidth()
                             )
                         }
                     }
                 }

                Spacer(modifier = Modifier.height(16.dp))

                 // Scrollable pipeline parameters
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
                        onParamsChange = { newParams ->
                            viewModel.updateParams(newParams)
                            viewModel.reprocessPickedDocument(context)
                        },
                        enableCannyAuto = {
                            coroutineScope.launch {
                                viewModel.enableCannyAuto(context)
                             }
                         },
                        disableCannyAuto = {
                            coroutineScope.launch {
                                viewModel.disableCannyAuto()
                             }
                         },
                    )
                 }
             }
         }
     }
}

@Composable
private fun PipelineParametersSection(
    params: io.github.iostreamchik.scanner.opencv.PipelineParams,
    onParamsChange: (io.github.iostreamchik.scanner.opencv.PipelineParams) -> Unit,
    enableCannyAuto: () -> Unit,
    disableCannyAuto: () -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(12.dp)
     ) {
         // Median Blur
        PipelineStageControls(
            stageName = "Median Blur",
            isExpanded = true
         ) {
            var kernelSize by remember { mutableIntStateOf(params.medianBlurKsize) }
            val debouncedKernel = debounceInt(kernelSize, 300)
            LaunchedEffect(debouncedKernel) {
                if (debouncedKernel != params.medianBlurKsize) {
                    onParamsChange(params.copy(medianBlurKsize = debouncedKernel))
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

         // CLAHE
        PipelineStageControls(
            stageName = "CLAHE",
            isExpanded = true
         ) {
            var clipLimit by remember { mutableFloatStateOf(params.claheClipLimit) }
            var tileSize by remember { mutableIntStateOf(params.claheTileSize) }

            val debouncedClip = debounceFloat(clipLimit, 300)
            val debouncedTile = debounceInt(tileSize, 300)
            LaunchedEffect(debouncedClip, debouncedTile) {
                if (debouncedClip != params.claheClipLimit ||
                    debouncedTile != params.claheTileSize
                 ) {
                    onParamsChange(params.copy(
                        claheClipLimit = debouncedClip,
                        claheTileSize = debouncedTile
                     ))
                 }
             }

            ParameterSlider(
                label = "Clip Limit",
                value = clipLimit,
                valueRange = 0.1f..5.0f,
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
                onValueChange = { tileSize = it.toInt() }
             )
         }

         // Morph Close
        PipelineStageControls(
            stageName = "Morph Close",
            isExpanded = true
         ) {
            var kernelSize by remember { mutableIntStateOf(params.morphCloseSize) }
            val debouncedKernel = debounceInt(kernelSize, 300)
            LaunchedEffect(debouncedKernel) {
                if (debouncedKernel != params.morphCloseSize) {
                    onParamsChange(params.copy(morphCloseSize = debouncedKernel))
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

         // Canny Edges
         PipelineStageControls(
            stageName = "Canny Edges",
            isExpanded = true
         ) {
            var lowThreshold by remember { mutableFloatStateOf(params.cannyLow) }
            var highThreshold by remember { mutableFloatStateOf(params.cannyHigh) }

            // Sync local slider state when params change externally (e.g., enableCannyAuto)
            LaunchedEffect(params.cannyLow, params.cannyHigh) {
                lowThreshold = params.cannyLow
                highThreshold = params.cannyHigh
            }

            val debouncedLow = debounceFloat(lowThreshold, 300)
            val debouncedHigh = debounceFloat(highThreshold, 300)
            LaunchedEffect(debouncedLow, debouncedHigh) {
                if (debouncedLow != params.cannyLow ||
                    debouncedHigh != params.cannyHigh
                 ) {
                    onParamsChange(params.copy(
                        cannyLow = debouncedLow,
                        cannyHigh = debouncedHigh
                     ))
                 }
             }

            // Show auto-threshold values when auto-detect is active
            if (params.cannyAutoDetect) {
                Text(
                    text = "Auto: ${lowThreshold.toInt()} / ${highThreshold.toInt()}",
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

             // Auto-detect switch — wired to new enable/disable methods
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
                            enableCannyAuto()
                         } else {
                            disableCannyAuto()
                         }
                     },
                     enabled = true
                 )
             }
         }

         // Strong Close
        PipelineStageControls(
            stageName = "Strong Close",
            isExpanded = true
         ) {
            var kernelSize by remember { mutableIntStateOf(params.strongCloseSize) }
            val debouncedKernel = debounceInt(kernelSize, 300)
            LaunchedEffect(debouncedKernel) {
                if (debouncedKernel != params.strongCloseSize) {
                    onParamsChange(params.copy(strongCloseSize = debouncedKernel))
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

         // Directional Suppression
        PipelineStageControls(
            stageName = "Directional Suppression",
            isExpanded = true
         ) {
            var kernelSize by remember { mutableIntStateOf(params.directionalKernelSize) }

            val debouncedKernel = debounceInt(kernelSize, 300)
            LaunchedEffect(debouncedKernel) {
                if (debouncedKernel != params.directionalKernelSize) {
                    onParamsChange(params.copy(
                        directionalKernelSize = debouncedKernel
                     ))
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
     }
}

@Composable
private fun PipelineStageControls(
    stageName: String,
    isExpanded: Boolean,
    content: @Composable () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        elevation = CardDefaults.cardElevation(2.dp)
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
                    matBundle = MockMatBundle()
                 )
             },
            onBack = {}
         )
     }
}
