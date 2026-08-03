package io.github.iostreamchik.scanner.presenter.filescan

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia.ImageOnly
import android.graphics.Bitmap
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.ImageSearch
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.key
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
import io.github.iostreamchik.scanner.presenter.composables.BitmapCard
import io.github.iostreamchik.scanner.presenter.camera.CameraIntent
import io.github.iostreamchik.scanner.presenter.camera.CameraViewModel
import io.github.iostreamchik.scanner.data.detector.MockDocumentDetector
import io.github.iostreamchik.scanner.data.opencv.MockMatBundle
import io.github.iostreamchik.scanner.data.repository.DocumentDetectorRepositoryImpl

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FileScanResultScreen(
    modifier: Modifier = Modifier,
    onBack: () -> Unit,
    viewModel: CameraViewModel,
) {
    val context = LocalContext.current
    val uiState by viewModel.state.collectAsStateWithLifecycle()
    val detectionParams by viewModel.detectionParams.collectAsStateWithLifecycle()

    var selectedItem by remember { mutableStateOf<Pair<String, Bitmap>?>(null) }
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    val pickMediaLauncher = rememberLauncherForActivityResult(
        contract = PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            viewModel.process(CameraIntent.ProcessDocument(context, uri) {})
        }
    }

    selectedItem?.let { (name, bmp) ->
        ModalBottomSheet(
            onDismissRequest = { selectedItem = null },
            sheetState = sheetState,
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = name.replaceFirstChar { it.uppercase() },
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold
                )
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp)
                ) {
                    BitmapCard(
                        modifier = Modifier.fillMaxSize(),
                        bitmap = bmp,
                        animated = false
                    )
                }
            }
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
                shape = CircleShape,
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
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            uiState.error?.let { msg ->
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

            val hasImage = uiState.originalBitmap != null
            val imageAspectRatio = uiState.originalBitmap?.let { it.width.toFloat() / it.height.toFloat() }
            key(hasImage) {
                if (!hasImage) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .fillMaxWidth(),
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
                            text = if (uiState.isProcessing) "Processing..." else "Select a file",
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
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        val hasOverlay = uiState.intermediateBitmaps.corners != null ||
                            uiState.intermediateBitmaps.mask != null
                        val intermediates = buildList {
                            uiState.intermediateBitmaps.blur?.let { add("blur" to it) }
                            uiState.intermediateBitmaps.clahe?.let { add("clahe" to it) }
                            uiState.intermediateBitmaps.morph?.let { add("morph" to it) }
                            if (!hasOverlay) {
                                uiState.intermediateBitmaps.edges?.let { add("edges" to it) }
                            }
                            uiState.intermediateBitmaps.mask?.let { add("mask" to it) }
                            uiState.intermediateBitmaps.corners?.let { add("corners" to it) }
                        }

                        val cardAspectRatio = imageAspectRatio ?: 1f
                        intermediates.chunked(2).forEach { row ->
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                repeat(2) { index ->
                                    val item = row.getOrNull(index)
                                    key(item?.first ?: "empty-$index") {
                                        Column(
                                            modifier = Modifier.weight(1f),
                                            horizontalAlignment = Alignment.CenterHorizontally,
                                            verticalArrangement = Arrangement.spacedBy(4.dp)
                                        ) {
                                            Text(
                                                text = item?.first?.replaceFirstChar { it.uppercase() } ?: "",
                                                fontSize = 14.sp,
                                                fontWeight = FontWeight.Medium,
                                            )
                                            Box(
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .aspectRatio(cardAspectRatio)
                                                    .clip(RoundedCornerShape(8.dp))
                                                    .then(
                                                        if (item != null) {
                                                            Modifier.clickable(
                                                                interactionSource = remember { MutableInteractionSource() },
                                                                indication = null,
                                                                onClick = { selectedItem = item }
                                                            )
                                                        } else {
                                                            Modifier
                                                        }
                                                    )
                                            ) {
                                                AnimatedContent(
                                                    targetState = item?.second,
                                                    transitionSpec = {
                                                        (fadeIn() + expandVertically()) togetherWith (fadeOut() + shrinkVertically())
                                                    }
                                                ) { bmp: Bitmap? ->
                                                    if (bmp != null) {
                                                        BitmapCard(
                                                            modifier = Modifier.fillMaxSize(),
                                                            bitmap = bmp,
                                                            animated = false
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            key("original") {
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
                                    Box(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .aspectRatio(cardAspectRatio)
                                            .clip(RoundedCornerShape(8.dp))
                                            .clickable(
                                                interactionSource = remember { MutableInteractionSource() },
                                                indication = null,
                                                onClick = { uiState.originalBitmap?.let { selectedItem = "original" to it } }
                                            )
                                    ) {
                                        AnimatedContent(
                                            targetState = uiState.originalBitmap,
                                            transitionSpec = {
                                                (fadeIn() + expandVertically()) togetherWith (fadeOut() + shrinkVertically())
                                            }
                                        ) { bmp: Bitmap? ->
                                            if (bmp != null) {
                                                BitmapCard(
                                                    modifier = Modifier.fillMaxSize(),
                                                    bitmap = bmp,
                                                    animated = false
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                            key("result") {
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
                                    Box(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .aspectRatio(cardAspectRatio)
                                            .clip(RoundedCornerShape(8.dp))
                                            .clickable(
                                                interactionSource = remember { MutableInteractionSource() },
                                                indication = null,
                                                onClick = { uiState.resultBitmap?.let { selectedItem = "detected" to it } }
                                            )
                                    ) {
                                        AnimatedContent(
                                            targetState = uiState.resultBitmap,
                                            transitionSpec = {
                                                (fadeIn() + expandVertically()) togetherWith (fadeOut() + shrinkVertically())
                                            }
                                        ) { bmp: Bitmap? ->
                                            if (bmp != null) {
                                                BitmapCard(
                                                    modifier = Modifier.fillMaxSize(),
                                                    bitmap = bmp,
                                                    animated = false
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        AssistChip(
                            onClick = { },
                            label = { Text("Detector: ${detectionParams.detectorName}") },
                            leadingIcon = {
                                Icon(
                                    imageVector = Icons.Default.ImageSearch,
                                    contentDescription = null,
                                    modifier = Modifier.size(AssistChipDefaults.IconSize)
                                )
                            }
                        )
                        Spacer(modifier = Modifier.height(84.dp))
                    }
                }
            }
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
                    repository = DocumentDetectorRepositoryImpl(MockDocumentDetector())
                )
            },
            onBack = {}
        )
    }
}
