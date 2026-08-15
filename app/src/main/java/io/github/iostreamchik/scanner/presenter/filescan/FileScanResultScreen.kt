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
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.key
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
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
import io.github.iostreamchik.scanner.data.repository.DocumentDetectorRepositoryImpl
import io.github.iostreamchik.scanner.presenter.theme.DocumentScannerTheme

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FileScanResultScreen(
    modifier: Modifier = Modifier,
    onBack: () -> Unit,
    viewModel: CameraViewModel,
) {
    val context = LocalContext.current
    val uiState by viewModel.state.collectAsStateWithLifecycle()
    val detectionParamsState = viewModel.detectionParams.collectAsStateWithLifecycle()

    var selectedItem by remember { mutableStateOf<Pair<String, Bitmap>?>(null) }
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    val pickMediaLauncher = rememberLauncherForActivityResult(
        contract = PickVisualMedia()
    ) { uri ->
        uri?.let { viewModel.process(CameraIntent.ProcessDocument(context, it) {}) }
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
                    text = when (name) {
                        "blur" -> stringResource(io.github.iostreamchik.scanner.R.string.step_blur)
                        "clahe" -> stringResource(io.github.iostreamchik.scanner.R.string.step_clahe)
                        "morph" -> stringResource(io.github.iostreamchik.scanner.R.string.step_morph)
                        "edges" -> stringResource(io.github.iostreamchik.scanner.R.string.step_edges)
                        "mask" -> stringResource(io.github.iostreamchik.scanner.R.string.step_mask)
                        "corners" -> stringResource(io.github.iostreamchik.scanner.R.string.step_corners)
                        "original" -> stringResource(io.github.iostreamchik.scanner.R.string.file_scan_original)
                        "detected" -> stringResource(io.github.iostreamchik.scanner.R.string.file_scan_detected)
                        else -> name.replaceFirstChar { it.uppercase() }
                    },
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold
                )
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp)
                        .aspectRatio(bmp.width.toFloat() / bmp.height.toFloat())
                ) {
                    BitmapCard(
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(RoundedCornerShape(8.dp)),
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
                    Column {
                        Text(
                            text = stringResource(io.github.iostreamchik.scanner.R.string.file_scan_title),
                            fontWeight = FontWeight.Bold,
                            fontSize = 20.sp
                        )
                        Text(
                            text = stringResource(
                                io.github.iostreamchik.scanner.R.string.file_scan_detector,
                                detectionParamsState.value.detectorName
                            ),
                            fontSize = 12.sp,
                            lineHeight = 12.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Default.ArrowBack,
                            contentDescription = stringResource(io.github.iostreamchik.scanner.R.string.content_description_back)
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
                    contentDescription = stringResource(io.github.iostreamchik.scanner.R.string.content_description_scan_from_file)
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
            uiState.errorId?.let { id ->
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 16.dp)
                ) {
                    Text(
                        text = stringResource(id),
                        color = Color.Red,
                        modifier = Modifier
                            .background(Color.Red.copy(alpha = 0.1f), RoundedCornerShape(8.dp))
                            .padding(12.dp)
                    )
                }
            }

            val hasImage by remember { derivedStateOf { uiState.originalBitmap != null } }
            val imageAspectRatio by remember {
                derivedStateOf {
                    uiState.originalBitmap?.let { it.width.toFloat() / it.height.toFloat() }
                }
            }
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
                            text = if (uiState.isProcessing)
                                stringResource(io.github.iostreamchik.scanner.R.string.file_scan_processing)
                            else
                                stringResource(io.github.iostreamchik.scanner.R.string.file_scan_select_file),
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
                            Text(stringResource(io.github.iostreamchik.scanner.R.string.file_scan_choose_file))
                        }
                    }
                } else {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        val intermediates = remember(uiState.intermediateBitmaps) {
                            val hasOverlay = uiState.intermediateBitmaps.corners != null ||
                                    uiState.intermediateBitmaps.mask != null
                            buildList {
                                uiState.intermediateBitmaps.blur?.let { add("blur" to it) }
                                uiState.intermediateBitmaps.clahe?.let { add("clahe" to it) }
                                uiState.intermediateBitmaps.morph?.let { add("morph" to it) }
                                if (!hasOverlay) {
                                    uiState.intermediateBitmaps.edges?.let { add("edges" to it) }
                                }
                                uiState.intermediateBitmaps.mask?.let { add("mask" to it) }
                                uiState.intermediateBitmaps.corners?.let { add("corners" to it) }
                            }
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
                                                text = when (item?.first) {
                                                    "blur" -> stringResource(io.github.iostreamchik.scanner.R.string.step_blur)
                                                    "clahe" -> stringResource(io.github.iostreamchik.scanner.R.string.step_clahe)
                                                    "morph" -> stringResource(io.github.iostreamchik.scanner.R.string.step_morph)
                                                    "edges" -> stringResource(io.github.iostreamchik.scanner.R.string.step_edges)
                                                    "mask" -> stringResource(io.github.iostreamchik.scanner.R.string.step_mask)
                                                    "corners" -> stringResource(io.github.iostreamchik.scanner.R.string.step_corners)
                                                    else -> ""
                                                },
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
                                                            Modifier.clickable(onClick = {
                                                                selectedItem = item
                                                            })
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
                                        text = stringResource(io.github.iostreamchik.scanner.R.string.file_scan_original),
                                        fontSize = 14.sp,
                                        fontWeight = FontWeight.Medium,
                                    )
                                    Box(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .aspectRatio(cardAspectRatio)
                                            .clip(RoundedCornerShape(8.dp))
                                            .clickable {
                                                uiState.originalBitmap?.let {
                                                    selectedItem = "original" to it
                                                }
                                            }
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
                                        text = stringResource(io.github.iostreamchik.scanner.R.string.file_scan_detected),
                                        fontSize = 14.sp,
                                        fontWeight = FontWeight.Medium,
                                    )
                                    val resultAspectRatio =
                                        uiState.resultBitmap?.let { it.width.toFloat() / it.height.toFloat() }
                                    Box(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .aspectRatio(resultAspectRatio ?: cardAspectRatio)
                                            .clip(RoundedCornerShape(8.dp))
                                            .clickable {
                                                uiState.resultBitmap?.let {
                                                    selectedItem = "detected" to it
                                                }
                                            }
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
    DocumentScannerTheme() {
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
}
