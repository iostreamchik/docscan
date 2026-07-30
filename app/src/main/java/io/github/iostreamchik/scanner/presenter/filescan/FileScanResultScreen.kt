package io.github.iostreamchik.scanner.presenter.filescan

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia.ImageOnly
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
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
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

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FileScanResultScreen(
    modifier: Modifier = Modifier,
    onBack: () -> Unit,
    viewModel: CameraViewModel,
) {
    val context = LocalContext.current
    val uiState by viewModel.state.collectAsStateWithLifecycle()

    val pickMediaLauncher = rememberLauncherForActivityResult(
        contract = PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            viewModel.process(CameraIntent.ProcessDocument(context, uri) {})
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
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
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
                                        .fillMaxWidth()
                                        .aspectRatio(imageAspectRatio ?: 1f)
                                        .clip(RoundedCornerShape(8.dp)),
                                    bitmap = uiState.intermediateBitmaps.blur,
                                    animated = true
                                )
                            }
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
                                        .fillMaxWidth()
                                        .aspectRatio(imageAspectRatio ?: 1f)
                                        .clip(RoundedCornerShape(8.dp)),
                                    bitmap = uiState.intermediateBitmaps.clahe,
                                    animated = true
                                )
                            }
                        }

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
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
                                        .fillMaxWidth()
                                        .aspectRatio(imageAspectRatio ?: 1f)
                                        .clip(RoundedCornerShape(8.dp)),
                                    bitmap = uiState.intermediateBitmaps.morph,
                                    animated = true
                                )
                            }
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
                                        .fillMaxWidth()
                                        .aspectRatio(imageAspectRatio ?: 1f)
                                        .clip(RoundedCornerShape(8.dp)),
                                    bitmap = uiState.intermediateBitmaps.edges,
                                    animated = true
                                )
                            }
                        }

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
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
                                        .fillMaxWidth()
                                        .aspectRatio(imageAspectRatio ?: 1f)
                                        .clip(RoundedCornerShape(8.dp)),
                                    bitmap = uiState.originalBitmap,
                                    animated = true
                                )
                            }
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
                                        .fillMaxWidth()
                                        .aspectRatio(imageAspectRatio ?: 1f)
                                        .clip(RoundedCornerShape(8.dp)),
                                    bitmap = uiState.resultBitmap,
                                    animated = true
                                )
                            }
                        }
                        AssistChip(
                            onClick = { },
                            label = { Text("Detector: ${viewModel.detector.detectorName}") },
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
                    matBundle = MockMatBundle(),
                    detector = MockDocumentDetector()
                )
            },
            onBack = {}
        )
    }
}
