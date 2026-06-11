package io.github.iostreamchik.scanner.local_files

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia.ImageOnly
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.ImageSearch
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.github.iostreamchik.scanner.BitmapCard
import io.github.iostreamchik.scanner.camera.CameraViewModel
import io.github.iostreamchik.scanner.opencv.MockMatBundle

/**
 * Displays the original image with contour overlay, filtered preview, and warped result
 * after a file-based scan completes.
 */
@Composable
fun FileScanResultScreen(
    modifier: Modifier = Modifier,
    onBack: () -> Unit,
    viewModel: CameraViewModel,
) {
    val context = LocalContext.current

    // Collect state from ViewModel
    val filteredBitmap by viewModel.filteredBitmap.collectAsStateWithLifecycle()
    val originalBitmap by viewModel.originalBitmap.collectAsStateWithLifecycle()
    val resultBitmap by viewModel.resultBitmap.collectAsStateWithLifecycle()

    val pickMediaLauncher = rememberLauncherForActivityResult(
        contract = PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            viewModel.processPickedDocument(context, uri) {}
        }
    }

    // Auto-show file picker when the screen opens for the first time
    LaunchedEffect(Unit) {
        pickMediaLauncher.launch(PickVisualMediaRequest(ImageOnly))
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
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Column(
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
                        )
                    }
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.Top
            ) {
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
        }
    }
}

@Preview
@Composable
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
