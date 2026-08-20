package io.github.iostreamchik.scanner.presenter.camera

import androidx.annotation.StringRes
import android.graphics.Bitmap
import androidx.compose.runtime.Stable
import io.github.iostreamchik.scanner.entity.IntermediateBitmaps
import io.github.iostreamchik.scanner.entity.PipelineParams

@Stable
data class CameraState(
    val intermediateBitmaps: IntermediateBitmaps = IntermediateBitmaps(),
    val originalBitmap: Bitmap? = null,
    val resultBitmap: Bitmap? = null,
    val torchOn: Boolean = false,
    val exposure: String = "",
    @StringRes val errorId: Int? = null,
    val isProcessing: Boolean = false,
    val pipelineParams: PipelineParams = PipelineParams(),
    val documentDetected: Boolean = false,
)

