package io.github.iostreamchik.scanner.presenter.camera

import android.graphics.Bitmap
import io.github.iostreamchik.scanner.entity.PipelineParams

data class CameraState(
    val intermediateBitmaps: IntermediateBitmaps = IntermediateBitmaps(),
    val originalBitmap: Bitmap? = null,
    val resultBitmap: Bitmap? = null,
    val torchOn: Boolean = false,
    val exposure: String = "",
    val error: String? = null,
    val isProcessing: Boolean = false,
    val pipelineParams: PipelineParams = PipelineParams(),
)

data class IntermediateBitmaps(
    val blur: Bitmap? = null,
    val clahe: Bitmap? = null,
    val morph: Bitmap? = null,
    val edges: Bitmap? = null,
    val mask: Bitmap? = null,
    val corners: Bitmap? = null,
)
