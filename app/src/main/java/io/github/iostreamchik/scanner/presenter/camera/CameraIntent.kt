package io.github.iostreamchik.scanner.presenter.camera

import androidx.annotation.StringRes
import android.content.Context
import android.net.Uri
import io.github.iostreamchik.scanner.entity.PipelineParams

sealed class CameraIntent {
    object ToggleTorch : CameraIntent()
    data class SetTorch(val on: Boolean) : CameraIntent()
    data class SetError(@StringRes val messageId: Int?) : CameraIntent()
    data class UpdateParams(val params: PipelineParams) : CameraIntent()
    data class ProcessDocument(val context: Context, val uri: Uri, val onComplete: () -> Unit) : CameraIntent()
}
