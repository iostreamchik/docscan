package io.github.iostreamchik.scanner.domain.model

import android.graphics.Bitmap
import androidx.compose.runtime.Stable

@Stable
data class IntermediateSnapshots(
    val blur: Bitmap? = null,
    val clahe: Bitmap? = null,
    val morph: Bitmap? = null,
    val edges: Bitmap? = null,
    val mask: Bitmap? = null,
    val corners: Bitmap? = null
)
