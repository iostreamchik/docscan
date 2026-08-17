package io.github.iostreamchik.scanner.presenter.composables

import android.graphics.Bitmap
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.core.graphics.createBitmap

@Composable
fun BitmapCard(
    modifier: Modifier = Modifier,
    bitmap: Bitmap?,
    animated: Boolean = false
) {
    val placeholder = remember { createBitmap(1, 1).asImageBitmap() }
    val imageBitmap = bitmap?.asImageBitmap() ?: placeholder

    Box(modifier = modifier) {
        if (animated) {
            val width = remember { mutableStateOf(0.dp) }
            val density = LocalDensity.current
            AnimatedContent(
                targetState = bitmap,
                transitionSpec = { fadeIn() togetherWith fadeOut() }
            ) { bmp ->
                if (bmp != null) {
                    Image(
                        modifier = Modifier
                            .fillMaxSize()
                            .onGloballyPositioned {
                                if (width.value == 0.dp)
                                    width.value = with(density) { it.size.width.toDp() }
                            },
                        bitmap = bmp.asImageBitmap(),
                        contentDescription = null,
                        contentScale = ContentScale.Crop,
                    )
                } else {
                    Box(modifier = Modifier
                        .width(width.value)
                        .fillMaxSize())
                }
            }
        } else {
            Image(
                modifier = Modifier.fillMaxSize(),
                bitmap = imageBitmap,
                contentDescription = null,
                contentScale = ContentScale.Crop,
            )
        }
    }
}
