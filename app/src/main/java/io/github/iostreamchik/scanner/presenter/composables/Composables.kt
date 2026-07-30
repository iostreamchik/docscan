package io.github.iostreamchik.scanner.presenter.composables

import android.graphics.Bitmap
import android.os.Build
import android.view.RoundedCorner
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.Composable
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.core.graphics.createBitmap
import io.github.iostreamchik.scanner.presenter.camera.ContourData
import kotlin.math.max

@Composable
fun rememberDeviceCornerRadiusDp(
    defaultValue: Dp = 24.dp
): Dp {
    val view = LocalView.current
    val density = LocalDensity.current

    return remember {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val insets = view.rootWindowInsets
            val corner = insets?.getRoundedCorner(RoundedCorner.POSITION_TOP_LEFT)
            corner?.radius?.let { radiusPx ->
                with(density) { radiusPx.toDp() }
            } ?: defaultValue
        } else {
            defaultValue
        }
    }
}

@Composable
fun BitmapCard(
    modifier: Modifier = Modifier,
    bitmap: Bitmap?,
    animated: Boolean = false
) {
    Box(modifier = modifier) {
        if (animated) {
            val width = remember { mutableStateOf(0.dp) }
            val density = LocalDensity.current
            AnimatedContent(
                targetState = bitmap,
                transitionSpec = {fadeIn() togetherWith fadeOut()}
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
                bitmap = bitmap?.asImageBitmap() ?: createBitmap(1, 1).asImageBitmap(),
                contentDescription = null,
                contentScale = ContentScale.Crop,
            )
        }
    }
}

@Composable
fun ContourCanvas(
    contourState: State<ContourData?>,
    modifier: Modifier = Modifier
) {
    // Pre-calculate rotated points and frame dimensions to avoid heavy math in DrawScope
    val rotatedContours = remember(contourState.value) {
        val data = contourState.value ?: return@remember null
        val originalW = data.frameWidth.toFloat()
        val originalH = data.frameHeight.toFloat()

        val rotatedW = if (data.rotation == 90 || data.rotation == 270) originalH else originalW
        val rotatedH = if (data.rotation == 90 || data.rotation == 270) originalW else originalH

        val contours = data.contours.map { contour ->
            val points = contour.toArray()
            points.map { pt ->
                when (data.rotation) {
                    90 -> Offset(originalH - pt.y.toFloat(), pt.x.toFloat())
                    180 -> Offset(originalW - pt.x.toFloat(), originalH - pt.y.toFloat())
                    270 -> Offset(pt.y.toFloat(), originalW - pt.x.toFloat())
                    else -> Offset(pt.x.toFloat(), pt.y.toFloat())
                }
            }
        }

        Triple(contours, rotatedW, rotatedH)
    }

    Canvas(modifier = modifier) {
        val contours = rotatedContours?.first ?: return@Canvas
        val rotatedW = rotatedContours?.second ?: return@Canvas
        val rotatedH = rotatedContours?.third ?: return@Canvas

        val scale = max(size.width / rotatedW, size.height / rotatedH)
        val dx = (size.width - (rotatedW * scale)) / 2f
        val dy = (size.height - (rotatedH * scale)) / 2f

        contours.forEach { points ->
            val scaledPoints = points.map { Offset(it.x * scale + dx, it.y * scale + dy) }

            if (scaledPoints.size >= 3) {
                val path = Path().apply {
                    moveTo(scaledPoints[0].x, scaledPoints[0].y)
                    for (i in 1 until scaledPoints.size) {
                        lineTo(scaledPoints[i].x, scaledPoints[i].y)
                    }
                    close()
                }

                drawPath(
                    path = path,
                    color = Color.Green.copy(alpha = 0.25f)
                )

                drawPath(
                    path = path,
                    color = Color.Green.copy(alpha = 0.9f),
                    style = Stroke(width = 4f)
                )
            }
        }
    }
}
