package io.github.iostreamchik.scanner.presenter.composables

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.State
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import io.github.iostreamchik.scanner.presenter.camera.ContourData
import kotlin.math.max

@Composable
fun ContourCanvas(
    contourState: State<ContourData?>,
    modifier: Modifier = Modifier
) {
    val rotatedContours = remember(contourState.value) {
        contourState.value?.let { data ->
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
