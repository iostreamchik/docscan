package io.github.iostreamchik.scanner.presenter.composables

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
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
    contourData: ContourData?,
    modifier: Modifier = Modifier
) {
    val contours = remember(contourData) {
        contourData?.contours?.map { contour ->
            contour.toArray().map { pt -> Offset(pt.x.toFloat(), pt.y.toFloat()) }
        }
    }

    Canvas(modifier = modifier) {
        val contours = contours ?: return@Canvas
        val frameW = contourData?.frameWidth?.toFloat() ?: return@Canvas
        val frameH = contourData?.frameHeight?.toFloat() ?: return@Canvas

        val scale = max(size.width / frameW, size.height / frameH)
        val dx = (size.width - (frameW * scale)) / 2f
        val dy = (size.height - (frameH * scale)) / 2f

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
