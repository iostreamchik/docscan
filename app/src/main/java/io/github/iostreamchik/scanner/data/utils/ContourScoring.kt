package io.github.iostreamchik.scanner.data.utils

import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.geometry.Geometry
import kotlin.math.abs
import kotlin.math.hypot
import io.github.iostreamchik.scanner.entity.PipelineParams

fun scoreContourWithParams(
    contour: MatOfPoint,
    width: Int,
    height: Int,
    params: PipelineParams
): Double {
    val area = abs(Geometry.contourArea(contour))

    val center = Geometry.boundingRect(contour).let {
        Point(it.x + it.width / 2.0, it.y + it.height / 2.0)
    }

    val frameCenter = Point(width / 2.0, height / 2.0)
    val centerDist = hypot(
        center.x - frameCenter.x,
        center.y - frameCenter.y
    )

    val maxDist = hypot(width / 2.0, height / 2.0)
    val centerScore = 1.0 - (centerDist / maxDist)

    val frameArea = width * height.toDouble()
    val areaRatio = area / frameArea
    val areaRatioScore = when {
        areaRatio <= 0.02 -> 1.0
        areaRatio >= 0.5 -> 0.2
        else -> 1.0 - ((areaRatio - 0.02) / 0.48) * 0.8
    }

    return area * params.scoreAreaWeight +
        centerScore * params.scoreCenterWeight * width * height +
        areaRatioScore * params.scoreAreaRatioWeight * width * height
}
