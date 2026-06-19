package io.github.iostreamchik.scanner

import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import kotlin.math.hypot
import io.github.iostreamchik.scanner.opencv.PipelineParams
import io.github.iostreamchik.scanner.opencv.*

/**
 * Scores a detected contour for document suitability.
 *
 * Weights:
 * - Area (0.5): larger contours are more likely to be documents
 * - Center proximity (0.3): documents tend to be near the frame center
 * - Area ratio (0.2): penalizes contours that fill the entire frame (likely background)
 */
fun scoreContour(
    contour: MatOfPoint,
    width: Int,
    height: Int
): Double {
    val area = org.opencv.imgproc.Imgproc.contourArea(contour)

    val center = org.opencv.imgproc.Imgproc.boundingRect(contour).let {
        Point(it.x + it.width / 2.0, it.y + it.height / 2.0)
    }

    val frameCenter = Point(width / 2.0, height / 2.0)
    val centerDist = hypot(
        center.x - frameCenter.x,
        center.y - frameCenter.y
    )

    val maxDist = hypot(width / 2.0, height / 2.0)

    val centerScore = 1.0 - (centerDist / maxDist)

    // Area ratio: penalize if contour fills too much of the frame
    // (likely background texture rather than a document)
    val frameArea = width * height.toDouble()
    val areaRatio = area / frameArea
    val areaRatioScore = if (areaRatio > 0.5) 0.2 else 1.0

    return area * 0.5 + centerScore * 0.3 * width * height + areaRatioScore * 0.2 * width * height
}

/**
 * Scores a detected contour using configurable scoring weights from [PipelineParams].
 *
 * Unlike [scoreContour], this uses smooth interpolation for the area ratio score
 * and configurable weights.
 */
fun scoreContourWithParams(
    contour: MatOfPoint,
    width: Int,
    height: Int,
    params: PipelineParams
): Double {
    val area = org.opencv.imgproc.Imgproc.contourArea(contour)

    val center = org.opencv.imgproc.Imgproc.boundingRect(contour).let {
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
    // Smooth interpolation: 1.0 at areaRatio <= 0.02, linearly down to 0.2 at areaRatio >= 0.5
    val areaRatioScore = when {
        areaRatio <= 0.02f -> 1.0
        areaRatio >= 0.5  -> 0.2
        else               -> 1.0 - ((areaRatio - 0.02) / 0.48) * 0.8
    }

    return area * params.scoreAreaWeight +
        centerScore * params.scoreCenterWeight * width * height +
        areaRatioScore * params.scoreAreaRatioWeight * width * height
}
