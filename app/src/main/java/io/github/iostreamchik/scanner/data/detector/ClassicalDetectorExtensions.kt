package io.github.iostreamchik.scanner.data.detector

import android.graphics.Bitmap
import io.github.iostreamchik.scanner.data.opencv.IMatBundle
import io.github.iostreamchik.scanner.data.utils.fixRotation
import io.github.iostreamchik.scanner.data.utils.toBitmap
import io.github.iostreamchik.scanner.entity.IntermediateBitmaps

/**
 * Captures intermediate bitmap snapshots from a classical detector's mat bundle.
 * Shared by DocumentDetectorMinimal and DocumentDetectorDirectionalSuppression.
 */
fun IMatBundle.captureClassicalSnapshots(rotation: Int): IntermediateBitmaps {
    val toBitmap = { mat: org.opencv.core.Mat ->
        mat.fixRotation(rotation).toBitmap()
            .copy(Bitmap.Config.ARGB_8888, false)
    }
    return IntermediateBitmaps(
        blur = toBitmap(this.getBlurred()),
        clahe = toBitmap(this.getEnhanced()),
        morph = toBitmap(this.getMorph()),
        edges = toBitmap(this.getEdges())
    )
}
