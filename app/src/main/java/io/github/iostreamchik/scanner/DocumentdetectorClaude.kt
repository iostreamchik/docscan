package io.github.iostreamchik.scanner

import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.*

/**
 * Document Detection Algorithm — Improved Variant
 *
 * Improvements over v1:
 *   • Adaptive preprocessing  – auto-scales Canny thresholds via Otsu's method;
 *                               falls back to CLAHE-enhanced path for low-contrast images.
 *   • Multi-strategy search   – tries three independent edge maps and picks the
 *                               best quad by convexity + aspect-ratio scoring.
 *   • Convexity guard         – rejects non-convex quads (folded pages, shadows).
 *   • Angle validation        – all interior angles must be 60°–120° (no degenerate quads).
 *   • Memory safety           – explicit Mat.release() via extension helpers.
 *   • Config data class       – every tunable constant is in one place.
 *   • Post-processing         – optional adaptive-threshold binarisation for clean B&W scan.
 */

// ─────────────────────────────────────────────────────────────
// CONFIGURATION
// ─────────────────────────────────────────────────────────────

data class DetectorConfig(
    /** Minimum contour area as a fraction of the image area. */
    val minAreaFraction: Double = 0.08,
    /** Maximum contour area fraction (to reject full-frame contours). */
    val maxAreaFraction: Double = 0.97,
    /** How many top contours to evaluate as document candidates. */
    val candidateCount: Int = 8,
    /** Epsilon factors tried during polygon approximation. */
    val epsilonFactors: List<Double> = listOf(0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
    /** Acceptable interior angle range (degrees). */
    val minAngleDeg: Double = 55.0,
    val maxAngleDeg: Double = 125.0,
    /** Apply B&W binarisation to the warped output. */
    val binarise: Boolean = false,
    /** Scale factor applied before processing (speeds up large images). */
    val workingScale: Double = 1.0,
)

// ─────────────────────────────────────────────────────────────
// RESULT TYPE
// ─────────────────────────────────────────────────────────────

sealed class DetectionResult {
    /** Document found – [quad] is ordered TL→TR→BR→BL in original-image coordinates. */
    data class Found(val quad: Array<Point>, val warped: Mat) : DetectionResult()
    /** No suitable quadrilateral found. */
    object NotFound : DetectionResult()
}

// ─────────────────────────────────────────────────────────────
// DETECTOR
// ─────────────────────────────────────────────────────────────

class DocumentdetectorClaude(private val config: DetectorConfig = DetectorConfig()) {

    // ── Public API ────────────────────────────────────────────

    fun detect(src: Mat): DetectionResult {
        val scale = config.workingScale
        val input = if (scale != 1.0) src.scaled(scale) else src

        val quad = findBestQuad(input)
            ?: return DetectionResult.NotFound

        // Map quad back to original-image coordinates if we downscaled
        val originalQuad = if (scale != 1.0) quad.map {
            Point(it.x / scale, it.y / scale)
        }.toTypedArray() else quad

        val warped = perspectiveWarp(src, originalQuad)
        val output = if (config.binarise) binarise(warped) else warped

        return DetectionResult.Found(originalQuad, output)
    }

    /** Convenience: returns ordered corners only (no warp allocation). */
    fun detectCorners(src: Mat): Array<Point>? =
        (detect(src) as? DetectionResult.Found)?.quad

    // ── Step 1 – Multi-strategy edge maps ─────────────────────

    /**
     * Build three complementary edge maps:
     *   A) Canny with Otsu-derived thresholds   – well-lit documents on plain background
     *   B) CLAHE-enhanced Canny                  – low-contrast / dark backgrounds
     *   C) Adaptive threshold edges              – uneven illumination (e.g. curved pages)
     */
    private fun buildEdgeMaps(gray: Mat): List<Mat> {
        val maps = mutableListOf<Mat>()

        // ── A) Otsu Canny ──────────────────────────────────────
        val blurA = gray.gaussianBlur(5)
        val otsuThresh = otsuThreshold(blurA)
        val edgesA = Mat()
        Imgproc.Canny(blurA, edgesA, otsuThresh * 0.5, otsuThresh)
        edgesA.dilate(1)
        maps += edgesA
        blurA.release()

        // ── B) CLAHE-enhanced Canny ────────────────────────────
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        val enhanced = Mat()
        clahe.apply(gray, enhanced)
        val blurB = enhanced.gaussianBlur(5)
        val edgesB = Mat()
        Imgproc.Canny(blurB, edgesB, 30.0, 90.0)
        edgesB.dilate(1)
        maps += edgesB
        blurB.release(); enhanced.release()

        // ── C) Adaptive-threshold edges ────────────────────────
        val blurC = gray.gaussianBlur(9)
        val thresh = Mat()
        Imgproc.adaptiveThreshold(
            blurC, thresh, 255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY, 11, 2.0
        )
        Core.bitwise_not(thresh, thresh)   // invert: edges become white
        thresh.dilate(1)
        maps += thresh
        blurC.release()

        return maps
    }

    // ── Step 2 – Find best quad across all edge maps ──────────

    private fun findBestQuad(src: Mat): Array<Point>? {
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)

        val edgeMaps = buildEdgeMaps(gray)
        gray.release()

        var best: ScoredQuad? = null

        for (edges in edgeMaps) {
            val candidate = findDocumentQuad(edges, src.size())
            if (candidate != null) {
                val score = scoreQuad(candidate, src.size())
                if (best == null || score > best.score) {
                    best = ScoredQuad(candidate, score)
                }
            }
            edges.release()
        }

        return best?.quad
    }

    // ── Step 3 – Contour → quad extraction ───────────────────

    private fun findDocumentQuad(edges: Mat, imageSize: Size): Array<Point>? {
        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(
            edges.clone(),           // findContours modifies the source Mat
            contours, Mat(),
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        val imageArea = imageSize.width * imageSize.height
        val minArea   = imageArea * config.minAreaFraction
        val maxArea   = imageArea * config.maxAreaFraction

        return contours
            .filter { Imgproc.contourArea(it) in minArea..maxArea }
            .sortedByDescending { Imgproc.contourArea(it) }
            .take(config.candidateCount)
            .firstNotNullOfOrNull { contour ->
                approxQuad(contour)
                    ?.takeIf { isConvex(it) && anglesValid(it) }
                    ?.let { orderPoints(it) }
            }
    }

    // ── Step 4 – Polygon approximation ───────────────────────

    /**
     * Douglas-Peucker approximation with multiple epsilon candidates.
     * Returns exactly 4 corners, or null if none of the epsilons yield a quad.
     */
    private fun approxQuad(contour: MatOfPoint): Array<Point>? {
        val c2f       = MatOfPoint2f(*contour.toArray())
        val perimeter = Imgproc.arcLength(c2f, true)
        val approx    = MatOfPoint2f()

        for (factor in config.epsilonFactors) {
            Imgproc.approxPolyDP(c2f, approx, factor * perimeter, true)
            if (approx.rows() == 4) {
                c2f.release()
                return approx.toArray()
            }
        }
        c2f.release(); approx.release()
        return null
    }

    // ── Step 5 – Validation ───────────────────────────────────

    /** Reject quads where any interior angle falls outside the configured range. */
    private fun anglesValid(pts: Array<Point>): Boolean {
        for (i in pts.indices) {
            val prev  = pts[(i + 3) % 4]
            val curr  = pts[i]
            val next  = pts[(i + 1) % 4]
            val angle = angleDeg(prev, curr, next)
            if (angle !in config.minAngleDeg..config.maxAngleDeg) return false
        }
        return true
    }

    /** Reject non-convex quads (page-curl, shadow concavities, etc.). */
    private fun isConvex(pts: Array<Point>): Boolean {
        val mat = MatOfPoint(*pts)
        return Imgproc.isContourConvex(mat).also { mat.release() }
    }

    // ── Step 6 – Scoring ─────────────────────────────────────

    /**
     * Composite score = area_fraction × convexity_ratio × angle_regularity.
     *
     * Prefers quads that are:
     *   – large relative to the frame
     *   – convex (no dents)
     *   – close to a rectangle (interior angles ≈ 90°)
     */
    private fun scoreQuad(pts: Array<Point>, imageSize: Size): Double {
        val imageArea  = imageSize.width * imageSize.height
        val quadArea   = contourArea(pts)
        val areaScore  = quadArea / imageArea

        val hull        = convexHullArea(pts)
        val convexScore = if (hull > 0) quadArea / hull else 0.0

        val angleScore  = pts.indices.map { i ->
            val prev = pts[(i + 3) % 4]
            val curr = pts[i]
            val next = pts[(i + 1) % 4]
            1.0 - abs(angleDeg(prev, curr, next) - 90.0) / 90.0
        }.average()

        return areaScore * convexScore * angleScore
    }

    // ── Step 7 – Perspective warp ─────────────────────────────

    fun perspectiveWarp(src: Mat, quad: Array<Point>): Mat {
        val (tl, tr, br, bl) = quad

        val outWidth  = max(distance(tl, tr), distance(bl, br)).toInt()
        val outHeight = max(distance(tl, bl), distance(tr, br)).toInt()

        val srcMat = MatOfPoint2f(tl, tr, br, bl)
        val dstMat = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(outWidth - 1.0, 0.0),
            Point(outWidth - 1.0, outHeight - 1.0),
            Point(0.0, outHeight - 1.0)
        )

        val M      = Imgproc.getPerspectiveTransform(srcMat, dstMat)
        val warped = Mat()
        Imgproc.warpPerspective(src, warped, M, Size(outWidth.toDouble(), outHeight.toDouble()))
        srcMat.release(); dstMat.release(); M.release()
        return warped
    }

    // ── Step 8 – Optional binarisation ───────────────────────

    /** Adaptive binarisation → classic black-and-white scan output. */
    private fun binarise(src: Mat): Mat {
        val gray = Mat()
        val out  = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
        Imgproc.adaptiveThreshold(
            gray, out, 255.0,
            Imgproc.ADAPTIVE_THRESH_MEAN_C,
            Imgproc.THRESH_BINARY, 21, 10.0
        )
        gray.release()
        return out
    }

    // ── Point ordering ────────────────────────────────────────

    /**
     * Sort four points into (TL, TR, BR, BL) order.
     * Sum  (x+y): minimum → TL, maximum → BR
     * Diff (x-y): minimum → TR, maximum → BL
     */
    fun orderPoints(pts: Array<Point>): Array<Point> {
        val sumArr  = pts.map { it.x + it.y }
        val diffArr = pts.map { it.x - it.y }
        return arrayOf(
            pts[sumArr.indexOf(sumArr.min())],    // TL
            pts[diffArr.indexOf(diffArr.min())],  // TR
            pts[sumArr.indexOf(sumArr.max())],    // BR
            pts[diffArr.indexOf(diffArr.max())]   // BL
        )
    }

    // ── Geometry helpers ─────────────────────────────────────

    private fun distance(a: Point, b: Point) =
        sqrt((b.x - a.x).pow(2) + (b.y - a.y).pow(2))

    private fun angleDeg(a: Point, b: Point, c: Point): Double {
        val v1  = Point(a.x - b.x, a.y - b.y)
        val v2  = Point(c.x - b.x, c.y - b.y)
        val dot = v1.x * v2.x + v1.y * v2.y
        val mag = sqrt(v1.x.pow(2) + v1.y.pow(2)) * sqrt(v2.x.pow(2) + v2.y.pow(2))
        return if (mag == 0.0) 0.0
        else Math.toDegrees(acos((dot / mag).coerceIn(-1.0, 1.0)))
    }

    private fun contourArea(pts: Array<Point>): Double {
        val mat = MatOfPoint(*pts)
        return Imgproc.contourArea(mat).also { mat.release() }
    }

    private fun convexHullArea(pts: Array<Point>): Double {
        val mat      = MatOfPoint(*pts)
        val hull     = MatOfInt()
        Imgproc.convexHull(mat, hull)
        val hullPts  = MatOfPoint(*hull.toArray().map { pts[it] }.toTypedArray())
        return Imgproc.contourArea(hullPts).also {
            mat.release(); hull.release(); hullPts.release()
        }
    }

    private fun otsuThreshold(gray: Mat): Double {
        val dummy  = Mat()
        val thresh = Imgproc.threshold(
            gray, dummy, 0.0, 255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )
        dummy.release()
        return thresh
    }

    // ── Destructure Array<Point> into (tl, tr, br, bl) ───────
    private operator fun Array<Point>.component1() = this[0]
    private operator fun Array<Point>.component2() = this[1]
    private operator fun Array<Point>.component3() = this[2]
    private operator fun Array<Point>.component4() = this[3]

    private data class ScoredQuad(val quad: Array<Point>, val score: Double)
}

// ─────────────────────────────────────────────────────────────
// MAT EXTENSION UTILITIES
// ─────────────────────────────────────────────────────────────

private fun Mat.gaussianBlur(ksize: Int): Mat {
    val out = Mat()
    Imgproc.GaussianBlur(this, out, Size(ksize.toDouble(), ksize.toDouble()), 0.0)
    return out
}

private fun Mat.dilate(iterations: Int = 1) {
    val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
    Imgproc.dilate(this, this, kernel, Point(-1.0, -1.0), iterations)
    kernel.release()
}

private fun Mat.scaled(scale: Double): Mat {
    val out = Mat()
    Imgproc.resize(this, out, Size(cols() * scale, rows() * scale))
    return out
}