package io.github.iostreamchik.scanner.opencv

/**
 * Sealed hierarchy for pipeline parameters.
 *
 * - `Auto` — brightness-adaptive CLAHE (the old `null` semantics)
 * - `Default` — hardcoded defaults for all parameters
 * - `Manual` — user-configured parameters (data class with `copy()`)
 *
 * Convenience accessors are provided as extension functions so the sealed
 * class itself has no properties that would conflict with `Manual`'s data
 * class properties.
 */
sealed class PipelineParams {

    object Auto : PipelineParams()
    data class Manual(
        // Blur
        val medianBlurKsize: Int = 5,

        // CLAHE
        val claheClipLimit: Float = 0.5f,
        val claheTileSize: Int = 8,

        // Morph Close (pre-Canny)
        val morphCloseSize: Int = 3,

        // Canny
        val cannyLow: Float = 0f,
        val cannyHigh: Float = 0f,
        val cannyAutoDetect: Boolean = true,

        // Strong Closing (post-Canny)
        val strongCloseSize: Int = 3,

        // Directional Suppression
        val directionalKernelSize: Int = 6,

        // Contour Detection
        val approxPolyDPTolerance: Float = 0.015f,
        val minAreaFraction: Float = 0.025f,

        // Scoring weights
        val scoreAreaWeight: Float = 0.5f,
        val scoreCenterWeight: Float = 0.3f,
        val scoreAreaRatioWeight: Float = 0.2f,
    ) : PipelineParams()
}

// ── Convenience accessors as extension functions ─────────────────────────
// Delegate to Manual's properties; return defaults for Auto/Default.
// This keeps the sealed class clean while letting callers read .medianBlurKsize
// etc. on any PipelineParams without exhaustive when expressions.

private inline val PipelineParams._manual: PipelineParams.Manual?
    get() = this as? PipelineParams.Manual

val PipelineParams.medianBlurKsize: Int
    get() = _manual?.medianBlurKsize ?: 5

val PipelineParams.claheClipLimit: Float
    get() = _manual?.claheClipLimit ?: 0.5f

val PipelineParams.claheTileSize: Int
    get() = _manual?.claheTileSize ?: 8

val PipelineParams.morphCloseSize: Int
    get() = _manual?.morphCloseSize ?: 3

val PipelineParams.cannyLow: Float
    get() = _manual?.cannyLow ?: 0f

val PipelineParams.cannyHigh: Float
    get() = _manual?.cannyHigh ?: 0f

val PipelineParams.cannyAutoDetect: Boolean
    get() = _manual?.cannyAutoDetect ?: true

val PipelineParams.strongCloseSize: Int
    get() = _manual?.strongCloseSize ?: 3

val PipelineParams.directionalKernelSize: Int
    get() = _manual?.directionalKernelSize ?: 6

val PipelineParams.approxPolyDPTolerance: Float
    get() = _manual?.approxPolyDPTolerance ?: 0.015f

val PipelineParams.minAreaFraction: Float
    get() = _manual?.minAreaFraction ?: 0.025f

val PipelineParams.scoreAreaWeight: Float
    get() = _manual?.scoreAreaWeight ?: 0.5f

val PipelineParams.scoreCenterWeight: Float
    get() = _manual?.scoreCenterWeight ?: 0.3f

val PipelineParams.scoreAreaRatioWeight: Float
    get() = _manual?.scoreAreaRatioWeight ?: 0.2f

/**
 * Creates a [PipelineParams.Manual] with the given property overrides.
 * Uses the convenience accessors to preserve the current value (or default)
 * for any parameter not explicitly overridden.
 *
 * This replaces the old `data class.copy()` pattern so UI code can call
 * `params.copyAsManual(medianBlurKsize = newValue)` regardless of whether
 * `params` is `Auto`, `Default`, or `Manual`.
 */
fun PipelineParams.copyAsManual(
    medianBlurKsize: Int = this.medianBlurKsize,
    claheClipLimit: Float = this.claheClipLimit,
    claheTileSize: Int = this.claheTileSize,
    morphCloseSize: Int = this.morphCloseSize,
    cannyLow: Float = this.cannyLow,
    cannyHigh: Float = this.cannyHigh,
    cannyAutoDetect: Boolean = this.cannyAutoDetect,
    strongCloseSize: Int = this.strongCloseSize,
    directionalKernelSize: Int = this.directionalKernelSize,
    approxPolyDPTolerance: Float = this.approxPolyDPTolerance,
    minAreaFraction: Float = this.minAreaFraction,
    scoreAreaWeight: Float = this.scoreAreaWeight,
    scoreCenterWeight: Float = this.scoreCenterWeight,
    scoreAreaRatioWeight: Float = this.scoreAreaRatioWeight,
): PipelineParams.Manual = PipelineParams.Manual(
    medianBlurKsize = medianBlurKsize,
    claheClipLimit = claheClipLimit,
    claheTileSize = claheTileSize,
    morphCloseSize = morphCloseSize,
    cannyLow = cannyLow,
    cannyHigh = cannyHigh,
    cannyAutoDetect = cannyAutoDetect,
    strongCloseSize = strongCloseSize,
    directionalKernelSize = directionalKernelSize,
    approxPolyDPTolerance = approxPolyDPTolerance,
    minAreaFraction = minAreaFraction,
    scoreAreaWeight = scoreAreaWeight,
    scoreCenterWeight = scoreCenterWeight,
    scoreAreaRatioWeight = scoreAreaRatioWeight,
)
