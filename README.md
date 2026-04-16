# DocumentScanner – Quad-Finder Document Detector

A robust Android document scanner using OpenCV contour-based quadrilateral detection.

## Features

- **Adaptive preprocessing** – Auto-scales Canny thresholds via Otsu's method;
  falls back to CLAHE-enhanced path for low-contrast images.
- **Multi-strategy search** – Tries three independent edge maps and picks the
  best quad by convexity + aspect-ratio scoring.
- **Validation** – Rejects non-convex quads (folded pages, shadows) and degenerate angles.
- **Stability tracking** – Confirms document across multiple frames before triggering capture.
- **Memory safety** – Explicit Mat.release() via extension helpers.
- **Configurable** – All tunable parameters in one `DetectorConfig` data class.

## How to Use

### 1. Basic detection (no stability tracking)

```kotlin
val detector = DocumentDetector()

val result = detector.detect(previewMat)
if (result is DetectionResult.Found) {
    val quad = result.quad          // [TL, TR, BR, BL] in preview coordinates
    val warp = result.warped        // perspective-corrected image

    // Save/warp/capture your document here
}
```

### 2. Stability-aware detection (recommended for camera preview)

```kotlin
val detector = DocumentDetector()

// Call on every preview frame
val state = detector.detectWithStability(
    src = previewMat,
    frameWidth = previewSize.width,
    frameHeight = previewSize.height,
)

if (state.isDetected && state.confidence > 0.8f) {
    // Document is stable – capture the frame!
    val corners = state.quad        // ordered TL→TR→BR→BL
}

// After a successful capture, reset stability tracking
detector.reset()
```

### 3. Using a custom config

```kotlin
val config = DocumentDetectorConfig(
    stableFramesRequired = 5,          // require 5 matching frames
    similarityThreshold = 10f,         // allow ±10px drift between frames
    workingScale = 0.75,               // process at 75% resolution
    binarise = true,                   // output classic B&W scan
)

val detector = DocumentDetector(config)
```

### 4. Lightweight config for low-memory devices

```kotlin
val config = DocumentDetectorLiteConfig() // smaller candidates, lower resolution
val detector = DocumentDetector(config)
```

## Architecture

The detector operates in two passes per frame:

1. **Contour extraction**
   - Converts to grayscale
   - Builds three edge maps (Otsu Canny, CLAHE-enhanced, Adaptive threshold)
   - Finds external contours and filters by area fraction

2. **Quadrilateral selection**
   - Applies multiple epsilon factors for polygon approximation
   - Scores candidates by area × convexity × angle regularity
   - Validates via convexity check + interior angle range (55°–125°)
   - Orders points into (TL, TR, BR, BL) via sum/diff sorting

3. **Perspective warp**
   - Computes homography from source → destination corners
   - Warrants a rectangle that fills the largest dimension
   - Optionally binarizes output for classic scan appearance

## Tuning Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `minAreaFraction` | 0.08 | Documents must be ≥8% of frame |
| `maxAreaFraction` | 0.97 | Reject full-frame contours (e.g., entire preview) |
| `candidateCount` | 8 | Number of top contours to evaluate |
| `epsilonFactors` | [0.01…0.06] | Tolerance for Douglas-Peucker simplification |
| `minAngleDeg` | 55° | Reject quads with angles <55° |
| `maxAngleDeg` | 125° | Reject quads with angles >125° |
| `binarise` | false | If true, output is black-and-white |
| `workingScale` | 1.0 | Process at this fraction of input size |
| `stableFramesRequired` | 5 | Frames to confirm before triggering |
| `similarityThreshold` | 10px | Drift allowed between stable frames |

## Dependencies

- [OpenCV for Android](https://opencv.org/android/)

## License

MIT License – feel free to use in commercial or open-source projects.
