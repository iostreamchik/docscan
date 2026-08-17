# data Package

Holds all detection backends, OpenCV infrastructure, utilities, and the repository implementation for document detection.

## Structure

```
data/
├── detector/
│   ├── IDocumentDetector.kt                        — Interface for pluggable detection backends
│   ├── CombinedDocumentDetector.kt                  — Async orchestrator (parallel classical + parallel ONNX fallbacks)
│   ├── scheme.md                                    — Design doc for the combined orchestration flow
│   ├── DocumentDetectorMinimal.kt                   — Minimal classical pipeline
│   ├── DocumentDetectorDirectionalSuppression.kt    — Classical pipeline with directional suppression
│   ├── HeatmapCornerDetector.kt                     — LCNet100+BiFPN ONNX heatmap corner regression
│   ├── CornerKeypointDetector.kt                    — LCNet ONNX corner keypoint detection
│   ├── SegmentationDetector.kt                      — DeepLabV3 semantic segmentation
│   ├── OnnxSessionManager.kt                        — Shared ONNX session init, NCHW tensor prep, cleanup
│   ├── MockDocumentDetector.kt                      — No-op detector for Compose previews
│   └── ClassicalDetectorExtensions.kt               — captureClassicalSnapshots extension on IMatBundle
│
├── opencv/
│   ├── MatBundle.kt                                 — IMatBundle interface + MatBundle implementation (pooled Mats)
│   ├── MockMatBundle.kt                             — Lazy no-op MatBundle for Compose previews
│   └── OpenCVAdapter.kt                             — Shared preprocessing, quad detection, and utility operations
│
├── repository/
│   └── DocumentDetectorRepositoryImpl.kt            — Adapter: IDocumentDetector → IDocumentDetectorRepository
│
└── utils/
    ├── ContourScoring.kt                            — scoreContourWithParams (configurable weights)
    ├── Extensions.kt                                — ImageProxy→Mat, Mat→Bitmap, fixRotation, warp, enhance, quad helpers
    └── QuadGeometry.kt                              — sortQuadPoints, quadDistance, quadHash, isRectangle, computeMaxAngleDeviation, validateQuadRectangularity
```

## Detection Backends

All detectors implement `IDocumentDetector`:

| Detector | Pipeline | Quad Selection |
|---|---|---|
| `DocumentDetectorMinimal` | `preprocessClassical` (default config) → morphClose (7×7) | `findBestQuad` (`minAreaFraction`, `approxPolyDPTolerance`, `rectangleTolerance=20°`), max `scoreContourWithParams` wins |
| `DocumentDetectorDirectionalSuppression` | `preprocessClassical` (directional config) → morphClose (strong close 3–5, odd-adjusted) → horizontal close (k×1) → vertical close (1×k) | `findBestQuad` (`approxEpsilon=0.015`, default tolerance), largest area wins |
| `HeatmapCornerDetector` | Resize to 256×256 → LCNet100+BiFPN ONNX (4 heatmap channels) → per channel: threshold → contours → centroid → scale to original coords | 4 centroids → `sortQuadPoints` → geometry validation (angle deviation, aspect ratio ≥ 0.15) |
| `CornerKeypointDetector` | Resize to 256×256 → LCNet ONNX (8 coords + 1 score) → scale to original coords → gradient-based corner refinement | Refined (or unrefined) corners → `sortQuadPoints` → geometry validation (angle deviation, aspect ratio ≥ 0.15) |
| `SegmentationDetector` | Aspect-ratio resize → pad to 384×384 → normalize → DeepLabV3 ONNX → sigmoid(fg−bg) → Otsu/maskThreshold → connected components → adaptive close/open/close → largest blob → crop + resize to scaled dims | Convex hull → `approxPolyDP` (0.02×perimeter) + diagonal-extreme fallback, max `scoreContourWithParams` wins |

### IDocumentDetector Interface

Core methods with default implementations where applicable:

- `preprocess(rawMat, scaledWidth, scaledHeight, params)` — full preprocessing pipeline. Classical detectors return the Canny edges Mat; ONNX detectors run inference and return either a pooled morph Mat (heatmap, corner keypoint) or a resized segmentation mask (segmentation)
- `detectQuad(morphImage, …, rotation = 0, params = PipelineParams())` — extract best quad in original coords
- `validateQuadSize(quad, originalWidth, originalHeight)` — default method; rejects quads filling >95% of frame via bounding rect
- `detectionParams` — optional `StateFlow<DetectionParameters>` (default returns null)
- `detectorName` — human-readable name (default returns `javaClass.simpleName`)
- `captureIntermediateSnapshots(rotation)` — snapshots from preprocess stage (default returns empty `IntermediateBitmaps`)
- `capturePostDetectionSnapshots(rotation)` — snapshots available only after detectQuad (default returns empty `IntermediateBitmaps`)
- `release()` — release all native resources

### Shared Classical Pipeline

`OpenCVAdapter.preprocessClassical()` and `OpenCVAdapter.findBestQuad()` factor out common classical detection logic:

**preprocessClassical** — shared by `DocumentDetectorMinimal` and `DocumentDetectorDirectionalSuppression`:
- `resizeToGray` → `medianBlur` (ksize ≥ 3)
- Brightness-driven CLAHE: brightness = mean (coerced 20–200); dim images (<80) get `dimBoostDivisor/(brightness+10)`, bright images (>130) get `(brightness−130)/brightBoostDivisor`; clip limit = `(0.5 + boosts).coerceIn(1.0, 1.5)`, tile = `claheTileSize` (≥ 8)
- Conditional morphClose (skipped when enhanced stdDev < 25)
- `GaussianBlur` 3×3 (σ=2) → Otsu (its threshold becomes `cannyHigh`, `cannyLow = 0.2 × cannyHigh`) → `Canny`
- Callback `DetectionParamsCallback` reports raw doubles (brightness, CLAHE clip, Canny high/low); each detector formats them into `DetectionParameters` strings

**findBestQuad** — shared candidate extraction:
- `findContours` → filter by area ≥ `minAreaFraction`×frame and ≥ 10 points → `approxPolyDP` (`approxEpsilon × perimeter`) → `isRectangle` (`rectangleTolerance`, default 15°) → scale to original coords → solidity check (≥ 0.5)
- Caller provides `QuadSelector` lambda to pick the winner from candidates

### PreprocessingConfig

Data class controlling classical preprocessing tuning:

| Property | Default | Purpose |
|---|---|---|
| `dimBoostDivisor` | 40.0 | Divisor for dim-image CLAHE boost |
| `brightBoostDivisor` | 60.0 | Divisor for bright-image CLAHE boost |
| `brightnessFormat` | `"%.0f"` | Display format for brightness metric |
| `claheFormat` | `"%.1f"` | Display format for CLAHE clip limit |
| `cannyFormat` | `"%.0f"` | Display format for Canny thresholds |

`DocumentDetectorDirectionalSuppression` overrides with `dimBoostDivisor=8.0`, `brightBoostDivisor=100.0`, `brightnessFormat="%.1f"`, `claheFormat="%.2f"`, `cannyFormat="%d"`.

### CombinedDocumentDetector

Orchestrates all five detectors (2 classical + 3 ONNX) via the `AsyncDetectorSource` enum:

```
preprocess() — parallel
  ├─ async minimalDetector.preprocess()
  └─ async opencv5Detector.preprocess()
  → caches both morph Mats + cloned raw Mat, returns directional morph

detectQuad()
  Phase 1 — parallel primary
  ├─ async minimalDetector.detectQuad(minimalMorph)
  └─ async opencv5Detector.detectQuad(directionalMorph)
  → score each by max angle deviation from 90° (computeMaxAngleDeviation),
    filter with validateQuadSize, winner = lowest deviation
  Phase 2 — parallel ONNX fallback (only if no valid primary)
  ├─ async heatmapCornerDetector  (runSingleDetector)
  ├─ async cornerKeypointDetector (runSingleDetector)
  └─ async onnxDetector           (runSingleDetector)
  → winner = first non-null in priority order HEATMAP_CORNER → CORNER_KEYPOINT
    → SEGMENTATION; losing coroutines are cancelled; each result is
    validateQuadSize-checked inside runSingleDetector
```

- `AsyncDetectorSource` values: `NONE`, `MINIMAL`, `DIRECTIONAL_SUPPRESSION`, `HEATMAP_CORNER`, `CORNER_KEYPOINT`, `SEGMENTATION` — each carries a `detectionParamsName` string
- `detectorName` = `lastUsedDetector.detectionParamsName` (falls back to "Combined" when NONE)
- Emits the winning detector's `detectionParams` (with the source name applied)
- Snapshot delegation: `captureIntermediateSnapshots` routes to the last-used detector (Minimal by default); `capturePostDetectionSnapshots` routes only for Segmentation

Caching:
- `cachedMorphImages` — `Map<AsyncDetectorSource, Mat?>` from parallel classical preprocess
- `cachedRawMat` — cloned raw Mat for ONNX fallback detectors
- `cachedScaledWidth` / `cachedScaledHeight` — scaled dimensions for fallback calls
- `angleDeviations` — per-source max angle deviation (diagnostics)

### OnnxSessionManager

Shared infrastructure for ONNX-based detectors. Encapsulates:
- Idempotent session initialization from asset model bytes (XNNPACK, 2 intra-op threads, memory pattern, ALL_OPT, PARALLEL execution mode)
- NCHW tensor preparation from RGB Mat (single interleaved `get` → cached reusable FloatArrays → deinterleave → OnnxTensor); zero per-call allocations, `require` on `inputSize` dimensions
- Session cleanup (the shared `OrtEnvironment` is DI-owned, not owned by the manager)

Public API:
- `init(tag)` — lazy session init (idempotent)
- `getSession()` — access `OrtSession`
- `prepareInputTensor(rgbMat, inputSize)` — NCHW tensor from RGB Mat (channel count derived from the Mat; input buffers cached per manager instance, safe because each detector owns its manager)
- `inputName` — first input name from session
- `close()` — release session

`HeatmapCornerDetector`, `CornerKeypointDetector`, and `SegmentationDetector` construct an `OnnxSessionManager(context, env, modelPath)` instead of inline ONNX boilerplate.

### ONNX Detectors

- **HeatmapCornerDetector**: `INPUT_SIZE` = 256px, model `onnx/lcnet100_h_e_bifpn_256_fp32.onnx`, output 4 per-corner heatmaps
  - Configurable: `heatmapThreshold` (0.05–0.7, default 0.3), `minCornerArea` (default 0.0001), `maxAngleDeviation` (5–60°, default 45°)
  - Inference runs inside `preprocess`: 256×256 → RGB /255 → run → per channel: scale to 8-bit → threshold → `findContours` (reused hierarchy Mat) → largest contour (area ≥ `minCornerArea`×heatmap area, min 4px) → centroid via `Geometry.moments` → scale to original coords
  - Caches the 4 corners (or null) + corner visualization bitmap (colored dots + white connecting lines over the raw image) + raw Mat clone
  - `detectQuad`: cached corners → `sortQuadPoints` → `computeMaxAngleDeviation` ≤ `maxAngleDeviation` + aspect ratio ≥ 0.15
  - `detectionParams`: `heatmapThreshold`, `cornerError` ("only N/4 corners" / "geometry failed")
  - `captureIntermediateSnapshots` returns `corners` bitmap

- **CornerKeypointDetector**: `INPUT_SIZE` = 256px, model `onnx/lcnet050_p_multi_decoder_l3_d64_256_fp32.onnx`, output 8 normalized coords + 1 score
  - Configurable: `minScore` (0.05–0.95, default 0.3), `maxAngleDeviation` (5–60°, default 45°), `applySigmoid` (default false)
  - Refinement: `cornerRefinementRadius` (5–100, default 30), `cornerRefinementGradientThreshold` (3–60, default 15)
  - Inference runs inside `preprocess`; `detectQuad` scales coords to original size, `sortQuadPoints`, then refines on the cached raw Mat via Sobel gradient magnitude: bisector search (2 iterations, converges at avg shift < 1.5px, applied when t > 2) + edge snapping (perpendicular search along neighbor edges, applied when 1.0 < dist < radius); falls back to unrefined corners if refinement fails geometry validation
  - Geometry validation: `computeMaxAngleDeviation` ≤ `maxAngleDeviation` + aspect ratio ≥ 0.15
  - `detectionParams`: `cornerScore`, `cornerError` ("score X < min Y" / "geometry failed")
  - `captureIntermediateSnapshots` returns `corners` bitmap

- **SegmentationDetector**: `INPUT_SIZE` = 384px, model `onnx/deeplabv3_mbv3_docseg.onnx`, output 2 channels (bg/fg, NCHW/NHWC auto-detected)
  - Configurable: `maskThreshold` (0.1–0.7, default 0.5), `useCustomNormalization` (default true)
  - Custom normalization: mean `(0.4611, 0.4359, 0.3905)`, std `(0.2193, 0.2150, 0.2109)`
  - Preprocess: aspect-ratio resize → pad to 384² (gray 128) → `NORM_MINMAX` [0,1] → mean/std (when custom) → inference → sigmoid(fg−bg) → zero padded regions → GaussianBlur 5×5 → Otsu on content region (falls back to fixed `maskThreshold` when the Otsu mask has < 0.05% foreground) → `connectedComponentsWithStats` → adaptive kernels from `docLinear = sqrt(largest area).coerceIn(10, 200)` (close = docLinear/6, 5–21 odd; open = docLinear/12, 3–9 odd) → close, open, close → largest blob (≥ 0.05%) → crop to content → resize to scaled dims (`INTER_NEAREST`)
  - `detectQuad`: rejects masks with < 3% foreground; contours (area ≥ `minAreaFraction`×frame, ≥ 10 points) → convex hull → `approxPolyDP` (0.02×perimeter) → `isRectangle`; primary candidates require solidity ≥ 0.3; fallback uses diagonal-extreme corners (solidity ≥ 0.5, `validateQuadRectangularity` 15°, aspect ratio ≥ 0.35); winner = max `scoreContourWithParams`
  - Caches segmentation mask in `cachedMask` (cloned out of the pooled morph) and raw bitmap in `cachedRawBitmap`
  - `captureIntermediateSnapshots` and `capturePostDetectionSnapshots` both return mask overlay (darkens non-document regions to 30%, rotation-corrected)

## OpenCV Infrastructure

### IMatBundle / MatBundle

Pooled Mat allocation — prevents per-frame GC pressure. `IMatBundle` is an interface defined in `MatBundle.kt`.

Pooled slots:

| Category | Slots |
|---|---|
| Core pipeline | `getGray()`, `getBlurred()`, `getEnhanced()`, `getMorph()`, `getTemp()`, `getEdges()`, `getMorphAdd()`, `getHierarchy()` |
| Directional suppression | `getGrayGaussian()`, `getHorizontalClose()`, `getVerticalClose()` |
| Statistics | `getMean()` (MatOfDouble), `getStd()` (MatOfDouble) |
| Kernels | `getKernel()`, `getKernel2()`, `getHorizontalKernel()`, `getVerticalKernel()` |
| Geometry | `getHull()` (MatOfInt), `getHullPoints()` (MatOfPoint2f), `getApprox()` (MatOfPoint2f) |
| Adaptive threshold | `getAdaptiveBinary()` |
| Otsu pipeline | `getOtsuBlur()`, `getOtsuThreshold()` |
| Sobel gradient | `getSobelX()`, `getSobelY()`, `getGradMag()` |

Each getter returns a reusable Mat — callers write into it every frame. `releaseAll()` releases every slot (call once on cleanup).

**MockMatBundle**: Returns a lazy shared empty `Mat()` (plus fresh empty `MatOf*` types) to avoid `UnsatisfiedLinkError` in Compose previews. `releaseAll()` is a no-op.

### OpenCVAdapter

Static utility object wrapping common OpenCV operations:

| Function | Description |
|---|---|
| `resizeToGray(source, width, height, gray)` | Resize + RGBA→grayscale into pooled destination |
| `getAverageBrightness(image, bundle)` | Mean pixel value via `meanStdDev` (pooled stats) |
| `getStdDev(image, bundle)` | Standard deviation via `meanStdDev` (pooled stats) |
| `applyClahe(source, dest, clipLimit, tileSize)` | CLAHE contrast enhancement |
| `createRectKernel(size, kernel)` | Create rectangular structuring element (releases + copies into kernel Mat) |
| `morphClose(source, dest, kernel)` | Morphological close operation |
| `findContours(image, hierarchy)` | Find contours with `RETR_LIST` + `CHAIN_APPROX_SIMPLE` |
| `isRectangle(approx, toleranceDegrees = 15.0)` | Delegates to `QuadGeometry.isRectangle` |
| `preprocessClassical(rawMat, scaledW/H, params, config, bundle, onParams)` | Shared classical preprocessing pipeline with adaptive CLAHE and conditional morph close |
| `findBestQuad(morphImage, bundle, scaledW/H, origW/H, minAreaFraction, approxEpsilon, rectangleTolerance, selector)` | Shared contour → quad candidate extraction with configurable epsilon and selector |

Type aliases: `DetectionParamsCallback` (brightness, claheClipLimit, cannyHigh, cannyLow → Unit) and `QuadSelector` (List<MatOfPoint> → MatOfPoint?).

## Repository

`DocumentDetectorRepositoryImpl` adapts `IDocumentDetector` → `IDocumentDetectorRepository` (domain interface). Pure delegation — reconstructs `IntermediateBitmaps` with explicit field mapping for both snapshot capture methods.

## Utilities

### ContourScoring.kt

`scoreContourWithParams(contour, width, height, params)` — weighted scoring:

- **Area** (`scoreAreaWeight`): raw contour area
- **Center proximity** (`scoreCenterWeight`): closeness to frame center (1 − dist/maxDist), scaled by frame area
- **Area ratio** (`scoreAreaRatioWeight`): full score (1.0) for small quads (≤ 0.02), linear penalty down to 0.2 for large quads (≥ 0.5)

### Extensions.kt

Extension functions on `Mat`, `MatOfPoint`, and `ImageProxy`:

| Function | Description |
|---|---|
| `ImageProxy.toMatRGBA()` | YUV_420_888 → RGBA via OpenCV (handles row stride padding, interleaved/planar UV) |
| `Mat.fixRotation(degrees)` | Rotate Mat to correct device orientation (90°, 180°, 270°) |
| `Mat.toBitmap()` | Mat → Android Bitmap (handles 1/4 channel conversion via RGB intermediate) |
| `Mat.toBitmap(width, height)` | Resize + convert to Bitmap |
| `Mat.rotate90Clockwise()` | Transpose + flip Y-axis |
| `Mat.rotate90CounterClockwise()` | Transpose + flip X-axis |
| `Mat.enhanceDocument()` | LAB-based shadow removal (55px Gaussian) + CLAHE (2.0, 8×8) + sharpening (σ=2, weight 1.3) |
| `Mat.sharpen()` | Gaussian unsharp mask (σ=2, weight 1.3) |
| `MatOfPoint.toSortedQuad()` | Sort 4 points clockwise from top-left (returns empty list if not exactly 4) |
| `calculateWarpedDimensions(tl, tr, br, bl)` | Compute output width/height from max edge distances |
| `warpDocumentHighQuality(src, quad, rotation)` | Perspective transform → rotation fix → sharpening → Bitmap |

### QuadGeometry.kt

Geometric operations on quads:

| Function | Description |
|---|---|
| `sortQuadPoints(points)` | Centroid angle sort → top-left anchor (min x+y) → clockwise winding (shoelace signed area) |
| `quadDistance(quad1, quad2, fw, fh)` | Average corner shift normalized by frame diagonal [0, 1] |
| `quadHash(quad)` | Simple coordinate hash for change detection |
| `isRectangle(approx, tolerance)` | All angles within tolerance of 90° (default 15°) |
| `computeMaxAngleDeviation(corners)` | Max deviation from 90° across all four interior angles |
| `validateQuadRectangularity(corners, maxDeviationDegrees)` | All angles within threshold of 90° (delegates to `computeMaxAngleDeviation`) |
| `computeAngle(p1, p2, center)` | Private helper — interior angle at vertex via dot product |

## Shared Entities (entity/ package)

| Class | Purpose |
|---|---|
| `DetectionParameters` | Runtime metrics (all `String`): `detectorName`, `claheClipLimit`, `cannyHigh/Low`, `brightness`, `maskThreshold`, `heatmapThreshold`, `cornerScore`, `cornerError` |
| `IntermediateBitmaps` | Snapshot container: `blur`, `clahe`, `morph`, `edges`, `mask`, `corners` |
| `PipelineParams` | Configurable pipeline knobs (defaults: `medianBlurKsize=5`, `claheClipLimit=1.5f`, `claheTileSize=5`, `morphCloseSize=5`, `cannyLow=0f`, `cannyHigh=0f`, `strongCloseSize=5`, `directionalKernelSize=5`, `approxPolyDPTolerance=0.025f`, `minAreaFraction=0.025f`, `scoreAreaWeight=0.5f`, `scoreCenterWeight=0.3f`, `scoreAreaRatioWeight=0.2f`) |

## Domain Contracts (domain/ package)

| Interface | Purpose |
|---|---|
| `IDocumentDetectorRepository` | Domain abstraction for detection; mirrors `IDocumentDetector` with `validateQuadSize` as abstract (no default) |

## Key Patterns

- **Mat pooling**: All Mats go through `IMatBundle`. Never allocate Mats in hot paths. Call `releaseAll()` on cleanup.
- **Bitmap safety**: Clone bitmaps before emitting — Compose gets independent copies. Never recycle bitmaps in `onCleared`.
- **Lazy preview safety**: `MockMatBundle` uses lazy `Mat()` to avoid `UnsatisfiedLinkError` when OpenCV native libs aren't loaded.
- **ONNX mask caching**: `SegmentationDetector.cachedMask` is a clone of the pooled morph — `matBundle.getMorph()` is a shared slot reused every frame.
- **ONNX raw caching**: `CornerKeypointDetector.cachedRawMat` stores the original Mat for gradient-based corner refinement; `HeatmapCornerDetector.cachedRawMat` backs the corner visualization.
- **Shared classical pipeline**: `OpenCVAdapter.preprocessClassical()` and `findBestQuad()` eliminate duplication between Minimal and DirectionalSuppression detectors.
- **Parallel ONNX fallback**: All three ONNX detectors run concurrently; the first valid result in fixed priority order (heatmap → corner keypoint → segmentation) wins and losing coroutines are cancelled.
- **OpenCV 5 APIs**: Uses `org.opencv.geometry.Geometry` for `contourArea()`, `arcLength()`, `approxPolyDP()`, `convexHull()`, `boundingRect()`, `moments()`, `getPerspectiveTransform()`.
- **No inline comments**: Code is self-documenting, clean, and minimal.
