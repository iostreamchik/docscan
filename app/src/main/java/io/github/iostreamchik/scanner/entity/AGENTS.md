# entity

Pure Kotlin data classes shared across all layers. The entity layer has **zero dependencies** on `data`, `domain`, or `presenter` — it depends only on `androidx.compose.runtime` for stability annotations and, in one case, `android.graphics.Bitmap` for snapshot transfer.

## Responsibility

- Define shared data classes consumed by every other layer
- Provide default parameter values so callers can omit unused fields
- Annotate with `@Immutable` or `@Stable` for Compose memory optimisation

## Package Structure

```
entity/
├── DetectionParameters.kt    — Runtime detection metrics (String values for UI display)
├── IntermediateBitmaps.kt    — Bitmap snapshots of detection pipeline stages
└── PipelineParams.kt         — Configurable detection pipeline knobs
```

## Files

| File | Type | Annotation | Responsibility |
|---|---|---|---|
| `DetectionParameters.kt` | Data class | `@Immutable` | Runtime metrics reported by the active detector via `StateFlow` |
| `IntermediateBitmaps.kt` | Data class | `@Stable` | Bitmap snapshots of detection pipeline stages (blur, clahe, morph, edges, mask, corners) |
| `PipelineParams.kt` | Data class | `@Immutable` | Configurable knobs controlling detection pipeline behaviour |

## Data Classes

### `PipelineParams`

Configurable parameters passed through the detection pipeline. Every parameter has a default value tuned for general document detection. All fields are primitive types (`Int`/`Float`) for zero-allocation copying.

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `medianBlurKsize` | `Int` | 5 | Median blur kernel size (must be odd) |
| `claheClipLimit` | `Float` | 1.5f | CLAHE contrast clip limit |
| `claheTileSize` | `Int` | 5 | CLAHE tile grid size (tileSize × tileSize) |
| `morphCloseSize` | `Int` | 5 | Morphological close kernel size |
| `cannyLow` | `Float` | 0f | Canny low threshold (0 = auto via Otsu) |
| `cannyHigh` | `Float` | 0f | Canny high threshold (0 = auto via Otsu) |
| `strongCloseSize` | `Int` | 5 | Aggressive post-Canny morph close kernel size |
| `directionalKernelSize` | `Int` | 5 | Directional suppression horizontal/vertical kernel size |
| `approxPolyDPTolerance` | `Float` | 0.025f | Polygon approximation tolerance (fraction of contour perimeter) |
| `minAreaFraction` | `Float` | 0.025f | Minimum quad area as fraction of total image area |
| `scoreAreaWeight` | `Float` | 0.5f | Weight for raw area in `scoreContourWithParams` |
| `scoreCenterWeight` | `Float` | 0.3f | Weight for center proximity in `scoreContourWithParams` |
| `scoreAreaRatioWeight` | `Float` | 0.2f | Weight for area-ratio penalty in `scoreContourWithParams` |

### `DetectionParameters`

Runtime metrics reported back from the active detector via `StateFlow<DetectionParameters>?`. All values are `String` for direct UI display without formatting overhead.

| Field | Type | Source |
|---|---|---|
| `detectorName` | `String` | Name of the detector that produced the result |
| `claheClipLimit` | `String` | Effective CLAHE clip limit used |
| `cannyHigh` | `String` | Effective Canny high threshold used |
| `cannyLow` | `String` | Effective Canny low threshold used |
| `brightness` | `String` | Input image brightness metric |
| `maskThreshold` | `String` | Segmentation mask threshold (ONNX detectors) |
| `heatmapThreshold` | `String` | Heatmap corner regression threshold (LCNet ONNX) |
| `cornerScore` | `String` | Corner keypoint confidence score (ONNX detectors) |
| `cornerError` | `String` | Corner keypoint fit error (ONNX detectors) |

### `IntermediateBitmaps`

Holds optional Bitmap snapshots of each detection pipeline stage. Uses `@Stable` (not `@Immutable`) because `Bitmap` is a mutable type — structural equality is not guaranteed, but referential stability reduces recompositions. This is the sole entity class that depends on an Android type (`android.graphics.Bitmap`).

| Field | Type | Stage |
|---|---|---|
| `blur` | `Bitmap?` | Median blur |
| `clahe` | `Bitmap?` | CLAHE contrast enhancement |
| `morph` | `Bitmap?` | Morphological close |
| `edges` | `Bitmap?` | Edge detection (Canny) |
| `mask` | `Bitmap?` | Segmentation mask (ONNX) or binary threshold |
| `corners` | `Bitmap?` | Corner keypoint detection output |

## Cross-Layer Usage

| Layer | Entity Class | Usage |
|---|---|---|
| `domain` | `PipelineParams` | Method parameter in `IDocumentDetectorRepository.detectQuad()` and `preprocess()` |
| `domain` | `DetectionParameters` | Element type of `IDocumentDetectorRepository.detectionParams` StateFlow |
| `domain` | `IntermediateBitmaps` | Return type of `captureIntermediateSnapshots()` / `capturePostDetectionSnapshots()` |
| `data` | `PipelineParams` | Passed through all detector preprocess and detectQuad calls |
| `data` | `DetectionParameters` | Emitted by detectors with runtime metrics |
| `data` | `IntermediateBitmaps` | Returned by all detector `captureIntermediateSnapshots` implementations |
| `presenter` | `PipelineParams` | Held in `CameraState.pipelineParams`; updated via `CameraIntent.UpdateParams` |
| `presenter` | `DetectionParameters` | Collected from `detectionParams` StateFlow in `CameraViewModel` |
| `presenter` | `IntermediateBitmaps` | Held in `CameraState.intermediateBitmaps`; consumed by `FileScanResultScreen` |

## Dependency Flow

```
entity (pure Kotlin data classes)
    ├── used by → domain (repository interfaces reference entity types)
    ├── used by → data (detectors consume PipelineParams, emit DetectionParameters)
    └── used by → presenter (ViewModels hold and update entity instances)

entity has NO imports from data/, domain/, or presenter/
```

## Adding a New Entity

1. Create the data class in `entity/` with all parameters having default values
2. Choose the correct stability annotation:
   - **`@Immutable`** — all fields are immutable types (`Int`, `Float`, `String`, `Boolean`, other `@Immutable` data classes). Compose treats the instance as structurally stable.
   - **`@Stable`** — any field is a mutable type (`Bitmap`, `MutableList`, custom mutable classes). Compose tracks referential equality only.
3. Prefer pure Kotlin types — if an Android or OpenCV type is unavoidable, document the rationale (see `IntermediateBitmaps` as precedent for `Bitmap`)
4. Update cross-layer usage tables in `domain/`, `data/`, and `presenter/` AGENTS.md files

## Key Patterns

- **`@Immutable` vs `@Stable`**: Use `@Immutable` for pure data classes (all primitive or `@Immutable` fields). Use `@Stable` when fields contain mutable types like `Bitmap`. Both avoid unnecessary recompositions but with different equality semantics.
- **Default parameters**: All fields have sensible defaults — callers construct partial instances via `copy()` without specifying every field.
- **String metrics in DetectionParameters**: Runtime metrics are `String` (not `Float`/`Int`) to avoid formatting overhead in the presenter layer.
- **No business logic**: Entity classes are pure data containers — no methods, no computed properties, no validation.
- **No inline comments**: Code is self-documenting, clean, and minimal.
