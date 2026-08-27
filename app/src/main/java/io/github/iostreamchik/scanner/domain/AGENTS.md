# domain

Business logic layer of the Clean Architecture. Defines **what** the app does without exposing **how** it does it. The domain layer depends only on `entity` for shared data classes and has no dependency on the `data` or `presenter` layers.

## Responsibility

- Declare repository interfaces that the `presenter` layer consumes and the `data` layer implements
- Enforce unidirectional data flow: presenter calls domain, domain delegates to data
- All domain models live in `entity/` — this layer contains only repository contracts

## Package Structure

```
domain/
└── repository/
    └── IDocumentDetectorRepository.kt — Repository interface for document detection
```

## Files

| File | Type | Responsibility |
|---|---|---|
| `repository/IDocumentDetectorRepository.kt` | Interface | Contract for document detection: preprocess, detectQuad, validateQuadSize, intermediate snapshots, lifecycle |

## Repository Interface

`IDocumentDetectorRepository` exposes the detection workflow to the presenter layer:

| Member | Signature | Purpose |
|---|---|---|
| `preprocess()` | `(rawMat, scaledWidth, scaledHeight, params) -> Mat` | Run image preprocessing pipeline, return processed Mat |
| `detectQuad()` | `(morphImage, scaledWidth, scaledHeight, originalWidth, originalHeight, params = PipelineParams(), rawMat = null) -> MatOfPoint?` | Extract document quad from preprocessed image; `rawMat` is the unscaled input image required by the ONNX fallback detectors (they are skipped when null) |
| `validateQuadSize()` | `(quad, originalWidth, originalHeight) -> Boolean` | Guard against false positives where quad fills the entire frame |
| `detectionParams` | `StateFlow<DetectionParameters>?` | Live detection metrics (nullable — null when no detector is active) |
| `detectorName` | `String` | Human-readable name of the active detector |
| `captureIntermediateSnapshots()` | `() -> IntermediateBitmaps` | Capture pipeline stage bitmaps after preprocess |
| `capturePostDetectionSnapshots()` | `() -> IntermediateBitmaps` | Capture bitmaps only available after detectQuad (e.g., ONNX mask) |
| `release()` | `() -> Unit` | Release native resources (pooled Mats, ONNX sessions) |

## Shared Entities (`entity/`)

The `entity` package sits alongside `domain` and holds all shared data classes used across layers. Entity classes are annotated with `@Immutable` (pure data) or `@Stable` (mutable references like Bitmaps) for Compose memory optimisation.

| File | Data Class | Annotation | Purpose |
|---|---|---|---|
| `PipelineParams.kt` | `PipelineParams` | `@Immutable` | Configurable knobs for the detection pipeline |
| `DetectionParameters.kt` | `DetectionParameters` | `@Immutable` | Runtime detection metrics exposed via `StateFlow` |
| `IntermediateBitmaps.kt` | `IntermediateBitmaps` | `@Stable` | Bitmap snapshots of detection pipeline stages |

### `PipelineParams`

Configurable parameters passed through the detection pipeline. Defaults tuned for general document detection.

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `medianBlurKsize` | `Int` | 5 | Median blur kernel size |
| `claheClipLimit` | `Float` | 1.5f | CLAHE contrast clip limit |
| `claheTileSize` | `Int` | 5 | CLAHE tile grid size |
| `morphCloseSize` | `Int` | 5 | Morphological close kernel size |
| `cannyLow` | `Float` | 0f | Canny low threshold (0 = auto) |
| `cannyHigh` | `Float` | 0f | Canny high threshold (0 = auto) |
| `strongCloseSize` | `Int` | 5 | Aggressive morph close kernel size |
| `directionalKernelSize` | `Int` | 5 | Directional suppression kernel size |
| `approxPolyDPTolerance` | `Float` | 0.025f | Polygon approximation tolerance (fraction of perimeter) |
| `minAreaFraction` | `Float` | 0.025f | Minimum quad area as fraction of image area |
| `scoreAreaWeight` | `Float` | 0.5f | Weight for area in contour scoring |
| `scoreCenterWeight` | `Float` | 0.3f | Weight for center proximity in contour scoring |
| `scoreAreaRatioWeight` | `Float` | 0.2f | Weight for area ratio in contour scoring |

### `DetectionParameters`

Runtime metrics reported back from the active detector via `StateFlow`. All values are `String` for direct UI display.

| Field | Type | Source |
|---|---|---|
| `detectorName` | `String` | Name of the detector that produced the result |
| `claheClipLimit` | `String` | Effective CLAHE clip limit used |
| `cannyHigh` | `String` | Effective Canny high threshold used |
| `cannyLow` | `String` | Effective Canny low threshold used |
| `brightness` | `String` | Input image brightness metric |
| `maskThreshold` | `String` | Segmentation mask threshold (ONNX detectors) |
| `heatmapThreshold` | `String` | Heatmap corner regression threshold (HeatmapCornerDetector) |
| `cornerScore` | `String` | Corner keypoint confidence score (ONNX detectors) |
| `cornerError` | `String` | Corner keypoint fit error (ONNX detectors) |

### `IntermediateBitmaps`

Annotated with `@Stable` because it holds mutable `Bitmap?` references. Carries optional Bitmap snapshots for each stage of the detection pipeline. Populated by the data layer and consumed by the presenter for debugging/preview UI.

| Field | Type | Source |
|---|---|---|
| `blur` | `Bitmap?` | Median blur stage |
| `clahe` | `Bitmap?` | CLAHE contrast enhancement |
| `morph` | `Bitmap?` | Morphological close result |
| `edges` | `Bitmap?` | Edge detection (Canny) output |
| `mask` | `Bitmap?` | Segmentation mask (ONNX) or binary threshold |
| `corners` | `Bitmap?` | Corner keypoint detection output |

## Dependency Flow

```
presenter (ViewModels, Screens)
    └── depends on → domain (repository interfaces)
                         └── depends on → entity (shared data classes)

data (RepositoryImpl, Detectors)
    └── implements → domain (repository interfaces)
    └── depends on   → entity (shared data classes)
    └── depends on   → OpenCV, ONNX, Android (infrastructure)
```

The domain layer must **never** import from `data/` or `presenter/`. Imports of `org.opencv.*` and `android.graphics.Bitmap` in domain interfaces are accepted as pragmatic exceptions — the interface defines the contract, and the types are passed through without the domain layer reasoning about them.

## Adding to the Domain Layer

### Adding a new repository interface

1. Create `repository/I<UseCase>Repository.kt` in `domain/repository/`
2. Define methods that express business intent (not implementation details)
3. Create `data/repository/<UseCase>RepositoryImpl.kt` implementing the interface
4. Bind in `Di.kt`: `single<I<UseCase>Repository> { <UseCase>RepositoryImpl(...) }`
5. Inject into the relevant ViewModel via constructor

### Adding a new shared entity

1. Create the data class in `entity/` with default parameter values
2. Annotate with `@Immutable` for pure data (no mutable references)
3. Annotate with `@Stable` if the class holds mutable references (e.g., `Bitmap?`)
4. Prefer pure Kotlin types — if an Android or OpenCV type is unavoidable, document the rationale

## Dependency Injection

Domain repositories are bound in `Di.kt` and injected into ViewModels:

```kotlin
single<IDocumentDetectorRepository> {
    DocumentDetectorRepositoryImpl(get<IDocumentDetector>(named("combined")))
}
```

The presenter layer depends only on the interface (`IDocumentDetectorRepository`), never on the implementation (`DocumentDetectorRepositoryImpl`).
