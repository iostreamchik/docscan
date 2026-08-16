# AGENTS.md

**DocumentScanner** — a Kotlin/Jetpack Compose Android app that detects document boundaries in camera previews and photos using OpenCV contour analysis and ONNX models (LCNet heatmap corner regression, LCNet corner keypoints, DeepLabV3 semantic segmentation), then applies perspective warp to produce clean scanned images.

**Min SDK:** 26 | **Compile SDK:** 37 | **Target SDK:** 36
**Kotlin:** 2.4.0 | **AGP:** 9.2.1 | **OpenCV:** 5.0.0.1 | **ONNX Runtime:** 1.27.0 | **CameraX:** 1.6.1 | **Koin:** 4.2.2

## Common Commands

```bash
# Build debug APK
./gradlew assembleDebug

# Build release APK
./gradlew assembleRelease

# Run unit tests
./gradlew testDebugUnitTest

# Run instrumented tests (requires device/emulator)
./gradlew connectedAndroidTest

# Open in Android Studio
open app/
```

## Architecture: Clean Architecture (4 Layers)

```
entity   — pure Kotlin data classes (PipelineParams, DetectionParameters, IntermediateBitmaps)
domain   — repository interfaces (IDocumentDetectorRepository)
data     — detector implementations, OpenCV infrastructure, utils, repository impl
presenter— ViewModels, Composables, navigation, theme
```

Dependency flow: **Presenter → Domain ← Data**, all layers may use **Entity**.

### Package-Level AGENTS.md Files

Each layer maintains its own `AGENTS.md` — **read the relevant one before making changes — do not perform a full code review of the package itself.**

| Layer | Path |
|---|---|
| `entity/` | `app/src/main/java/io/github/iostreamchik/scanner/entity/AGENTS.md` |
| `domain/` | `app/src/main/java/io/github/iostreamchik/scanner/domain/AGENTS.md` |
| `data/` | `app/src/main/java/io/github/iostreamchik/scanner/data/AGENTS.md` |
| `presenter/` | `app/src/main/java/io/github/iostreamchik/scanner/presenter/AGENTS.md` |

## Source Layout

```
io.github.iostreamchik.scanner/
├── ScannerApp.kt                        — Application class (OpenCV init + Koin)
├── Di.kt                                — Koin module with all bindings
│
├── entity/                              — Pure data classes
│   ├── PipelineParams.kt
│   ├── DetectionParameters.kt
│   └── IntermediateBitmaps.kt           — Bitmap snapshots of detection pipeline stages
│
├── domain/                              — Business logic contracts
│   └── repository/IDocumentDetectorRepository.kt
│
├── data/                                — Detection backends + OpenCV infra
│   ├── detector/                        — 10 files: detectors + interface + orchestrator + helpers
│   │   ├── IDocumentDetector.kt
│   │   ├── CombinedDocumentDetector.kt  — Orchestrator (parallel Minimal+Directional, then Heatmap/CornerKeypoint/Segmentation fallbacks)
│   │   ├── DocumentDetectorMinimal.kt
│   │   ├── DocumentDetectorDirectionalSuppression.kt
│   │   ├── HeatmapCornerDetector.kt     — LCNet100+BiFPN ONNX (256px, heatmap corner regression)
│   │   ├── CornerKeypointDetector.kt    — LCNet ONNX (256px, corner keypoints)
│   │   ├── SegmentationDetector.kt      — DeepLabV3 ONNX (384px, semantic seg)
│   │   ├── OnnxSessionManager.kt        — ONNX Runtime session lifecycle + tensor preparation
│   │   ├── ClassicalDetectorExtensions.kt — Shared intermediate snapshot capture for classical detectors
│   │   └── MockDocumentDetector.kt
│   ├── opencv/                          — Mat pooling + OpenCV utilities
│   │   ├── MatBundle.kt (contains IMatBundle interface) / MockMatBundle.kt
│   │   └── OpenCVAdapter.kt
│   ├── repository/
│   │   └── DocumentDetectorRepositoryImpl.kt  — IDocumentDetector → IDocumentDetectorRepository adapter
│   └── utils/
│       ├── Extensions.kt                — ImageProxy→Mat, Mat→Bitmap, warp, enhance, quad helpers
│       ├── QuadGeometry.kt              — sortQuadPoints, quadDistance, quadHash, isRectangle, computeAngle
│       └── ContourScoring.kt            — scoreContourWithParams
│
└── presenter/                           — UI layer
    ├── MainActivity.kt
    ├── navigation/NavigationGraph.kt    — 2 destinations: Camera, FileScan
    ├── camera/                          — Live camera detection screen
    │   ├── CameraScreen.kt
    │   ├── CameraViewModel.kt           — Frame processing, quad fusion/stability, warping
    │   ├── CameraState.kt / CameraIntent.kt / ContourData.kt
    ├── filescan/FileScanResultScreen.kt — Photo picker + 6 intermediate bitmaps
    ├── composables/                     — BitmapCard, ContourCanvas, DeviceCornerRadius
    └── theme/                           — Color.kt, Theme.kt, Type.kt
```

## Detection Pipeline

Five detectors implement `IDocumentDetector`, orchestrated by `CombinedDocumentDetector`:

1. **Minimal** and **DirectionalSuppression** run in parallel (classical OpenCV pipelines)
2. Best result selected by max angle deviation from 90°
3. **HeatmapCornerDetector** (LCNet100+BiFPN ONNX) as first fallback
4. **CornerKeypointDetector** (LCNet ONNX) as second fallback
5. **SegmentationDetector** (DeepLabV3 ONNX) as final fallback

Images are scaled to 448px max dimension for classical detectors. ONNX models are stored in `assets/onnx/`.

## Key Patterns

- **No inline comments** — code must be self-documenting, clean, and minimal
- **Mat pooling** — all OpenCV Mats go through `IMatBundle`/`MatBundle`. Never allocate Mats in hot paths. Release in `finally` blocks or `onCleared()`.
- **Bitmap safety** — clone bitmaps before emitting to `StateFlow`. Never use `remember(bitmap)` as a key. Never recycle bitmaps in `onCleared`.
- **State management** — all UI state via `StateFlow`, collected with `collectAsStateWithLifecycle`.
- **Unidirectional Data Flow** — Intent → ViewModel → State → Composable.
- **OpenCV 5 APIs** — uses `org.opencv.geometry.Geometry` for geometry operations.
- **DI** — Koin (runtime, no annotation processing). Named bindings for ViewModel instances and detector variants.

## Koin Named Bindings

| Name | Type | Usage |
|---|---|---|
| `"combined"` | `IDocumentDetector` | CombinedDocumentDetector (used by repository) |
| `"minimal"` | `IDocumentDetector` | DocumentDetectorMinimal |
| `"directionalSuppression"` | `IDocumentDetector` | DocumentDetectorDirectionalSuppression |
| `"heatmapCorner"` | `IDocumentDetector` | HeatmapCornerDetector |
| `"cornerKeypoint"` | `IDocumentDetector` | CornerKeypointDetector |
| `"segmentation"` | `IDocumentDetector` | SegmentationDetector (DeepLabV3) |

`CameraViewModel` is registered without a named binding (single `viewModel {}` declaration used by both screens).

## AndroidManifest Paths

- **Application:** `.ScannerApp`
- **Activity:** `.presenter.MainActivity`
