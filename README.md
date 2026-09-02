# DocumentScanner

An Android document scanner that detects document boundaries in live camera
previews and photos, then applies a perspective warp to produce a clean,
rectangular scan. Detection combines classical OpenCV contour analysis with
on-device ONNX neural models (heatmap corner regression, corner keypoints, and
semantic segmentation) for robustness across a wide range of lighting and
scene conditions.

Built as a learning resource for computer vision, image processing, and
on-device ML on Android.

<p align="center">
  <img src="scanner.png" alt="DocumentScanner" width="280">
</p>

## Features

- **Hybrid detection** – Five detectors behind a single orchestrator:
  two classical OpenCV pipelines plus three ONNX neural backends.
- **Neural fallbacks** – When classical contours under-perform, the pipeline
  falls back to LCNet heatmap corner regression, then DeepLabV3 semantic
  segmentation (an LCNet corner-keypoint detector is also present but disabled
  by default).
- **Live camera scanning** – Real-time quad detection with multi-frame
  stability/fusion before triggering capture.
- **Photo scanning** – Pick an image and inspect intermediate pipeline stages
  as a set of diagnostic bitmaps.
- **Perspective warp** – Homography-based rectification that fills the
  largest dimension and optionally enhances the output for a classic scan look.
- **Memory safety** – All OpenCV `Mat`s flow through a pooled `IMatBundle`;
  no allocations in hot paths, releases in `finally`/`onCleared`.
- **Clean architecture** – Strict `entity → domain ← data → presenter`
  layering with Koin dependency injection.

## Detection Pipeline

Five detectors implement `IDocumentDetector`, orchestrated by
`CombinedDocumentDetector`:

1. **Minimal** and **DirectionalSuppression** run in parallel (classical
   OpenCV pipelines). The winner is the quad whose corners deviate least from
   perfect 90° angles.
2. **HeatmapCornerDetector** (LCNet100 + BiFPN ONNX, 256px) is the first
   neural fallback.
3. **CornerKeypointDetector** (LCNet ONNX, 256px) is the second fallback —
   kept in the codebase as a comparison point but **disabled in release**
   (`skipKeypointDetector = true`) because its large prediction errors made it
   unreliable.
4. **SegmentationDetector** (DeepLabV3 ONNX, 384px) is the final fallback.

Frames are scaled to a 448px max dimension in `CameraViewModel` before being
passed to the detectors. ONNX models ship in `assets/onnx/`:

| Model | File | Backend | Size |
|---|---|---|---|
| LCNet100 + BiFPN | `lcnet100_h_e_bifpn_256_fp32.onnx` | Heatmap corner regression | ~5 MB |
| LCNet050 | `lcnet050_p_multi_decoder_l3_d64_256_fp32.onnx` | Corner keypoints | ~5 MB |
| DeepLabV3 (MBv3) | `deeplabv3_mbv3_docseg.onnx` | Semantic segmentation | 42 MB |

The segmentation model is by far the heaviest; it can be omitted from the
release build if app size must be reduced.

### Why heatmap beats raw keypoints

Both the heatmap and keypoint models work on a 256×256 image, but a small
prediction error in model space is magnified when upscaled to the original
frame. A keypoint model predicts an exact corner, so a ~1.4px error at 256px
becomes ~7px on a 1920×1080 frame. The heatmap model instead predicts a
*region* where each corner is likely; taking the weighted centroid of that
blob is far more accurate after upscaling, which is why it is the primary
neural fallback.

### Multi-frame stability

A single-frame detection can jitter between camera frames. To keep the live
preview stable, the app keeps a sliding window of the last 4 detected quads
and averages each corner across the window, producing a smooth quad that
tracks the document even when individual frames disagree.

## Architecture

Clean Architecture with four layers. Each layer carries its own `AGENTS.md`
describing its conventions.

```
entity    — pure Kotlin data classes (PipelineParams, DetectionParameters,
            IntermediateBitmaps)
domain    — repository interfaces (IDocumentDetectorRepository)
data      — detector implementations, OpenCV infra, utils, repository impl
presenter — ViewModels, Composables, navigation, theme
```

Dependency flow: **Presenter → Domain ← Data**, with every layer allowed to
depend on **Entity**.

```
io.github.iostreamchik.scanner/
├── ScannerApp.kt                        — Application (OpenCV init + Koin)
├── Di.kt                                — Koin module with all bindings
├── entity/                              — PipelineParams, DetectionParameters,
│                                          IntermediateBitmaps
├── domain/repository/                   — IDocumentDetectorRepository
├── data/
│   ├── detector/                        — 5 detectors + CombinedDocumentDetector
│   │                                      + OnnxSessionManager
│   ├── opencv/                          — MatBundle pooling + OpenCVAdapter
│   ├── repository/                      — DocumentDetectorRepositoryImpl
│   └── utils/                           — Extensions, QuadGeometry, ContourScoring
└── presenter/
    ├── MainActivity.kt
    ├── navigation/NavigationGraph.kt    — Camera + FileScan destinations
    ├── camera/                          — Live camera detection screen
    ├── filescan/                        — Photo picker + intermediate bitmaps
    ├── composables/                     — BitmapCard, ContourCanvas, etc.
    └── theme/                           — Color, Theme, Type
```

## Key Patterns

- **Unidirectional Data Flow** – Intent → ViewModel → State → Composable;
  all UI state exposed via `StateFlow` and collected with
  `collectAsStateWithLifecycle`.
- **Mat pooling** – OpenCV `Mat`s are always obtained from and returned to
  `IMatBundle`/`MatBundle`.
- **Bitmap safety** – Bitmaps are cloned before being emitted to `StateFlow`;
  they are never recycled in `onCleared`.
- **DI** – Koin (runtime, no annotation processing) with named bindings for
  each detector variant and the combined orchestrator.

## Tech Stack

| Component | Version |
|---|---|
| Min SDK / Target / Compile | 26 / 36 / 37 |
| Kotlin | 2.4.0 |
| Android Gradle Plugin | 9.2.1 |
| OpenCV | 5.0.0.1 |
| ONNX Runtime | 1.29.0 |
| CameraX | 1.6.1 |
| Koin | 4.2.2 |
| Jetpack Compose | — |

## Model Sources

- [DocsaidLab / DocAligner](https://github.com/DocsaidLab/DocAligner) – LCNet
  heatmap and corner-keypoint detectors.
- [mukund-ks / DeepLabV3-Segmentation](https://github.com/mukund-ks/DeepLabV3-Segmentation) –
  DeepLabV3-MobileNetV3 semantic segmentation.

## Building

```bash
# Debug APK
./gradlew assembleDebug

# Release APK
./gradlew assembleRelease

# Unit tests
./gradlew testDebugUnitTest

# Instrumented tests (requires a device/emulator)
./gradlew connectedAndroidTest
```

## License

[Apache License 2.0](LICENSE) – see the [LICENSE](LICENSE) file.
