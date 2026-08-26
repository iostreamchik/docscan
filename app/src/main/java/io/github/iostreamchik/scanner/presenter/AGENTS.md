# presenter — AGENTS.md

UI presentation layer for **DocumentScanner**. Owns all Jetpack Compose screens, ViewModels, navigation, shared composables, and theme. Follows Unidirectional Data Flow (UDF): **Intent → ViewModel → State → Composable**.

## Package Layout

```
presenter/
├── MainActivity.kt                  — Entry activity, edge-to-edge, Scaffold + AppNavGraph host
├── navigation/
│   └── NavigationGraph.kt           — NavigationDestination object + AppNavGraph (Camera, FileScanResult)
├── camera/
│   ├── CameraScreen.kt              — CameraX preview, permission flow, contour overlay, torch,
│   │                                 detection params panel, bottom previews
│   ├── CameraIntent.kt              — Sealed class of user/system intents
│   ├── CameraState.kt               — @Stable UI state data class
│   ├── CameraViewModel.kt           — Frame processing, quad fusion/stability, warping, file scan
│   └── ContourData.kt               — @Stable contour rendering data (contours + frame dims)
├── filescan/
│   └── FileScanResultScreen.kt      — Photo picker, intermediate bitmaps grid, modal bottom sheet,
│   │                                 detector chip, processing state
├── composables/
│   ├── BitmapCard.kt                — Animated or static bitmap display with fade transitions
│   ├── ContourCanvas.kt             — Canvas overlay for rotated/scaled contour drawing
│   └── DeviceCornerRadius.kt        — rememberDeviceCornerRadiusDp() via WindowInsets
└── theme/
    ├── Color.kt                     — BlueLight/BlueDark palettes + CameraBackground
    ├── Theme.kt                     — DocumentScannerTheme (dark/light/dynamic color)
    └── Type.kt                      — Typography (bodyLarge default)
```

## Architecture

### Unidirectional Data Flow

```
User Action / Camera Frame
        │
        ▼
CameraIntent (sealed class)
        │
        ▼
CameraViewModel.process(intent)
        │
        ▼
MutableStateFlow<CameraState>
        │
        ▼
collectAsStateWithLifecycle() in Composable
        │
        ▼
Recomposition → UI update
```

### Two Screens

| Screen | ViewModel | Navigation |
|---|---|---|
| `CameraScreen` | `CameraViewModel` via `koinViewModel` | Start destination |
| `FileScanResultScreen` | `CameraViewModel` via `koinViewModel` | Navigated from Camera FAB |

Both screens use `koinViewModel<CameraViewModel>()` without named qualifiers. Separate ViewModel instances result from NavHost's per-composable composition scoping.

### Intent Types (`CameraIntent`)

| Intent | Purpose |
|---|---|
| `ToggleTorch` | Flip torch on/off |
| `SetTorch(on)` | Set explicit torch state (from TorchState observer) |
| `SetError(messageId)` | Show/clear error surface (string resource ID, `Int?`) |
| `UpdateParams(params)` | Replace pipeline params for next detection |
| `ProcessDocument(context, uri, onComplete)` | Decode image → run detection → emit result |

### State Shape (`CameraState`)

| Field | Type | Description |
|---|---|---|
| `intermediateBitmaps` | `IntermediateBitmaps` | 6 optional bitmaps (blur, clahe, morph, edges, mask, corners) |
| `originalBitmap` | `Bitmap?` | Source frame or picked photo |
| `resultBitmap` | `Bitmap?` | Warped scanned document or enhanced original |
| `torchOn` | `Boolean` | Current torch state |
| `exposure` | `String` | Exposure info (reserved) |
| `errorId` | `Int?` | String resource ID for error message |
| `isProcessing` | `Boolean` | Loading state for file scan |
| `pipelineParams` | `PipelineParams` | Current detection parameters |

## Key Patterns

### No Inline Comments

Code must be self-documenting. Never add inline comments, block comments, or docstrings.

### Bitmap Safety

- **Clone before emit**: Every bitmap emitted to `StateFlow` is `.copy(Bitmap.Config.ARGB_8888, false)` so Compose owns an independent copy.
- **Never `remember(bitmap)` as a key**: Bitmap recycling creates race conditions. Use `key(hasImage)` or content-derived keys instead.
- **Never recycle bitmaps in `onCleared`**: Let GC handle it. Clear state flows with `null` is sufficient.

### State Management

- All UI state flows through `StateFlow` in ViewModels.
- Collect with `collectAsStateWithLifecycle()` — never raw `collectAsState()`.
- Use `setState { copy(...) }` helper for immutable state transitions.
- `@Stable` annotation on `CameraState` and `ContourData` for optimized recomposition.

### CameraX Integration

- `ImageAnalysis` uses `STRATEGY_KEEP_ONLY_LATEST` to drop stale frames.
- Processing runs on `cameraExecutor` (dedicated single-thread executor).
- Resolution targets 2000×2000 with `FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER` on both Preview and ImageAnalysis.
- Camera bound inside `LaunchedEffect(previewView)` with `cameraProvider.unbindAll()` before binding.
- `PROCESS_WIDTH = 448.0` — all frames scaled to 448px max dimension before detection.

### Permission Flow

`CameraScreen` handles camera permission with three states:
1. **Granted** — camera preview starts normally
2. **Rationale shown** — "Grant" button re-prompts permission
3. **Permanently denied** — "Open Settings" button launches app settings

### Contour Rendering Throttle

`CameraScreen` throttles `ContourData` updates to **30ms intervals** (`CONTOUR_UPDATE_THROTTLE_MS`) to prevent excessive Compose recomposition from high-frame-rate camera input.

### Quad Stability & Fusion

`CameraViewModel` maintains a 4-frame quad history:
- **Stability check** every frame: average corner movement < 2% of frame diagonal.
- **Fusion**: Average corner positions across all history entries when stable.
- **Hash dedup**: Skip warping when quad hash hasn't changed (saves expensive perspective transform). Recycles previous warped bitmap on hash hit.
- History stores pure Kotlin `List<Point>` snapshots (sorted via `sortQuadPoints()` at insertion) — never native Mats, since the UI releases the same quad instances it receives via `detectedQuads`.
- Detection work runs in `viewModelScope.launch { withContext(Dispatchers.Default) { ... }}` with `currentDetectionJob` tracking for cancellation of in-flight jobs on new frames.

### Detection Params Panel

`CameraScreen` shows real-time detection telemetry:
- Detector name from `detectionParams.detectorName`
- CLAHE clip limit, Canny thresholds, brightness — only for classical detectors
- ONNX detectors (`AsyncDetectorSource.HEATMAP_CORNER`, `CORNER_KEYPOINT`, `SEGMENTATION`) show "N/A" via `derivedStateOf` check against `detectionParamsName`

### Bottom Preview Panes

`CameraScreen` shows two bottom preview cards:
1. **Left**: `BitmapCard` displaying `mask` bitmap (falls back to `edges`)
2. **Right**: `Image` with remembered `ImageBitmap` from `resultBitmap` (warped document or enhanced original)

### File Scan Screen

`FileScanResultScreen` features:
- **Empty state**: Icon + "Select File" prompt with FAB and inline Button
- **Processing state**: Shows "Processing" text while detection runs
- **Results grid**: Intermediate bitmaps in 2-column rows with step labels (edges hidden when mask/corners present)
- **Original + Detected**: Side-by-side at the bottom ("Detected" shows warped result)
- **Modal bottom sheet**: Tap any bitmap to view full-size with localized step name
- **Detector chip**: TopAppBar shows current detector name from `detectionParams` flow
- Uses `key(hasImage)` to separate empty/loaded states, `aspectRatio` derived from original bitmap

### Composable Previews

- Use `LocalInspectionMode.current` to branch preview vs. runtime behavior.
- Preview ViewModels wired with `MockDocumentDetector` via `DocumentDetectorRepositoryImpl`.
- `BitmapCard` supports `animated` mode (fade transitions + size-stable crop) and static mode.

### Navigation

- `NavigationGraph.kt` defines `NavigationDestination` object with string routes.
- ViewModels resolved via `koinViewModel<CameraViewModel>()` without named qualifiers.
- `AppNavGraph` wraps in `NavHost` with `rememberNavController()`.

## Dependencies On Other Packages

| Dependency | From | Usage |
|---|---|---|
| `domain.repository.IDocumentDetectorRepository` | `CameraViewModel` | Detection orchestration; provides `detectionParams` flow |
| `entity.PipelineParams` | `CameraState`, `CameraIntent` | Detection configuration |
| `entity.DetectionParameters` | `CameraViewModel`, `CameraScreen` | Runtime detection telemetry (detector name, CLAHE, Canny, brightness) |
| `entity.IntermediateBitmaps` | `CameraState`, `CameraViewModel` | Intermediate stage bitmaps from detector |
| `data.utils.*` | `CameraViewModel` | `toMatRGBA()`, `toBitmap()`, `warpDocumentHighQuality()`, `fixRotation()`, `enhanceDocument()`, `sortQuadPoints()`, `quadDistance()`, `quadHash()` |
| `data.detector.AsyncDetectorSource` | `CameraScreen` | Classical vs ONNX detector branching via `detectionParamsName` (HEATMAP_CORNER, CORNER_KEYPOINT, SEGMENTATION) |
| `data.detector.MockDocumentDetector` | Previews | No-op detector for Compose previews |
| `data.repository.DocumentDetectorRepositoryImpl` | Previews | Adapter wrapping mock detector |

## Adding A New Screen

1. Create subpackage under `presenter/` (e.g., `settings/`)
2. Define `ScreenState` data class (annotate with `@Stable`) with all UI state fields
3. Define `ScreenIntent` sealed class for user actions
4. Create `ScreenViewModel` with `MutableStateFlow<ScreenState>` and `process(intent)` method
5. Create `ScreenScreen.kt` composable that collects state and emits intents
6. Add route to `NavigationDestination` in `NavigationGraph.kt`
7. Add `composable` block with `koinViewModel<ScreenViewModel>()` binding
8. Register ViewModel in `Di.kt` via `viewModel { ScreenViewModel(...) }` (use `named()` only if multiple instances required)

## Adding A New Composable

1. Place in `composables/` if shared across screens, or in screen subpackage if screen-specific
2. Accept `modifier: Modifier = Modifier` as first parameter
3. Use `collectAsStateWithLifecycle()` for any StateFlow collection
4. Provide `@Preview` composable using mock dependencies
5. Handle `LocalInspectionMode.current` for preview-safe rendering
