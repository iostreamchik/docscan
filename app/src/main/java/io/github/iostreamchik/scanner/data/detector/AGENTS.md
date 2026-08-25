# AGENTS.md — Detector Layer

## Detection Pipeline Architecture

Five detectors implement `IDocumentDetector`, orchestrated by `CombinedDocumentDetector`:

1. **Minimal** and **DirectionalSuppression** — classical OpenCV, run in parallel
2. **HeatmapCornerDetector** — LCNet100+BiFPN ONNX (256px, heatmap corner regression)
3. **CornerKeypointDetector** — LCNet ONNX (256px, corner keypoints)
4. **SegmentationDetector** — DeepLabV3 ONNX (384px, semantic segmentation)

## Memory Management Pattern

### Classical Detectors (Minimal, DirectionalSuppression)
- Use `IMatBundle` (pooled Mats) for all intermediate results
- No per-detector caching — Mats are in the bundle, bitmaps built on-demand
- Cleanup: `matBundle.releaseAll()` in `release()`

### ONNX Detectors (Heatmap, CornerKeypoint, Segmentation) — TARGET STATE
- All Mats → stored in `MatBundle` (including unpooled raw mat and mask)
- Bitmaps → built on-demand from bundle Mats in `captureIntermediateSnapshots()`
- Only small data structures cached on detector: `List<Point>` or `FloatArray`
- Cleanup: `matBundle.releaseAll()` in `release()`

### Current State (TO FIX)

| Detector | Caching Fields | Problem |
|---|---|---|
| HeatmapCornerDetector | `cachedCorners`, `cachedRawMat`, `cachedCornerBitmap`, `cachedRotation` | Raw mat + bitmap caching scattered; bitmap built in preprocess, reused later |
| CornerKeypointDetector | `cachedCoords`, `cachedScore`, `cachedCornerBitmap`, `cachedRawMat`, `cachedRotation` | Same pattern; raw mat also used for edge refinement |
| SegmentationDetector | `cachedMask`, `cachedRawBitmap` | Raw bitmap stored (should derive from raw mat); mask is a Mat that could go in bundle |

### Unified MatBundle Fields (ADD)

| Field | Type | Purpose | Pooling |
|---|---|---|---|
| `getRawMat()` | `Mat` | Full-resolution input (cloned per frame) | No — unpooled, released in `releaseAll()` |
| `getSegmentationMask()` | `Mat` | Post-processed segmentation mask | Yes — resized to `scaledWidth x scaledHeight` |

### Data Flow After Refactor

```
preprocess():
  1. Store raw mat in bundle (clone)
  2. Run ONNX inference
  3. Store results in bundle (mask, heatmap Mats)
  4. Compute corner data → cached on detector (small, not memory-intensive)

detectQuad():
  1. Read corner data from detector cache
  2. Read mask from bundle (if segmentation)
  3. Return quad

captureIntermediateSnapshots():
  1. Read raw mat from bundle → convert to bitmap
  2. Read overlay data from bundle (heatmap/mask)
  3. Build visualization bitmap on-demand
  4. No cached bitmaps needed

release():
  matBundle.releaseAll()  // one call, handles everything
```

### Implementation Steps

1. **Add fields to MatBundle / IMatBundle**
   - `getRawMat(): Mat` — unpooled, `create()` with actual dimensions
   - `getSegmentationMask(): Mat` — pooled, resized to scaled dimensions

2. **Refactor SegmentationDetector**
   - Remove `cachedMask` → use `matBundle.getSegmentationMask()`
   - Remove `cachedRawBitmap` → derive from `matBundle.getRawMat()?.toBitmap()` in snapshots
   - Update `preprocess()` to store mask in bundle
   - Update `buildMaskOverlay()` to use bundle raw mat

3. **Refactor HeatmapCornerDetector**
   - Remove `cachedRawMat` → use `matBundle.getRawMat()`
   - Remove `cachedCornerBitmap` → build from bundle Mats in `captureIntermediateSnapshots()`
   - Store heatmap visualization Mats in bundle (reuse existing `getHeatmapSum/Norm/Colored`)
   - Keep `cachedCorners` (small data, needed for detectQuad)

4. **Refactor CornerKeypointDetector**
   - Remove `cachedRawMat` → use `matBundle.getRawMat()`
   - Remove `cachedCornerBitmap` → build from bundle raw mat + coords in `captureIntermediateSnapshots()`
   - Keep `cachedCoords`, `cachedScore` (small data, needed for detectQuad)

5. **Update CombinedDocumentDetector**
   - No changes needed — it delegates `captureIntermediateSnapshots()` to the active detector

6. **Update ClassicalDetectorExtensions**
   - Already uses bundle — no changes needed

### Benefits
- **One cleanup call** per detector: `matBundle.releaseAll()`
- **No bitmap caching** — bitmaps are built from Mats on demand (same as classical)
- **Consistent API** — all detectors follow the same pattern
- **Fewer fields** — ONNX detectors go from ~5 cached fields to ~2
