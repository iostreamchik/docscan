# AGENTS.md — Detector Layer

## Detection Pipeline Architecture

Five detectors implement `IDocumentDetector`, orchestrated by `CombinedDocumentDetector`:

1. **Minimal** and **DirectionalSuppression** — classical OpenCV, run in parallel
2. **HeatmapCornerDetector** — LCNet100+BiFPN ONNX (256px, heatmap corner regression)
3. **CornerKeypointDetector** — LCNet ONNX (256px, corner keypoints) — currently skipped (`skipKeypointDetector = true`)
4. **SegmentationDetector** — DeepLabV3 ONNX (384px, semantic segmentation)

## Memory Management Pattern

All detectors (classical and ONNX) follow the same unified pattern:

- All Mats live in `IMatBundle` (pooled) — including the unpooled raw mat and the segmentation mask
- No bitmap caching — bitmaps are built on-demand from bundle Mats in `captureIntermediateSnapshots()`
- Only small data structures cached on the detector (needed by `detectQuad()`):
  - `HeatmapCornerDetector` — `cachedCorners: List<Point>?`
  - `CornerKeypointDetector` — `cachedCoords: FloatArray?`, `cachedScore: Float`
  - `SegmentationDetector` — none
- Cleanup: `matBundle.releaseAll()` in `release()` — one call handles everything

### Special MatBundle Fields

| Field | Type | Purpose | Pooling |
|---|---|---|---|
| `getRawMat()` | `Mat` | Full-resolution input (cloned per frame) | No — unpooled, released in `releaseAll()` |
| `getSegmentationMask()` | `Mat` | Post-processed segmentation mask | Yes — resized to `scaledWidth x scaledHeight` |

### Data Flow

```
preprocess():
  1. Store raw mat in bundle (clone)
  2. Run inference (classical pipeline or ONNX)
  3. Store intermediate results in bundle (morph, mask, heatmap Mats)
  4. Compute corner data → cached on detector (small, not memory-intensive)

detectQuad():
  1. Read corner data from detector cache
  2. Read mask from bundle (if segmentation)
  3. Return quad

captureIntermediateSnapshots():
  1. Read raw mat from bundle → convert to bitmap
  2. Read overlay data from bundle (heatmap/mask)
  3. Build visualization bitmap on-demand

release():
  matBundle.releaseAll()  // one call, handles everything
```
