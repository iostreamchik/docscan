package io.github.iostreamchik.scanner

/**
 * Document detection pipeline type.
 */
enum class PipelineType {
    /**
     * Canny edge detection pipeline (default).
     * Good for even lighting, high-contrast documents.
     */
    CANNY_OTSU,

    /**
     * Canny edge detection pipeline optimized for light documents on light surfaces.
     * Uses fixed thresholds (15, 30) for better detection in low-contrast scenarios.
     */
    CANNY_LIGHT_DOCS,

    /**
     * Adaptive Gaussian thresholding pipeline.
     * Better for shadows, glare, and uneven lighting.
     */
    ADAPTIVE
}