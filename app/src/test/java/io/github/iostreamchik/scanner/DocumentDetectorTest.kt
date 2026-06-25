package io.github.iostreamchik.scanner

import org.junit.Test

class DocumentDetectorTest {

    @Test
    fun computeAutoClaheClipLimit() {
        // Dim: strong boost
        assert(DocumentDetector.computeAutoClaheClipLimit(40.0) > 3.0)
        // Mid-range: baseline
        assert(DocumentDetector.computeAutoClaheClipLimit(100.0) == 1.5)
        // Bright: boosted
        assert(DocumentDetector.computeAutoClaheClipLimit(180.0) > 2.5)
        // Caps at 4.0
        assert(DocumentDetector.computeAutoClaheClipLimit(250.0) <= 4.0)
    }

}