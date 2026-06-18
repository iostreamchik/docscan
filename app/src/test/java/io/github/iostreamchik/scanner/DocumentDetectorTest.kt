package io.github.iostreamchik.scanner

import io.github.iostreamchik.scanner.opencv.CannyThresholdCalculator
import io.github.iostreamchik.scanner.opencv.MockMatBundle
import org.junit.Test

class DocumentDetectorTest {

    val detector  = DocumentDetector(
        matBundle = MockMatBundle(),
        thresholdCalculator = CannyThresholdCalculator(
            matBundle = MockMatBundle()
        )
    )

    @Test
    fun calculateClacheClipLimit() {
        val brightnesses = arrayOf(60.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0)
        brightnesses.forEachIndexed { index, item ->
            println("brightness: $item -> ${detector.calculateClacheClipLimit(item)}")
        }
    }

}