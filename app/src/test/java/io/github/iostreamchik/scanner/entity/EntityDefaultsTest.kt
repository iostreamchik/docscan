package io.github.iostreamchik.scanner.entity

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class EntityDefaultsTest {

    @Test
    fun pipelineParamsWeightsSumToOne() {
        val params = PipelineParams()
        val sum = params.scoreAreaWeight + params.scoreCenterWeight + params.scoreAreaRatioWeight
        assertEquals(1.0f, sum, 1e-6f)
    }

    @Test
    fun pipelineParamsDefaultsPositive() {
        val params = PipelineParams()
        assertTrue(params.medianBlurKsize > 0)
        assertTrue(params.medianBlurKsize % 2 == 1)
        assertTrue(params.claheClipLimit > 0f)
        assertTrue(params.claheTileSize > 0)
        assertTrue(params.morphCloseSize > 0)
        assertTrue(params.strongCloseSize > 0)
        assertTrue(params.directionalKernelSize > 0)
        assertTrue(params.approxPolyDPTolerance > 0f)
        assertTrue(params.minAreaFraction > 0f)
        assertTrue(params.scoreAreaWeight > 0f)
        assertTrue(params.scoreCenterWeight > 0f)
        assertTrue(params.scoreAreaRatioWeight > 0f)
    }

    @Test
    fun pipelineParamsCopyOverridesSingleField() {
        val original = PipelineParams()
        val copy = original.copy(medianBlurKsize = 7)
        assertEquals(7, copy.medianBlurKsize)
        assertEquals(original.claheClipLimit, copy.claheClipLimit)
        assertEquals(original.scoreAreaWeight, copy.scoreAreaWeight)
    }

    @Test
    fun detectionParametersDefaultsEmpty() {
        val params = DetectionParameters()
        assertTrue(params.detectorName.isEmpty())
        assertTrue(params.claheClipLimit.isEmpty())
        assertTrue(params.cannyHigh.isEmpty())
        assertTrue(params.cannyLow.isEmpty())
        assertTrue(params.brightness.isEmpty())
        assertTrue(params.maskThreshold.isEmpty())
        assertTrue(params.heatmapThreshold.isEmpty())
        assertTrue(params.cornerScore.isEmpty())
        assertTrue(params.cornerError.isEmpty())
    }

    @Test
    fun intermediateBitmapsDefaultsNull() {
        val snapshots = IntermediateBitmaps()
        assertTrue(snapshots.blur == null)
        assertTrue(snapshots.clahe == null)
        assertTrue(snapshots.morph == null)
        assertTrue(snapshots.edges == null)
        assertTrue(snapshots.mask == null)
        assertTrue(snapshots.corners == null)
    }

    @Test
    fun intermediateBitmapsEquality() {
        assertEquals(IntermediateBitmaps(), IntermediateBitmaps())
        assertEquals(IntermediateBitmaps().copy(), IntermediateBitmaps())
    }

    @Test
    fun detectionParametersEquality() {
        assertEquals(DetectionParameters(), DetectionParameters())
        assertEquals(
            DetectionParameters(detectorName = "Minimal"),
            DetectionParameters(detectorName = "Minimal")
        )
    }
}
