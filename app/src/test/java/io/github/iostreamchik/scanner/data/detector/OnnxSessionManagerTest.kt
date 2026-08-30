package io.github.iostreamchik.scanner.data.detector

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class OnnxSessionManagerTest {

    @Test
    fun permutationSizeCorrect() {
        // 256px, 3 channels: 256*256*3 = 196608
        val perm = OnnxSessionManager.computeNchwPermutation(256, 3)
        assertEquals(196608, perm.size)

        // 384px, 3 channels: 384*384*3 = 442368
        val perm384 = OnnxSessionManager.computeNchwPermutation(384, 3)
        assertEquals(442368, perm384.size)
    }

    @Test
    fun permutationIdentityForSingleChannel() {
        // 1 channel: NCHW == NHWC, permutation should map i → i
        val perm = OnnxSessionManager.computeNchwPermutation(4, 1)
        for (i in perm.indices) {
            assertEquals(i.toLong(), perm[i].toLong())
        }
    }

    @Test
    fun permutation2x2Rgb() {
        // 2x2 image, 3 channels: 12 elements
        // NHWC: [R00, G00, B00, R01, G01, B01, R10, G10, B10, R11, G11, B11]
        // NCHW: [R00, R01, R10, R11, G00, G01, G10, G11, B00, B01, B10, B11]
        val perm = OnnxSessionManager.computeNchwPermutation(2, 3)
        val expected = intArrayOf(
            0,  // C0, S00 → NHWC[0]
            3,  // C0, S01 → NHWC[3]
            6,  // C0, S10 → NHWC[6]
            9,  // C0, S11 → NHWC[9]
            1,  // C1, S00 → NHWC[1]
            4,  // C1, S01 → NHWC[4]
            7,  // C1, S10 → NHWC[7]
            10, // C1, S11 → NHWC[10]
            2,  // C2, S00 → NHWC[2]
            5,  // C2, S01 → NHWC[5]
            8,  // C2, S10 → NHWC[8]
            11  // C2, S11 → NHWC[11]
        )
        assertArrayEquals(expected, perm)
    }

    @Test
    fun permutation3x3Rgb() {
        // 3x3 image, 3 channels: 27 elements
        val perm = OnnxSessionManager.computeNchwPermutation(3, 3)
        assertEquals(27, perm.size)

        // Verify first channel: spatial indices 0..8 map to NHWC positions 0, 3, 6, ..., 24
        for (s in 0 until 9) {
            assertEquals(s * 3, perm[s])
        }
        // Second channel: spatial indices 0..8 map to NHWC positions 1, 4, 7, ..., 25
        for (s in 0 until 9) {
            assertEquals(s * 3 + 1, perm[9 + s])
        }
        // Third channel: spatial indices 0..8 map to NHWC positions 2, 5, 8, ..., 26
        for (s in 0 until 9) {
            assertEquals(s * 3 + 2, perm[18 + s])
        }
    }

    @Test
    fun permutationNoDuplicates() {
        // Permutation must be a valid permutation (each index appears exactly once)
        val perm = OnnxSessionManager.computeNchwPermutation(8, 3)
        val sorted = perm.toTypedArray().sorted().toIntArray()
        for (i in sorted.indices) {
            assertEquals(i, sorted[i])
        }
    }

    @Test
    fun permutationNoDuplicates4Channels() {
        // Test with 4 channels (RGBA)
        val perm = OnnxSessionManager.computeNchwPermutation(4, 4)
        assertEquals(64, perm.size)
        val sorted = perm.toTypedArray().sorted().toIntArray()
        for (i in sorted.indices) {
            assertEquals(i, sorted[i])
        }
    }

    @Test
    fun permutationLargeInput() {
        // Verify the common 256px case
        val perm = OnnxSessionManager.computeNchwPermutation(256, 3)
        assertEquals(196608, perm.size)

        // Spot-check: C0, S(0,0) → NHWC[0]; C2, S(255,255) → NHWC[65535*3+2]
        assertEquals(0, perm[0])
        assertEquals(65535 * 3 + 2, perm[2 * 256 * 256 + 65535])

        // Verify no duplicates
        val seen = BooleanArray(perm.size)
        for (v in perm) {
            assertEquals(false, seen[v])
            seen[v] = true
        }
    }
}
