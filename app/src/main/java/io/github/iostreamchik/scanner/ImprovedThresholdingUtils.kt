package io.github.iostreamchik.scanner

import org.opencv.core.Core
import org.opencv.core.MatOfDouble
import org.opencv.imgproc.Imgproc

/**
 * Enhanced utilities for improved thresholding and document detection
 */
object ImprovedThresholdingUtils {
    
    /**
     * Calculate average brightness from a Mat
     */
    fun calculateAverageBrightness(mat: org.opencv.core.Mat): Double {
        val meanStdDev = MatOfDouble()
        val stdDev = MatOfDouble()
        Core.meanStdDev(mat, meanStdDev, stdDev)
        return meanStdDev.toArray()[0]
    }

    /**
     * Calculate contrast from a Mat  
     */
    fun calculateContrast(mat: org.opencv.core.Mat): Double {
        val meanStdDev = MatOfDouble()
        val stdDev = MatOfDouble()
        Core.meanStdDev(mat, meanStdDev, stdDev)
        return stdDev.toArray()[0]
    }

    /**
     * Calculate optimal Canny thresholds based on lighting conditions
     */
    fun calculateOptimalCannyThresholds(
        avgBrightness: Double,
        contrast: Double
    ): Pair<Double, Double> {
        // Determine lighting condition based on brightness and contrast
        val lightCondition = classifyLighting(avgBrightness, contrast)
        
        return when (lightCondition) {
            LightingCondition.BRIGHT -> {
                // For bright images, use higher thresholds to avoid over-detection
                val high = minOf(255.0, maxOf(50.0, avgBrightness * 1.2))
                val low = high * 0.4
                Pair(low, high)
            }
            LightingCondition.DARK -> {
                // For dark images, use lower thresholds to detect more edges
                val high = minOf(200.0, maxOf(30.0, avgBrightness * 1.5))
                val low = high * 0.3
                Pair(low, high)
            }
            LightingCondition.NORMAL -> {
                // For normal conditions, use more adaptive approach with improved sigma
                val sigma = 0.45  // Increased from 0.33 for better sensitivity
                val high = minOf(255.0, maxOf(40.0, avgBrightness * (1.0 + sigma)))
                val low = maxOf(20.0, high * 0.5)
                Pair(low, high)
            }
        }
    }

    /**
     * Classify lighting conditions based on brightness and contrast
     */
    fun classifyLighting(avgBrightness: Double, contrast: Double): LightingCondition {
        // If brightness is very high (over 200), it's likely bright condition
        if (avgBrightness > 200.0) {
            return LightingCondition.BRIGHT
        }
        
        // If brightness is very low (under 50), it's likely dark condition  
        if (avgBrightness < 50.0) {
            return LightingCondition.DARK
        }
        
        // If contrast is very low (under 20), it's likely a uniform or dark image
        if (contrast < 20.0) {
            return LightingCondition.DARK
        }
        
        // Otherwise, normal lighting conditions
        return LightingCondition.NORMAL
    }

    enum class LightingCondition {
        BRIGHT,
        DARK,
        NORMAL
    }
}