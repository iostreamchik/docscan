package io.github.iostreamchik.scanner.pipeline

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import kotlinx.coroutines.delay

/**
 * Debounces an integer value: returns the value only after [delayMs] milliseconds
 * of no changes. Useful for avoiding excessive param updates during slider drags.
 */
@Composable
fun debounceInt(initialValue: Int, delayMs: Long): Int {
    var debouncedValue by remember { mutableIntStateOf(initialValue) }
    LaunchedEffect(initialValue, delayMs) {
        delay(delayMs)
        debouncedValue = initialValue
    }
    return debouncedValue
}

/**
 * Debounces a float value: returns the value only after [delayMs] milliseconds
 * of no changes. Useful for avoiding excessive param updates during slider drags.
 */
@Composable
fun debounceFloat(initialValue: Float, delayMs: Long): Float {
    var debouncedValue by remember { mutableFloatStateOf(initialValue) }
    LaunchedEffect(initialValue, delayMs) {
        delay(delayMs)
        debouncedValue = initialValue
    }
    return debouncedValue
}
