package io.github.iostreamchik.scanner.pipeline

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay

/**
 * A parameter slider with label, value display, and a Material 3 slider.
 * The value is formatted by [valueFormatter] and displayed next to the label.
 */
@Composable
fun ParameterSlider(
    label: String,
    value: Float,
    valueRange: ClosedFloatingPointRange<Float>,
    step: Float,
    valueFormatter: (Float) -> String,
    onValueChange: (Float) -> Unit,
    enabled: Boolean = true,
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = label,
                fontSize = 13.sp,
                fontWeight = FontWeight.Medium,
                color = Color.Gray
            )
            Text(
                text = valueFormatter(value),
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )
        }
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = valueRange,
            steps = if (step > 0) ((valueRange.endInclusive - valueRange.start) / step).toInt() else 0,
            modifier = Modifier.fillMaxWidth(),
            enabled = enabled,
            colors = SliderDefaults.colors(
                activeTickColor = MaterialTheme.colorScheme.primary,
                inactiveTickColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
            )
        )
    }
}

/**
 * Debounces an integer value for the given number of milliseconds.
 * Returns the most recent value after the delay period.
 */
@Composable
fun debounceInt(value: Int, delayMs: Long): Int {
    var debouncedValue by remember { mutableIntStateOf(value) }
    LaunchedEffect(value) {
        delay(delayMs)
        debouncedValue = value
    }
    return debouncedValue
}

/**
 * Debounces a float value for the given number of milliseconds.
 */
@Composable
fun debounceFloat(value: Float, delayMs: Long): Float {
    var debouncedValue by remember { mutableFloatStateOf(value) }
    LaunchedEffect(value) {
        delay(delayMs)
        debouncedValue = value
    }
    return debouncedValue
}

/**
 * Small chip showing a label and numeric value (e.g. brightness/contrast).
 */
@Composable
fun InfoChip(label: String, value: String) {
    Surface(
        modifier = Modifier.padding(horizontal = 4.dp),
        shape = RoundedCornerShape(16.dp),
        color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.4f)
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(4.dp),
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
        ) {
            Text(
                text = label,
                fontSize = 12.sp,
                color = Color.Gray
            )
            Text(
                text = value,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onPrimaryContainer
            )
        }
    }
}

@Preview
@Composable
private fun PipelineControlsPreview() {
    Surface {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            ParameterSlider(
                label = "Kernel Size",
                value = 5f,
                valueRange = 3f..21f,
                step = 2f,
                valueFormatter = { "${it.toInt()}" },
                onValueChange = {}
            )
            ParameterSlider(
                label = "Clip Limit",
                value = 0.8f,
                valueRange = 0.1f..5.0f,
                step = 0.1f,
                valueFormatter = { "%.1f".format(it) },
                onValueChange = {}
            )
            InfoChip(label = "Brightness", value = "128")
            InfoChip(label = "Contrast", value = "45")
        }
    }
}
