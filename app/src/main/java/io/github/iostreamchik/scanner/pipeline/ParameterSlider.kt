package io.github.iostreamchik.scanner.pipeline

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.sp

/**
 * A slider UI component for pipeline parameters, showing a label and formatted value.
 */
@Composable
fun ParameterSlider(
    label: String,
    value: Float,
    valueRange: ClosedFloatingPointRange<Float>,
    step: Float,
    valueFormatter: (Float) -> String,
    onValueChange: (Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    val steps = if (step > 0f) {
        ((valueRange.endInclusive - valueRange.start) / step).toInt()
    } else 0

    Column(modifier = modifier.fillMaxWidth()) {
        Text(
            text = "$label: ${valueFormatter(value)}",
            fontSize = 12.sp,
        )
        Slider(
            value = value,
            valueRange = valueRange,
            onValueChange = onValueChange,
            steps = steps,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}
