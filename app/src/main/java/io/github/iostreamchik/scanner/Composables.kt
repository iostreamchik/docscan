package io.github.iostreamchik.scanner

import android.os.Build
import android.view.RoundedCorner
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

@Composable
fun rememberDeviceCornerRadiusDp(
    defaultValue: Dp = 24.dp
): Dp {
    val view = LocalView.current
    val density = LocalDensity.current

    return remember {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val insets = view.rootWindowInsets
            val corner = insets?.getRoundedCorner(RoundedCorner.POSITION_TOP_LEFT)
            corner?.radius?.let { radiusPx ->
                with(density) { radiusPx.toDp() }
            } ?: defaultValue
        } else {
            defaultValue
        }
    }
}
