package io.github.iostreamchik.scanner

import androidx.activity.ComponentActivity
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import io.github.iostreamchik.scanner.camera.CameraScreen
import io.github.iostreamchik.scanner.local_files.FileScanResultScreen
import io.github.iostreamchik.scanner.pipeline.PipelineSettingsScreen
import io.github.iostreamchik.scanner.pipeline.PipelineSettingsViewModel

object NavigationDestination {
    const val Camera = "camera"
    const val FileScanResult = "file_scan_result"
    const val PipelineSettings = "pipeline_settings"
}

@Composable
fun AppNavGraph(
    modifier: Modifier = Modifier,
    startDestination: String = NavigationDestination.Camera,
) {
    val navController = rememberNavController()

    NavHost(
        navController = navController,
        startDestination = startDestination,
        modifier = modifier.fillMaxSize()
    ) {
        composable(NavigationDestination.Camera) {
            val context = LocalContext.current
            CameraScreen(
                modifier = Modifier.fillMaxSize(),
                viewModel = viewModel(viewModelStoreOwner = (context as? ComponentActivity) ?: it),
                toScanFromFile = {
                    navController.navigate(NavigationDestination.FileScanResult)
                },
                toOpenSettings = {
                    navController.navigate(NavigationDestination.PipelineSettings)
                }
            )
        }
        composable(NavigationDestination.FileScanResult) {
            val context = LocalContext.current
            FileScanResultScreen(
                modifier = Modifier.fillMaxSize(),
                viewModel = viewModel(viewModelStoreOwner = (context as? ComponentActivity) ?: it),
                onBack = {
                    navController.popBackStack()
                }
            )
        }
        composable(NavigationDestination.PipelineSettings) {
            val context = LocalContext.current
            PipelineSettingsScreen(
                modifier = Modifier.fillMaxSize(),
                viewModel = viewModel<PipelineSettingsViewModel>(
                    key = "pipelineSettings",
                    viewModelStoreOwner = (context as? ComponentActivity) ?: it
                ),
                onBack = {
                    navController.popBackStack()
                }
            )
        }
    }
}
