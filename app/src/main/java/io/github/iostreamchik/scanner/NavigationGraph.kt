package io.github.iostreamchik.scanner

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import org.koin.androidx.compose.koinViewModel
import org.koin.core.qualifier.named
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import io.github.iostreamchik.scanner.camera.CameraScreen
import io.github.iostreamchik.scanner.camera.CameraViewModel
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
            val viewModel = koinViewModel<CameraViewModel>(named("camera"))
            CameraScreen(
                modifier = Modifier.fillMaxSize(),
                viewModel = viewModel,

                toScanFromFile = {
                    navController.navigate(NavigationDestination.FileScanResult)
                },
                toOpenSettings = {
                    navController.navigate(NavigationDestination.PipelineSettings)
                }
            )
        }
        composable(NavigationDestination.FileScanResult) {
            val viewModel = koinViewModel<CameraViewModel>(named("fileScan"))
            FileScanResultScreen(
                modifier = Modifier.fillMaxSize(),
                viewModel = viewModel,
                onBack = {
                    navController.popBackStack()
                }
            )
        }
        composable(NavigationDestination.PipelineSettings) {
            val viewModel = koinViewModel<PipelineSettingsViewModel>(named("pipelineSettings"))
            PipelineSettingsScreen(
                modifier = Modifier.fillMaxSize(),
                viewModel = viewModel,
                onBack = {
                    navController.popBackStack()
                }
            )
        }
    }
}
