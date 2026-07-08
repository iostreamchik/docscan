package io.github.iostreamchik.scanner

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import io.github.iostreamchik.scanner.camera.CameraScreen
import io.github.iostreamchik.scanner.camera.CameraViewModel
import io.github.iostreamchik.scanner.local_files.FileScanResultScreen
import io.github.iostreamchik.scanner.pipeline.PipelineSettingsScreen
import io.github.iostreamchik.scanner.pipeline.PipelineSettingsViewModel
import org.koin.compose.koinInject

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
            val matBundle = koinInject<io.github.iostreamchik.scanner.opencv.IMatBundle>()
            val thresholdCalculator = koinInject<io.github.iostreamchik.scanner.opencv.ICannyThresholdCalculator>()
            val detector = koinInject<io.github.iostreamchik.scanner.IDocumentDetector>()
            val viewModel = viewModel<CameraViewModel>(key = "camera") {
                CameraViewModel(matBundle = matBundle, thresholdCalculator = thresholdCalculator, detector = detector)
            }
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
            val matBundle = koinInject<io.github.iostreamchik.scanner.opencv.IMatBundle>()
            val thresholdCalculator = koinInject<io.github.iostreamchik.scanner.opencv.ICannyThresholdCalculator>()
            val detector = koinInject<io.github.iostreamchik.scanner.IDocumentDetector>()
            val viewModel = viewModel<CameraViewModel>(key = "fileScan") {
                CameraViewModel(matBundle = matBundle, thresholdCalculator = thresholdCalculator, detector = detector)
            }
            FileScanResultScreen(
                modifier = Modifier.fillMaxSize(),
                viewModel = viewModel,
                onBack = {
                    navController.popBackStack()
                }
            )
        }
        composable(NavigationDestination.PipelineSettings) {
            val matBundle = koinInject<io.github.iostreamchik.scanner.opencv.IMatBundle>()
            val detector = koinInject<io.github.iostreamchik.scanner.IDocumentDetector>()
            val viewModel = viewModel<PipelineSettingsViewModel>(key = "pipelineSettings") {
                PipelineSettingsViewModel(matBundle = matBundle, detector = detector)
            }
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
