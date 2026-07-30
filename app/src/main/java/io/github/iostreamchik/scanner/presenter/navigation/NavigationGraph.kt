package io.github.iostreamchik.scanner.presenter.navigation

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import org.koin.androidx.compose.koinViewModel
import org.koin.core.qualifier.named
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import io.github.iostreamchik.scanner.presenter.camera.CameraScreen
import io.github.iostreamchik.scanner.presenter.camera.CameraViewModel
import io.github.iostreamchik.scanner.presenter.filescan.FileScanResultScreen


object NavigationDestination {
    const val Camera = "camera"
    const val FileScanResult = "file_scan_result"
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
                modifier = Modifier
                    .fillMaxSize()
                    .background(MaterialTheme.colorScheme.background),
                viewModel = viewModel,

                toScanFromFile = {
                    navController.navigate(NavigationDestination.FileScanResult)
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
    }
}
