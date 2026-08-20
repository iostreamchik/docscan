package io.github.iostreamchik.scanner.presenter.navigation

import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import io.github.iostreamchik.scanner.presenter.camera.CameraScreen
import io.github.iostreamchik.scanner.presenter.camera.CameraViewModel
import io.github.iostreamchik.scanner.presenter.filescan.FileScanResultScreen
import org.koin.androidx.compose.koinViewModel


object NavigationDestination {
    const val camera = "camera"
    const val fileScanResult = "file_scan_result"
}

@Composable
fun AppNavGraph(
    modifier: Modifier = Modifier,
    startDestination: String = NavigationDestination.camera,
) {
    val navController = rememberNavController()

    NavHost(
        navController = navController,
        startDestination = startDestination,
        modifier = modifier.fillMaxSize()
    ) {
        composable(
            NavigationDestination.camera,
            enterTransition = {
                slideInVertically { -it } + fadeIn()
            },
            exitTransition = {
                slideOutVertically { -it } + fadeOut()
            }
        ) {
            val viewModel = koinViewModel<CameraViewModel>()
            CameraScreen(
                modifier = Modifier.fillMaxSize(),
                viewModel = viewModel,

                toScanFromFile = {
                    navController.navigate(NavigationDestination.fileScanResult)
                }
            )
        }
        composable(
            NavigationDestination.fileScanResult,
            enterTransition = {
                slideInVertically { it } + fadeIn()
            },
            exitTransition = {
                slideOutVertically { it } + fadeOut()
            }
        ) {
            val viewModel = koinViewModel<CameraViewModel>()
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
