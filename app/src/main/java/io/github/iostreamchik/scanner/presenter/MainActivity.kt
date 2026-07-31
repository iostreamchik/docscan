package io.github.iostreamchik.scanner.presenter

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Scaffold
import androidx.compose.ui.Modifier
import io.github.iostreamchik.scanner.presenter.navigation.AppNavGraph
import io.github.iostreamchik.scanner.presenter.theme.DocumentScannerTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            DocumentScannerTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    innerPadding
                    AppNavGraph(
                        modifier = Modifier
                            .fillMaxSize(),
                    )
                }
            }
        }
    }
}