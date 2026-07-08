package io.github.iostreamchik.scanner

import io.github.iostreamchik.scanner.camera.CameraViewModel
import io.github.iostreamchik.scanner.opencv.CannyThresholdCalculatorV3
import io.github.iostreamchik.scanner.opencv.ICannyThresholdCalculator
import io.github.iostreamchik.scanner.opencv.IMatBundle
import io.github.iostreamchik.scanner.opencv.MatBundle
import org.koin.android.ext.koin.androidContext
import org.koin.android.ext.koin.androidLogger
import org.koin.core.context.startKoin
import org.koin.core.logger.Level
import org.koin.core.module.dsl.viewModel
import org.koin.dsl.module

val appModule = module {
    single<IMatBundle> { MatBundle() }
    single<ICannyThresholdCalculator> { CannyThresholdCalculatorV3(get()) }
    single<IDocumentDetector> {
        OnnxDocumentDetector(
            get(),
            get(),
            "onnx/deeplabv3_mbv3_docseg.onnx",
            useCustomNormalization = true
        )
    }
    viewModel { CameraViewModel(matBundle = get(), thresholdCalculator = get(), detector = get()) }
}

fun initKoin(context: android.content.Context) {
    startKoin {
        androidLogger(Level.INFO)
        androidContext(context)
        modules(appModule)
    }
}
