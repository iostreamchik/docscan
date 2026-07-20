package io.github.iostreamchik.scanner

import io.github.iostreamchik.scanner.camera.CameraViewModel
import io.github.iostreamchik.scanner.old_detectors.DocumentDetector
import io.github.iostreamchik.scanner.old_detectors.DocumentDetectorOpenCV5
import io.github.iostreamchik.scanner.detector.CombinedDocumentDetector
import io.github.iostreamchik.scanner.detector.DocumentDetectorMinimal
import io.github.iostreamchik.scanner.detector.IDocumentDetector
import io.github.iostreamchik.scanner.detector.OnnxDocumentDetector
import io.github.iostreamchik.scanner.opencv.CannyThresholdCalculatorV3
import io.github.iostreamchik.scanner.pipeline.PipelineSettingsViewModel
import io.github.iostreamchik.scanner.opencv.ICannyThresholdCalculator
import io.github.iostreamchik.scanner.opencv.IMatBundle
import io.github.iostreamchik.scanner.opencv.MatBundle
import org.koin.android.ext.koin.androidContext
import org.koin.android.ext.koin.androidLogger
import org.koin.core.context.startKoin
import org.koin.core.logger.Level
import org.koin.core.module.dsl.viewModel
import org.koin.core.qualifier.named
import org.koin.dsl.module

val appModule = module {
    factory<IMatBundle> { MatBundle() }
    single<ICannyThresholdCalculator> { CannyThresholdCalculatorV3(get()) }
    single<IDocumentDetector> {
        DocumentDetectorOpenCV5(get())
    }
    single<IDocumentDetector>(named("classic")) {
        DocumentDetector(get())
    }
    single<IDocumentDetector>(named("onnx")) {
        OnnxDocumentDetector(
            get(),
            get(),
            "onnx/deeplabv3_mbv3_docseg.onnx",
            useCustomNormalization = true
        )
    }
    single<IDocumentDetector>(named("complex")) {
        DocumentDetectorOpenCV5(get())
    }
    single<IDocumentDetector>(named("minimal")) {
        DocumentDetectorMinimal(get())
    }
    single<IDocumentDetector>(named("combined")) {
        CombinedDocumentDetector(get())
    }
    viewModel<CameraViewModel>(named("camera")) {
        CameraViewModel(
            matBundle = get(),
            thresholdCalculator = get(),
            detector = get(named("combined"))
        )
    }
    viewModel<CameraViewModel>(named("fileScan")) {
        CameraViewModel(matBundle = get(), thresholdCalculator = get(), detector = get(named("combined")))
    }
    viewModel<PipelineSettingsViewModel>(named("pipelineSettings")) {
        PipelineSettingsViewModel(matBundle = get(), detector = get())
    }
}

fun initKoin(context: android.content.Context) {
    startKoin {
        androidLogger(Level.INFO)
        androidContext(context)
        modules(appModule)
    }
}
