package io.github.iostreamchik.scanner

import io.github.iostreamchik.scanner.presenter.camera.CameraViewModel
import io.github.iostreamchik.scanner.data.detector.CombinedDocumentDetector
import io.github.iostreamchik.scanner.data.detector.CornerKeypointDetector
import io.github.iostreamchik.scanner.data.detector.DocumentDetectorDirectionalSuppression
import io.github.iostreamchik.scanner.data.detector.DocumentDetectorMinimal
import io.github.iostreamchik.scanner.data.detector.IDocumentDetector
import io.github.iostreamchik.scanner.data.detector.OnnxDocumentDetector
import io.github.iostreamchik.scanner.data.opencv.IMatBundle
import io.github.iostreamchik.scanner.data.opencv.MatBundle
import io.github.iostreamchik.scanner.data.repository.DocumentDetectorRepositoryImpl
import io.github.iostreamchik.scanner.domain.repository.IDocumentDetectorRepository
import org.koin.android.ext.koin.androidContext
import org.koin.android.ext.koin.androidLogger
import org.koin.core.context.startKoin
import org.koin.core.logger.Level
import org.koin.core.module.dsl.viewModel
import org.koin.core.qualifier.named
import org.koin.dsl.module

val appModule = module {
    factory<IMatBundle> { MatBundle() }
    single<IDocumentDetector>(named("onnx")) {
        OnnxDocumentDetector(
            get(),
            get(),
            "onnx/deeplabv3_mbv3_docseg.onnx",
            useCustomNormalization = true
        )
    }
    single<IDocumentDetector>(named("minimal")) {
        DocumentDetectorMinimal(get())
    }
    single<IDocumentDetector>(named("cornerKeypoint")) {
        CornerKeypointDetector(get(), get())
    }
    single<IDocumentDetector>(named("combined")) {
        CombinedDocumentDetector(get())
    }
    single<IDocumentDetector>(named("directionalSuppression")) {
        DocumentDetectorDirectionalSuppression(get())
    }
    single<IDocumentDetectorRepository> {
        DocumentDetectorRepositoryImpl(get<IDocumentDetector>(named("combined")))
    }
    viewModel<CameraViewModel>(named("camera")) {
        CameraViewModel(
            repository = get<IDocumentDetectorRepository>()
        )
    }
    viewModel<CameraViewModel>(named("fileScan")) {
        CameraViewModel(
            repository = get<IDocumentDetectorRepository>()
        )
    }
}

fun initKoin(context: android.content.Context) {
    startKoin {
        androidLogger(Level.INFO)
        androidContext(context)
        modules(appModule)
    }
}
