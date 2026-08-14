package io.github.iostreamchik.scanner

import ai.onnxruntime.OrtEnvironment
import io.github.iostreamchik.scanner.presenter.camera.CameraViewModel
import io.github.iostreamchik.scanner.data.detector.CombinedDocumentDetector
import io.github.iostreamchik.scanner.data.detector.CornerKeypointDetector
import io.github.iostreamchik.scanner.data.detector.DocumentDetectorDirectionalSuppression
import io.github.iostreamchik.scanner.data.detector.DocumentDetectorMinimal
import io.github.iostreamchik.scanner.data.detector.HeatmapCornerDetector
import io.github.iostreamchik.scanner.data.detector.IDocumentDetector
import io.github.iostreamchik.scanner.data.detector.SegmentationDetector
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
    single { OrtEnvironment.getEnvironment() }
    factory<IMatBundle> { MatBundle() }

    factory<IDocumentDetector>(named("minimal")) {
        DocumentDetectorMinimal(get())
    }
    factory<IDocumentDetector>(named("directionalSuppression")) {
        DocumentDetectorDirectionalSuppression(get())
    }

    single<IDocumentDetector>(named("heatmapCorner")) {
        HeatmapCornerDetector(get(), get(), get())
    }
    single<IDocumentDetector>(named("cornerKeypoint")) {
        CornerKeypointDetector(get(), get(), get())
    }
    single<IDocumentDetector>(named("segmentation")) {
        SegmentationDetector(get(), get(), get())
    }

    single<IDocumentDetector>(named("combined")) {
        CombinedDocumentDetector(
            get<IDocumentDetector>(named("minimal")),
            get<IDocumentDetector>(named("directionalSuppression")),
            get<IDocumentDetector>(named("heatmapCorner")),
            get<IDocumentDetector>(named("cornerKeypoint")),
            get<IDocumentDetector>(named("segmentation")),
        )
    }

    single<IDocumentDetectorRepository> {
        DocumentDetectorRepositoryImpl(get<IDocumentDetector>(named("combined")))
    }

    viewModel {
        CameraViewModel(repository = get<IDocumentDetectorRepository>())
    }
}

fun initKoin(context: android.content.Context) {
    startKoin {
        androidLogger(Level.INFO)
        androidContext(context)
        modules(appModule)
    }
}
