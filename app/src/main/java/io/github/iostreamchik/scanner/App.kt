package io.github.iostreamchik.scanner

import android.app.Application
import android.util.Log
import org.opencv.android.OpenCVLoader

class App: Application() {

    override fun onCreate() {
        super.onCreate()
        val success = OpenCVLoader.initLocal()
        Log.d("OpenCV", if (success) "OpenCV initialized successfully" else "OpenCV initialization failed")
        initKoin(this)
    }
}