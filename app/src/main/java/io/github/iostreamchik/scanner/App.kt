package io.github.iostreamchik.scanner

import android.app.Application
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat

class App: Application() {

    override fun onCreate() {
        super.onCreate()
        val success = OpenCVLoader.initLocal()

        if (success) {
            Log.d("OpenCV", "OpenCV initialized successfully")
        } else {
            Log.e("OpenCV", "OpenCV initialization failed")
        }

        val mat = Mat(10, 10, CvType.CV_8UC1)
        Log.d("OpenCV", "Mat created: ${mat.rows()} x ${mat.cols()}")
    }
}