package io.github.iostreamchik.scanner.data.detector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import org.opencv.core.Core
import org.opencv.core.Mat
import java.nio.FloatBuffer

class OnnxSessionManager(
    private val context: Context,
    private val env: OrtEnvironment,
    private val modelPath: String,
) {
    private var session: OrtSession? = null

    val inputName: String?
        get() = session?.inputNames?.iterator()?.next()

    fun init(tag: String) {
        if (session != null) return

        val sessionOptions = OrtSession.SessionOptions().apply {
            addXnnpack(emptyMap())
            setIntraOpNumThreads(2)
            setMemoryPatternOptimization(true)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL)
        }

        try {
            val modelBytes = context.assets.open(modelPath).use { it.readBytes() }
            session = env.createSession(modelBytes, sessionOptions)
        } catch (e: Exception) {
            Log.e(tag, "Failed to load ONNX model", e)
        } finally {
            sessionOptions.close()
        }
    }



    fun getSession(): OrtSession? = session

    fun prepareInputTensor(
        rgbMat: Mat,
        inputSize: Int = 256,
        channels: Int = 3
    ): OnnxTensor {
        val channelSize = inputSize * inputSize
        val nchwBuffer = FloatArray(channelSize * channels)
        val channelBuffer = FloatArray(channelSize)

        val channelMats = mutableListOf<Mat>()
        Core.split(rgbMat, channelMats)

        try {
            for (c in 0 until channels) {
                channelMats[c].get(0, 0, channelBuffer)
                System.arraycopy(channelBuffer, 0, nchwBuffer, c * channelSize, channelSize)
            }
        } finally {
            channelMats.forEach { it.release() }
        }

        val inputShape = longArrayOf(1L, channels.toLong(), inputSize.toLong(), inputSize.toLong())
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(nchwBuffer), inputShape)
    }

    fun close() {
        session?.close()
        session = null
    }
}
