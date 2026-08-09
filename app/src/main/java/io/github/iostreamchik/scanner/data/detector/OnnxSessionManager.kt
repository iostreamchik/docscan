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
    private val modelPath: String,
) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null

    val inputName: String?
        get() = session?.inputNames?.iterator()?.next()

    fun init(tag: String) {
        if (session != null) return

        val sessionOptions = OrtSession.SessionOptions().apply {
            addXnnpack(emptyMap())
            setIntraOpNumThreads(1)
            setMemoryPatternOptimization(true)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
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

    fun getEnvironment(): OrtEnvironment = env

    fun getSession(): OrtSession? = session

    fun prepareInputTensor(
        rgbMat: Mat,
        inputSize: Int,
        channels: Int = 3
    ): OnnxTensor {
        val channelSize = inputSize * inputSize
        val nchwData = FloatArray(inputSize * inputSize * channels)

        val channelMats = mutableListOf<Mat>()
        Core.split(rgbMat, channelMats)

        for (c in 0 until channels) {
            val channelData = FloatArray(channelSize)
            channelMats[c].get(0, 0, channelData)
            System.arraycopy(channelData, 0, nchwData, c * channelSize, channelSize)
        }
        channelMats.forEach { it.release() }

        val inputShape = longArrayOf(1, channels.toLong(), inputSize.toLong(), inputSize.toLong())
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(nchwData), inputShape)
    }

    fun close() {
        session?.close()
        session = null
        try {
            env.close()
        } catch (_: Exception) {
        }
    }
}
