package io.github.iostreamchik.scanner.data.detector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import org.opencv.core.Mat
import java.nio.FloatBuffer

class OnnxSessionManager(
    private val context: Context,
    private val env: OrtEnvironment,
    private val modelPath: String,
) {
    private var session: OrtSession? = null
    private var nchwBuffer: FloatArray? = null
    private var interleavedBuffer: FloatArray? = null

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
            try {
                sessionOptions.close()
            } catch (e: Exception) {
                Log.e(tag, "Failed to close session options", e)
            }
        }
    }

    fun getSession(): OrtSession? = session

    fun prepareInputTensor(rgbMat: Mat, inputSize: Int): OnnxTensor {
        val channels = rgbMat.channels()
        require(rgbMat.rows() == inputSize && rgbMat.cols() == inputSize)

        val channelSize = inputSize * inputSize
        val nchw = nchwBuffer ?: FloatArray(channelSize * channels).also { nchwBuffer = it }
        val interleaved = interleavedBuffer ?: FloatArray(channelSize * channels).also { interleavedBuffer = it }

        rgbMat.get(0, 0, interleaved)
        for (i in 0 until channelSize) {
            val src = i * channels
            for (c in 0 until channels) {
                nchw[c * channelSize + i] = interleaved[src + c]
            }
        }

        val inputShape = longArrayOf(1L, channels.toLong(), inputSize.toLong(), inputSize.toLong())
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(nchw), inputShape)
    }

    fun close(tag: String) {
        try {
            session?.close()
        } catch (e: Exception) {
            Log.e(tag, "Failed to close ONNX session", e)
        } finally {
            session = null
        }
    }
}
