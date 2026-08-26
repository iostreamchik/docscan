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
    private var permutationIndex: IntArray? = null
    private var matBuffer: FloatArray? = null
    private var nchwBuffer: FloatArray? = null


    fun init(tag: String) {
        if (session != null) return
        val availableProcessors = Runtime.getRuntime().availableProcessors()
        val intraOpThreads = maxOf(1, minOf(availableProcessors / 2, 4))

        val sessionOptions = OrtSession.SessionOptions().apply {
            addXnnpack(emptyMap())
            setIntraOpNumThreads(intraOpThreads)
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
        val totalElements = channelSize * channels

        // Precompute permutation index for NHWC→NCHW deinterleave
        val perm = permutationIndex?.takeIf { it.size == totalElements }
            ?: IntArray(totalElements).also { idx ->
                for (i in 0 until channelSize) {
                    for (c in 0 until channels) {
                        idx[c * channelSize + i] = i * channels + c
                    }
                }
                permutationIndex = idx
            }

        // Read interleaved (NHWC) from OpenCV Mat into reusable buffer
        val matBuf = matBuffer?.takeIf { it.size == totalElements }
            ?: FloatArray(totalElements).also { matBuffer = it }
        rgbMat.get(0, 0, matBuf)

        // Apply permutation in a single pass
        val nchw = nchwBuffer?.takeIf { it.size == totalElements }
            ?: FloatArray(totalElements).also { nchwBuffer = it }
        for (i in 0 until totalElements) {
            nchw[i] = matBuf[perm[i]]
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
            matBuffer = null
            permutationIndex = null
        }
    }
}
