package com.example.translatorapp.ocr

import android.graphics.Rect
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Core
import kotlin.coroutines.resumeWithException

class OcrService : ImageAnalysis.Analyzer {

    data class DetectionResult(
        val language: String,
        val text: String,
        val boundingBox: Rect?,
        val roiMat: Mat? = null
    )

    private val recognizers = mapOf(
        "latin" to TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS),
        "zh" to TextRecognition.getClient(ChineseTextRecognizerOptions.Builder().build()),
        "ja" to TextRecognition.getClient(JapaneseTextRecognizerOptions.Builder().build()),
        "ko" to TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
    )

    private val _detections = MutableSharedFlow<List<DetectionResult>>()
    val detections: SharedFlow<List<DetectionResult>> = _detections.asSharedFlow()

    private val seenBoxes = mutableListOf<Rect>()
    private val scope = CoroutineScope(Dispatchers.Default)

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image ?: run {
            imageProxy.close()
            return
        }

        var fullMat = imageProxy.toMat() ?: run {
            imageProxy.close()
            return
        }

        // Rotate Mat so it matches ML Kit bounding boxes
        val rotation = imageProxy.imageInfo.rotationDegrees
        fullMat = rotateMat(fullMat, rotation)

        val image = InputImage.fromMediaImage(mediaImage, rotation)

        scope.launch {
            try {
                val results = mutableListOf<DetectionResult>()

                for ((lang, recognizer) in recognizers) {
                    try {
                        val textResult = withTimeout(1000L) {
                            recognizer.process(image).await()
                        }

                        for (block in textResult.textBlocks) {
                            for (line in block.lines) { // ✅ capture each line separately
                                val box = line.boundingBox ?: continue
                                if (isDuplicateBox(box)) continue

                                val roiMat = extractRoiMat(fullMat, box)
                                seenBoxes.add(box)

                                results.add(
                                    DetectionResult(
                                        language = lang,
                                        text = line.text,
                                        boundingBox = box,
                                        roiMat = roiMat
                                    )
                                )
                            }
                        }

                    } catch (e: Exception) {
                        Log.e("OCR", "Recognizer $lang failed", e)
                    }
                }

                if (results.isNotEmpty()) {
                    _detections.emit(results)
                    Log.d("OCR", "Detected ${results.size} text lines")

                    // Update shared repository
                    FrameOcrRepository.updateFrame(fullMat)
                    FrameOcrRepository.updateDetections(results)
                }

            } catch (e: Exception) {
                Log.e("OCR", "Error in analysis", e)
            } finally {
                fullMat.release()
                imageProxy.close()
            }
        }
    }
    private fun isDuplicateBox(box: Rect): Boolean {
        return seenBoxes.any { existingBox ->
            kotlin.math.abs(existingBox.top - box.top) <= 2 &&
                    kotlin.math.abs(existingBox.bottom - box.bottom) <= 2 &&
                    kotlin.math.abs(existingBox.left - box.left) <= 2 &&
                    kotlin.math.abs(existingBox.right - box.right) <= 2
        }
    }
    private fun rotateMat(src: Mat, rotationDegrees: Int): Mat {
        if (rotationDegrees == 0) return src

        val dst = Mat()
        when (rotationDegrees) {
            90 -> Core.rotate(src, dst, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(src, dst, Core.ROTATE_180)
            270 -> Core.rotate(src, dst, Core.ROTATE_90_COUNTERCLOCKWISE)
            else -> return src
        }
        return dst
    }


    private fun extractRoiMat(fullMat: Mat, boundingBox: Rect): Mat? {
        return try {
            val x = boundingBox.left.coerceIn(0, fullMat.cols() - 1)
            val y = boundingBox.top.coerceIn(0, fullMat.rows() - 1)
            val width = boundingBox.width().coerceAtMost(fullMat.cols() - x)
            val height = boundingBox.height().coerceAtMost(fullMat.rows() - y)

            if (width > 0 && height > 0) {
                val roi = Mat(fullMat, org.opencv.core.Rect(x, y, width, height))
                roi.clone()
            } else null
        } catch (e: Exception) {
            Log.e("OCR", "ROI Extraction Failed", e)
            null
        }
    }
    fun resetCache() {
        seenBoxes.clear()
    }

    fun cleanup() {
        scope.cancel()
    }
}

@OptIn(ExperimentalGetImage::class)
fun ImageProxy.toMat(): Mat? {
    val mediaImage = this.image ?: return null

    val yBuffer = mediaImage.planes[0].buffer
    val uBuffer = mediaImage.planes[1].buffer
    val vBuffer = mediaImage.planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = android.graphics.YuvImage(
        nv21,
        android.graphics.ImageFormat.NV21,
        mediaImage.width,
        mediaImage.height,
        null
    )

    val out = java.io.ByteArrayOutputStream()
    yuvImage.compressToJpeg(
        android.graphics.Rect(0, 0, mediaImage.width, mediaImage.height),
        90,
        out
    )

    val bytes = out.toByteArray()
    val bitmap = android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)

    val mat = Mat()
    org.opencv.android.Utils.bitmapToMat(bitmap, mat)
    bitmap.recycle()

    return mat
}

suspend fun <T> com.google.android.gms.tasks.Task<T>.await(): T {
    return suspendCancellableCoroutine { cont ->
        addOnSuccessListener { cont.resume(it) { _, _, _ -> } }
        addOnFailureListener { cont.resumeWithException(it) }
    }
}
