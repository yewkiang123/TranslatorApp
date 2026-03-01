package com.example.translatorapp.ocr

import android.graphics.Bitmap
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
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.coroutines.resumeWithException

class OcrService : ImageAnalysis.Analyzer {

    data class DetectionResult(
        val language: String,
        val text: String,
        val boundingBox: Rect?,
        val cornerPoints: Array<android.graphics.Point>? = null,
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
    private val _motionStable = MutableSharedFlow<Boolean>(replay = 1)
    val motionStable: SharedFlow<Boolean> = _motionStable.asSharedFlow()

    private val seenBoxes = mutableListOf<Rect>()
    private val scope = CoroutineScope(Dispatchers.Default)
    private var prevGraySmall: Mat? = null
    private var lastMotionTimeMs = 0L
    private var unstableSinceMs = 0L
    private var lastStableState: Boolean? = null
    @Volatile private var ocrArmed = true
    @Volatile private var ocrInFlight = false
    @Volatile private var sourceLanguageHint = "auto"
    @Volatile private var sessionResetRequested = false
    private var preferredRecognizer = "latin"
    var ocrBurstFrames = 2
    private var ocrFramesProcessed = 0
    var recognizerTimeoutMs = 900L
    var earlyExitMinLines = 2

    var motionThreshold = 10.5
    var stableHoldMs = 1200L
    var unstableHoldMs = 1000L
    var hardMotionThreshold = 19.0

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        var fullMat = imageProxy.toMat() ?: run {
            imageProxy.close()
            return
        }

        // Rotate Mat so it matches ML Kit bounding boxes
        val rotation = imageProxy.imageInfo.rotationDegrees
        fullMat = rotateMat(fullMat, rotation)
        applyPendingSessionReset()

        // Always publish the latest raw camera frame
        FrameOcrRepository.updateLatestCameraFrame(fullMat)

        val isStable = updateMotionState(fullMat)
        emitMotionState(isStable)
        if (!isStable) {
            fullMat.release()
            imageProxy.close()
            return
        }
        if (!ocrArmed) {
            fullMat.release()
            imageProxy.close()
            return
        }
        if (ocrInFlight) {
            fullMat.release()
            imageProxy.close()
            return
        }

        val bitmap = matToBitmap(fullMat) ?: run {
            fullMat.release()
            imageProxy.close()
            return
        }
        val image = InputImage.fromBitmap(bitmap, 0)
        ocrInFlight = true
        fullMat.release()
        imageProxy.close()

        scope.launch {
            try {
                val results = mutableListOf<DetectionResult>()
                val recognizerHitCounts = HashMap<String, Int>()

                val orderedRecognizers = buildRecognizerOrder()
                val isAutoSource = sourceLanguageHint == "auto"
                for ((index, lang) in orderedRecognizers.withIndex()) {
                    val recognizer = recognizers[lang] ?: continue
                    var recognizerFoundCount = 0
                    try {
                        val textResult = withTimeout(recognizerTimeoutMs) {
                            recognizer.process(image).await()
                        }

                        for (block in textResult.textBlocks) {
                            for (line in block.lines) {
                                val box = line.boundingBox ?: continue
                                if (isDuplicateBox(box)) continue

                                seenBoxes.add(box)
                                recognizerFoundCount++

                                results.add(
                                    DetectionResult(
                                        language = lang,
                                        text = line.text,
                                        boundingBox = box,
                                        cornerPoints = line.cornerPoints
                                    )
                                )
                            }
                        }

                    } catch (e: Exception) {
                        Log.e("OCR", "Recognizer $lang failed", e)
                    }

                    if (recognizerFoundCount > 0) {
                        recognizerHitCounts[lang] = recognizerFoundCount
                    }

                    if (isAutoSource &&
                        index == 0 &&
                        recognizerFoundCount >= earlyExitMinLines
                    ) {
                        break
                    }
                }

                if (recognizerHitCounts.isNotEmpty()) {
                    preferredRecognizer = recognizerHitCounts.maxByOrNull { it.value }?.key ?: preferredRecognizer
                }

                if (results.isNotEmpty()) {
                    _detections.emit(results)
                    Log.d("OCR", "Detected ${results.size} text lines")

                    // Update shared repository
                    FrameOcrRepository.updateDetections(results)
                    ocrFramesProcessed++
                    if (ocrFramesProcessed >= ocrBurstFrames) {
                        ocrArmed = false
                    }
                }

            } catch (e: Exception) {
                Log.e("OCR", "Error in analysis", e)
            } finally {
                bitmap.recycle()
                ocrInFlight = false
            }
        }
    }

    fun setSourceLanguageHint(languageCode: String) {
        sourceLanguageHint = languageCode.trim().lowercase().ifBlank { "auto" }
    }

    private fun applyPendingSessionReset() {
        if (!sessionResetRequested) return
        sessionResetRequested = false
        seenBoxes.clear()
        prevGraySmall?.release()
        prevGraySmall = null
        lastMotionTimeMs = 0L
        unstableSinceMs = 0L
        lastStableState = null
        ocrArmed = true
        ocrFramesProcessed = 0
        preferredRecognizer = languageCodeToRecognizer(sourceLanguageHint)
    }

    private fun languageCodeToRecognizer(languageCode: String): String {
        return when {
            languageCode.startsWith("zh") -> "zh"
            languageCode.startsWith("ja") -> "ja"
            languageCode.startsWith("ko") -> "ko"
            else -> "latin"
        }
    }

    private fun buildRecognizerOrder(): List<String> {
        val sourceHint = sourceLanguageHint
        if (sourceHint != "auto") {
            return listOf(languageCodeToRecognizer(sourceHint))
        }

        if (!recognizers.containsKey(preferredRecognizer)) {
            preferredRecognizer = "latin"
        }
        val ordered = ArrayList<String>(recognizers.size)
        ordered.add(preferredRecognizer)
        recognizers.keys.forEach { key ->
            if (key != preferredRecognizer) {
                ordered.add(key)
            }
        }
        return ordered
    }

    private fun matToBitmap(frame: Mat): Bitmap? {
        if (frame.empty() || frame.cols() <= 0 || frame.rows() <= 0) return null

        return try {
            val bitmap = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(frame, bitmap)
            bitmap
        } catch (e: Exception) {
            Log.e("OCR", "Failed to convert Mat to Bitmap for OCR", e)
            null
        }
    }

    private fun updateMotionState(frame: Mat): Boolean {
        val now = System.currentTimeMillis()

        val gray = Mat()
        when (frame.channels()) {
            4 -> Imgproc.cvtColor(frame, gray, Imgproc.COLOR_RGBA2GRAY)
            3 -> Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY)
            1 -> frame.copyTo(gray)
            else -> Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY)
        }

        val small = Mat()
        Imgproc.resize(gray, small, Size(160.0, 120.0))
        gray.release()

        val prev = prevGraySmall
        if (prev == null || prev.empty()) {
            prevGraySmall?.release()
            prevGraySmall = small.clone()
            small.release()
            lastMotionTimeMs = now
            return false
        }

        val diff = Mat()
        Core.absdiff(prev, small, diff)
        val mean = Core.mean(diff)
        diff.release()

        val motion = mean.`val`[0]
        if (motion > hardMotionThreshold) {
            lastMotionTimeMs = now
            unstableSinceMs = now - unstableHoldMs
            seenBoxes.clear()
        } else if (motion > motionThreshold) {
            lastMotionTimeMs = now
            if (unstableSinceMs == 0L) unstableSinceMs = now
        } else {
            unstableSinceMs = 0L
        }

        prev.release()
        prevGraySmall = small.clone()
        small.release()

        val stableEnough = (now - lastMotionTimeMs) >= stableHoldMs
        val unstableEnough = unstableSinceMs != 0L && (now - unstableSinceMs) >= unstableHoldMs

        val stable = when {
            unstableEnough -> false
            stableEnough -> true
            else -> (lastStableState == true)
        }
        if (!stable && lastStableState == true) {
            // Only reset OCR when we truly transitioned to unstable
            ocrArmed = true
            ocrFramesProcessed = 0
            seenBoxes.clear()
            preferredRecognizer = "latin"
        }
        return stable
    }

    // Sharpness gating removed; stability timing handles blur.

    private fun emitMotionState(stable: Boolean) {
        if (lastStableState == stable) return
        lastStableState = stable
        scope.launch {
            _motionStable.emit(stable)
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
        src.release()
        return dst
    }
    fun resetCache() {
        seenBoxes.clear()
    }

    fun resetSessionState() {
        sessionResetRequested = true
    }

    fun cleanup() {
        scope.cancel()
        prevGraySmall?.release()
        prevGraySmall = null
    }
}

@OptIn(ExperimentalGetImage::class)
fun ImageProxy.toMat(): Mat? {
    if (this.format != android.graphics.ImageFormat.YUV_420_888) return null

    val nv21 = yuv420ToNv21(this)
    val yuv = Mat(this.height + this.height / 2, this.width, CvType.CV_8UC1)
    yuv.put(0, 0, nv21)

    val rgba = Mat()
    Imgproc.cvtColor(yuv, rgba, Imgproc.COLOR_YUV2RGBA_NV21, 4)
    yuv.release()

    return rgba
}

private fun yuv420ToNv21(image: ImageProxy): ByteArray {
    val width = image.width
    val height = image.height
    val ySize = width * height
    val uvSize = width * height / 4
    val out = ByteArray(ySize + uvSize * 2)

    copyPlane(image.planes[0], width, height, out, 0, 1)
    copyPlane(image.planes[2], width / 2, height / 2, out, ySize, 2)
    copyPlane(image.planes[1], width / 2, height / 2, out, ySize + 1, 2)

    return out
}

private fun copyPlane(
    plane: ImageProxy.PlaneProxy,
    width: Int,
    height: Int,
    out: ByteArray,
    offset: Int,
    outputStride: Int
) {
    val buffer = plane.buffer
    buffer.rewind()

    val rowStride = plane.rowStride
    val pixelStride = plane.pixelStride
    val rowData = ByteArray(rowStride)
    var outputPos = offset

    for (row in 0 until height) {
        val bytesToRead = if (pixelStride == 1 && outputStride == 1) {
            width
        } else {
            (width - 1) * pixelStride + 1
        }

        buffer.get(rowData, 0, bytesToRead)

        if (pixelStride == 1 && outputStride == 1) {
            System.arraycopy(rowData, 0, out, outputPos, width)
            outputPos += width
        } else {
            var inputPos = 0
            repeat(width) {
                out[outputPos] = rowData[inputPos]
                outputPos += outputStride
                inputPos += pixelStride
            }
        }

        if (row < height - 1) {
            val skip = rowStride - bytesToRead
            if (skip > 0) {
                buffer.position(buffer.position() + skip)
            }
        }
    }
}

suspend fun <T> com.google.android.gms.tasks.Task<T>.await(): T {
    return suspendCancellableCoroutine { cont ->
        addOnSuccessListener { cont.resume(it) { _, _, _ -> } }
        addOnFailureListener { cont.resumeWithException(it) }
    }
}
