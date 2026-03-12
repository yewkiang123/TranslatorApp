package com.example.translatorapp.ocr

import android.graphics.Bitmap
import android.graphics.Rect
import android.os.SystemClock
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.translatorapp.OpenCVInitializer
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

    data class OcrDetectionEvent(
        val requestId: Long,
        val results: List<DetectionResult>,
        val frameCapturedAtMs: Long,
        val stableAtMs: Long,
        val ocrCompletedAtMs: Long
    )

    data class TextRegion(
        val boundingBox: Rect?,
        val cornerPoints: Array<android.graphics.Point>? = null
    )

    data class DetectionResult(
        val language: String,
        val text: String,
        val boundingBox: Rect?,
        val cornerPoints: Array<android.graphics.Point>? = null,
        val blockBoundingBox: Rect? = null,
        val blockCornerPoints: Array<android.graphics.Point>? = null,
        val textRegions: List<TextRegion> = emptyList(),
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
    private val _detectionEvents = MutableSharedFlow<OcrDetectionEvent>()
    val detectionEvents: SharedFlow<OcrDetectionEvent> = _detectionEvents.asSharedFlow()
    private val _analyzerFrames = MutableSharedFlow<Long>(extraBufferCapacity = 32)
    val analyzerFrames: SharedFlow<Long> = _analyzerFrames.asSharedFlow()
    private val _motionStable = MutableSharedFlow<Boolean>(replay = 1)
    val motionStable: SharedFlow<Boolean> = _motionStable.asSharedFlow()

    private val seenDetections = mutableListOf<SeenDetection>()
    private val scope = CoroutineScope(Dispatchers.Default)
    private var prevGraySmall: Mat? = null
    private var lastMotionTimeMs = 0L
    private var unstableSinceMs = 0L
    private var lastStableState: Boolean? = null
    private val ocrRequestLock = Any()
    private val activeRequestScanGenerations = HashSet<Long>()
    @Volatile private var blockOcrUntilMs = 0L
    @Volatile private var motionRevision = 0L
    @Volatile private var scanGeneration = 0L
    @Volatile private var ocrArmed = true
    @Volatile private var sourceLanguageHint = "auto"
    @Volatile private var sessionResetRequested = false
    private var preferredRecognizer = "latin"
    private var nextOcrRequestId = 1L
    private var lastPeriodicRearmAtMs = 0L
    var ocrBurstFrames = 2
    private var ocrFramesProcessed = 0
    var recognizerTimeoutMs = 700L
    var fallbackRecognizerTimeoutMs = 500L
    var periodicRefreshIntervalMs = 2500L

    var motionThreshold = 10.5
    var stableHoldMs = 700L
    var unstableHoldMs = 1000L
    var activationBlockThreshold = 13.0
    var hardMotionThreshold = 19.0
    var motionCooldownMs = 350L

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        val frameCapturedAtMs = SystemClock.elapsedRealtime()
        _analyzerFrames.tryEmit(frameCapturedAtMs)

        if (!OpenCVInitializer.isReady()) {
            imageProxy.close()
            return
        }

        var fullMat = try {
            imageProxy.toMat()
        } catch (e: UnsatisfiedLinkError) {
            Log.e("OCR", "OpenCV native symbols unavailable while converting camera frame", e)
            null
        } catch (e: Exception) {
            Log.e("OCR", "Failed to convert camera frame to Mat", e)
            null
        } ?: run {
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
        val stableAtMs = SystemClock.elapsedRealtime()
        armPeriodicRefreshIfDue(stableAtMs)
        if (!ocrArmed) {
            fullMat.release()
            imageProxy.close()
            return
        }
        val requestMotionRevision = motionRevision
        val requestScanGeneration = scanGeneration
        if (!tryStartOcrRequest(requestScanGeneration)) {
            fullMat.release()
            imageProxy.close()
            return
        }
        val requestId = nextOcrRequestId++
        FrameOcrRepository.updateOcrSourceFrame(requestId, fullMat)
        val bitmap = matToBitmap(fullMat) ?: run {
            finishOcrRequest(requestScanGeneration)
            FrameOcrRepository.discardOcrSourceFrame(requestId)
            fullMat.release()
            imageProxy.close()
            return
        }
        val image = InputImage.fromBitmap(bitmap, 0)
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
                    val timeoutMs = OcrRecognizerPolicy.timeoutForRecognizer(
                        isAutoSource = isAutoSource,
                        recognizerIndex = index,
                        primaryTimeoutMs = recognizerTimeoutMs,
                        fallbackTimeoutMs = fallbackRecognizerTimeoutMs
                    )
                    try {
                        val textResult = withTimeout(timeoutMs) {
                            recognizer.process(image).await()
                        }

                        for (block in textResult.textBlocks) {
                            val blockBox = block.boundingBox
                            val blockCorners = block.cornerPoints
                            for (line in block.lines) {
                                val box = line.boundingBox ?: continue
                                if (isDuplicateDetection(line.text, box)) continue

                                seenDetections.add(
                                    OcrDuplicatePolicy.createSeenDetection(line.text, box)
                                )
                                recognizerFoundCount++
                                val textRegions = line.elements.mapNotNull { element ->
                                    val elementBox = element.boundingBox
                                    val elementCorners = element.cornerPoints
                                    if (elementBox == null && elementCorners == null) {
                                        null
                                    } else {
                                        TextRegion(
                                            boundingBox = elementBox,
                                            cornerPoints = elementCorners
                                        )
                                    }
                                }

                                results.add(
                                    DetectionResult(
                                        language = lang,
                                        text = line.text,
                                        boundingBox = box,
                                        cornerPoints = line.cornerPoints,
                                        blockBoundingBox = blockBox,
                                        blockCornerPoints = blockCorners,
                                        textRegions = textRegions
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

                    if (OcrRecognizerPolicy.shouldStopAfterRecognizer(
                            isAutoSource = isAutoSource,
                            recognizerFoundCount = recognizerFoundCount
                        )
                    ) {
                        break
                    }
                }

                if (recognizerHitCounts.isNotEmpty()) {
                    preferredRecognizer = recognizerHitCounts.maxByOrNull { it.value }?.key ?: preferredRecognizer
                }

                val ocrCompletedAtMs = SystemClock.elapsedRealtime()
                if (!shouldAcceptOcrResult(
                        requestMotionRevision = requestMotionRevision,
                        requestScanGeneration = requestScanGeneration,
                        nowAtMs = ocrCompletedAtMs
                    )
                ) {
                    FrameOcrRepository.discardOcrSourceFrame(requestId)
                    return@launch
                }
                _detections.emit(results)
                _detectionEvents.emit(
                    OcrDetectionEvent(
                        requestId = requestId,
                        results = results,
                        frameCapturedAtMs = frameCapturedAtMs,
                        stableAtMs = stableAtMs,
                        ocrCompletedAtMs = ocrCompletedAtMs
                    )
                )
                if (results.isNotEmpty()) {
                    Log.d("OCR", "Detected ${results.size} text lines")

                    // Update shared repository
                    FrameOcrRepository.updateDetections(results)
                    ocrFramesProcessed++
                    if (ocrFramesProcessed >= ocrBurstFrames) {
                        ocrArmed = false
                        lastPeriodicRearmAtMs = ocrCompletedAtMs
                    }
                }

            } catch (e: Exception) {
                Log.e("OCR", "Error in analysis", e)
                FrameOcrRepository.discardOcrSourceFrame(requestId)
            } finally {
                bitmap.recycle()
                finishOcrRequest(requestScanGeneration)
            }
        }
    }

    fun setSourceLanguageHint(languageCode: String) {
        sourceLanguageHint = languageCode.trim().lowercase().ifBlank { "auto" }
    }

    private fun applyPendingSessionReset() {
        if (!sessionResetRequested) return
        sessionResetRequested = false
        seenDetections.clear()
        prevGraySmall?.release()
        prevGraySmall = null
        lastMotionTimeMs = 0L
        unstableSinceMs = 0L
        lastStableState = null
        synchronized(ocrRequestLock) {
            activeRequestScanGenerations.clear()
        }
        blockOcrUntilMs = 0L
        motionRevision = 0L
        scanGeneration = 0L
        ocrArmed = true
        ocrFramesProcessed = 0
        nextOcrRequestId = 1L
        lastPeriodicRearmAtMs = 0L
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
        val now = SystemClock.elapsedRealtime()

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
        val motionDecision = OcrMotionPolicy.evaluate(
            motion = motion,
            nowAtMs = now,
            lastMotionAtMs = lastMotionTimeMs,
            unstableSinceMs = unstableSinceMs,
            lastStableState = lastStableState,
            blockOcrUntilMs = blockOcrUntilMs,
            motionThreshold = motionThreshold,
            activationBlockThreshold = activationBlockThreshold,
            hardMotionThreshold = hardMotionThreshold,
            stableHoldMs = stableHoldMs,
            unstableHoldMs = unstableHoldMs,
            motionCooldownMs = motionCooldownMs
        )
        lastMotionTimeMs = motionDecision.lastMotionAtMs
        unstableSinceMs = motionDecision.unstableSinceMs
        blockOcrUntilMs = motionDecision.blockOcrUntilMs
        val wasStable = lastStableState == true
        if (motionDecision.invalidateInFlightRequests) {
            motionRevision++
            seenDetections.clear()
        }

        prev.release()
        prevGraySmall = small.clone()
        small.release()

        val stable = motionDecision.stable
        if (stable && !wasStable) {
            scanGeneration++
            ocrArmed = true
            ocrFramesProcessed = 0
            seenDetections.clear()
            lastPeriodicRearmAtMs = 0L
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

    private fun shouldAcceptOcrResult(
        requestMotionRevision: Long,
        requestScanGeneration: Long,
        nowAtMs: Long
    ): Boolean {
        return OcrMotionPolicy.shouldAcceptResult(
            requestMotionRevision = requestMotionRevision,
            currentMotionRevision = motionRevision,
            requestScanGeneration = requestScanGeneration,
            currentScanGeneration = scanGeneration,
            nowAtMs = nowAtMs,
            blockOcrUntilMs = blockOcrUntilMs,
            isStable = lastStableState == true
        )
    }

    private fun tryStartOcrRequest(requestScanGeneration: Long): Boolean {
        synchronized(ocrRequestLock) {
            if (activeRequestScanGenerations.contains(requestScanGeneration)) {
                return false
            }
            activeRequestScanGenerations.add(requestScanGeneration)
            return true
        }
    }

    private fun finishOcrRequest(requestScanGeneration: Long) {
        synchronized(ocrRequestLock) {
            activeRequestScanGenerations.remove(requestScanGeneration)
        }
    }

    private fun armPeriodicRefreshIfDue(nowAtMs: Long) {
        if (!OcrRefreshPolicy.shouldRunPeriodicRefresh(
                isArmed = ocrArmed,
                lastRearmAtMs = lastPeriodicRearmAtMs,
                nowAtMs = nowAtMs,
                refreshIntervalMs = periodicRefreshIntervalMs
            )
        ) {
            return
        }

        ocrArmed = true
        ocrFramesProcessed = 0
        seenDetections.clear()
        lastPeriodicRearmAtMs = nowAtMs
    }

    private fun isDuplicateDetection(text: String, box: Rect): Boolean {
        return OcrDuplicatePolicy.isDuplicate(text, box, seenDetections)
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
        seenDetections.clear()
    }

    fun requestImmediateRefresh() {
        ocrArmed = true
        ocrFramesProcessed = 0
        seenDetections.clear()
        lastPeriodicRearmAtMs = 0L
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
    val yuv = Mat()
    val rgba = Mat()

    return try {
        yuv.create(this.height + this.height / 2, this.width, CvType.CV_8UC1)
        yuv.put(0, 0, nv21)
        Imgproc.cvtColor(yuv, rgba, Imgproc.COLOR_YUV2RGBA_NV21, 4)
        if (rgba.empty()) {
            rgba.release()
            null
        } else {
            rgba
        }
    } catch (e: UnsatisfiedLinkError) {
        rgba.release()
        throw e
    } catch (e: Exception) {
        Log.e("OCR", "toMat failed", e)
        rgba.release()
        null
    } finally {
        yuv.release()
    }
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
