package com.example.translatorapp.ocr

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.opencv.core.Mat

/**
 * Singleton to hold the current camera frame and latest OCR results.
 */
object FrameOcrRepository {
    private const val MAX_PENDING_OCR_SOURCE_FRAMES = 4

    private val frameLock = Any()
    private val ocrSourceFrames = PendingRequestStore<Mat>(
        maxEntries = MAX_PENDING_OCR_SOURCE_FRAMES,
        duplicate = { frame -> frame.clone() },
        release = { frame -> frame.release() }
    )

    // Holds the latest processed frame (with overlays)
    private val _currentFrame = MutableStateFlow<Mat?>(null)
    val currentFrame: StateFlow<Mat?> = _currentFrame

    // Holds the latest raw camera frame
    private val _latestCameraFrame = MutableStateFlow<Mat?>(null)
    val latestCameraFrame: StateFlow<Mat?> = _latestCameraFrame

    // Holds the latest OCR detection results
    private val _currentDetections = MutableStateFlow<List<OcrService.DetectionResult>>(emptyList())
    val currentDetections: StateFlow<List<OcrService.DetectionResult>> = _currentDetections

    /**
     * Update the current frame
     */
    fun updateFrame(frame: Mat) {
        synchronized(frameLock) {
            _currentFrame.value?.release()
            _currentFrame.value = frame.clone()
        }
    }

    /**
     * Update the latest raw camera frame
     */
    fun updateLatestCameraFrame(frame: Mat) {
        synchronized(frameLock) {
            _latestCameraFrame.value?.release()
            _latestCameraFrame.value = frame.clone()
        }
    }

    fun updateOcrSourceFrame(requestId: Long, frame: Mat) {
        synchronized(frameLock) {
            ocrSourceFrames.put(requestId, frame)
        }
    }

    fun snapshotCurrentFrame(): Mat? {
        synchronized(frameLock) {
            val frame = _currentFrame.value ?: return null
            if (frame.empty()) return null
            return frame.clone()
        }
    }

    fun snapshotLatestCameraFrame(): Mat? {
        synchronized(frameLock) {
            val frame = _latestCameraFrame.value ?: return null
            if (frame.empty()) return null
            return frame.clone()
        }
    }

    fun snapshotOcrSourceFrame(requestId: Long): Mat? {
        synchronized(frameLock) {
            val frame = ocrSourceFrames.snapshot(requestId) ?: return null
            if (frame.empty()) {
                frame.release()
                return null
            }
            return frame
        }
    }

    fun discardOcrSourceFrame(requestId: Long) {
        synchronized(frameLock) {
            ocrSourceFrames.discard(requestId)
        }
    }

    /**
     * Update the current OCR results
     */
    fun updateDetections(detections: List<OcrService.DetectionResult>) {
        _currentDetections.value.forEach { it.roiMat?.release() }
        _currentDetections.value = detections
    }

    /**
     * Clear the stored data (optional)
     */
    fun clearFrame() {
        synchronized(frameLock) {
            _currentFrame.value?.release()
            _currentFrame.value = null
        }
    }

    fun clearOcrSourceFrame() {
        synchronized(frameLock) {
            ocrSourceFrames.clear()
        }
    }
}
