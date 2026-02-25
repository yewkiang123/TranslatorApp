package com.example.translatorapp.ocr

import android.graphics.Rect
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.opencv.core.Mat

/**
 * Singleton to hold the current camera frame and latest OCR results.
 */
object FrameOcrRepository {

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
        _currentFrame.value?.release() // release previous frame to avoid memory leaks
        _currentFrame.value = frame.clone() // clone to avoid shared reference issues
    }

    /**
     * Update the latest raw camera frame
     */
    fun updateLatestCameraFrame(frame: Mat) {
        _latestCameraFrame.value?.release()
        _latestCameraFrame.value = frame.clone()
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
        _currentFrame.value?.release()
        _currentFrame.value = null
    }
}
