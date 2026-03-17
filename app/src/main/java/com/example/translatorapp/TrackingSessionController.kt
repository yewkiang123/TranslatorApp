package com.example.translatorapp

import android.graphics.Bitmap
import android.util.Log
import android.view.View
import androidx.lifecycle.LifecycleCoroutineScope
import com.example.translatorapp.databinding.ActivityMainBinding
import com.example.translatorapp.ocr.FrameOcrRepository
import com.example.translatorapp.ocr.OcrService
import com.example.translatorapp.ocr.SceneTextReplacer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.conflate
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import kotlin.math.abs

internal class TrackingSessionController(
    private val binding: ActivityMainBinding,
    private val lifecycleScope: LifecycleCoroutineScope,
    private val isFreezeFrameMode: () -> Boolean,
    private val onTrackingLost: () -> Unit,
    private val onTrackingStateCleared: () -> Unit,
    private val onGeneratingActive: (Boolean) -> Unit,
    private val onGeneratedTextVisible: (Boolean) -> Unit,
    private val onOverlayRendered: () -> Unit
) {
    private val trackingLock = Any()
    private val trackingEngine = TrackingEngine(createTrackingConfig())
    private val replacer = SceneTextReplacer()

    private var renderJob: Job? = null
    private var trackingJob: Job? = null
    private var reusableBitmap: Bitmap? = null
    @Volatile private var overlayEnabled = false
    @Volatile private var needsFreshInpaint = true

    private var trackedResults: MutableList<OcrService.DetectionResult> = mutableListOf()
    private var trackedBaseResults: MutableList<OcrService.DetectionResult> = mutableListOf()
    private var trackedTranslations: MutableList<String> = mutableListOf()
    private var trackingSessionId = 0L
    private var trackingAnchorGray: Mat? = null
    private var trackingAccumulatedTransform = TrackingTransform()
    private var trackingMissCount = 0

    fun bindOverlayRenderer() {
        renderJob?.cancel()
        renderJob = lifecycleScope.launch {
            FrameOcrRepository.currentFrame
                .filterNotNull()
                .conflate()
                .collectLatest {
                    val mat = FrameOcrRepository.snapshotCurrentFrame() ?: return@collectLatest

                    if (mat.empty()) {
                        mat.release()
                        return@collectLatest
                    }
                    if (isFreezeFrameMode()) {
                        mat.release()
                        return@collectLatest
                    }

                    val w = mat.cols()
                    val h = mat.rows()
                    if (w <= 0 || h <= 0) {
                        mat.release()
                        return@collectLatest
                    }

                    val bmp = withContext(Dispatchers.Default) {
                        try {
                            val rgba = Mat()
                            when (mat.channels()) {
                                4 -> mat.copyTo(rgba)
                                3 -> org.opencv.imgproc.Imgproc.cvtColor(
                                    mat,
                                    rgba,
                                    org.opencv.imgproc.Imgproc.COLOR_BGR2RGBA
                                )

                                1 -> org.opencv.imgproc.Imgproc.cvtColor(
                                    mat,
                                    rgba,
                                    org.opencv.imgproc.Imgproc.COLOR_GRAY2RGBA
                                )

                                else -> {
                                    mat.release()
                                    rgba.release()
                                    return@withContext null
                                }
                            }
                            if (rgba.empty() || rgba.cols() <= 0 || rgba.rows() <= 0) {
                                mat.release()
                                rgba.release()
                                return@withContext null
                            }

                            val out = reusableBitmap?.takeIf {
                                it.width == rgba.cols() && it.height == rgba.rows()
                            } ?: Bitmap.createBitmap(
                                rgba.cols(),
                                rgba.rows(),
                                Bitmap.Config.ARGB_8888
                            ).also {
                                reusableBitmap = it
                            }

                            org.opencv.android.Utils.matToBitmap(rgba, out)
                            rgba.release()
                            mat.release()
                            out
                        } catch (e: Exception) {
                            mat.release()
                            Log.e(TAG, "render failed", e)
                            null
                        }
                    }

                    if (bmp != null && overlayEnabled && !isFreezeFrameMode()) {
                        binding.processedOverlay.setImageBitmap(bmp)
                        onOverlayRendered()
                    }
                }
        }
    }

    fun startTrackingLoop() {
        trackingJob?.cancel()
        trackingJob = lifecycleScope.launch(Dispatchers.Default) {
            while (isActive) {
                val loopStartedAt = android.os.SystemClock.elapsedRealtime()
                if (isFreezeFrameMode()) {
                    kotlinx.coroutines.delay(TRACK_INTERVAL_MS)
                    continue
                }
                val frame = FrameOcrRepository.snapshotLatestCameraFrame()
                if (frame != null && !frame.empty()) {
                    try {
                        trackAndRender(frame)
                    } catch (e: Exception) {
                        Log.e(TAG, "Tracking loop failed", e)
                        clearTrackingState(clearDisplayedOverlay = false)
                        onTrackingLost()
                    }
                } else {
                    frame?.release()
                }
                val remainingDelay =
                    TRACK_INTERVAL_MS - (android.os.SystemClock.elapsedRealtime() - loopStartedAt)
                if (remainingDelay > 0L) {
                    kotlinx.coroutines.delay(remainingDelay)
                }
            }
        }
    }

    fun stop() {
        renderJob?.cancel()
        trackingJob?.cancel()
        synchronized(trackingLock) {
            trackingAnchorGray?.release()
            trackingAnchorGray = null
            trackedResults = mutableListOf()
            trackedBaseResults = mutableListOf()
            trackedTranslations = mutableListOf()
            trackingAccumulatedTransform = TrackingTransform()
            trackingMissCount = 0
        }
        replacer.clearStableState()
        reusableBitmap = null
    }

    fun toGray(frame: Mat): Mat {
        return trackingEngine.toGray(frame)
    }

    fun hasActiveTrackingSession(): Boolean {
        return synchronized(trackingLock) {
            trackedResults.isNotEmpty()
        }
    }

    fun beginSession(
        results: List<OcrService.DetectionResult>,
        anchorGray: Mat,
        replaceTracking: Boolean
    ): Long {
        if (replaceTracking) {
            hideOverlayOnly()
            replacer.clearStableState()
        }

        val sessionId = synchronized(trackingLock) {
            trackingSessionId++
            trackedBaseResults = results.map { it.copy() }.toMutableList()
            trackedResults = results.map { it.copy() }.toMutableList()
            trackedTranslations = MutableList(results.size) { "" }
            trackingAnchorGray?.release()
            trackingAnchorGray = anchorGray
            trackingAccumulatedTransform = TrackingTransform()
            trackingMissCount = 0
            trackingSessionId
        }
        needsFreshInpaint = true
        return sessionId
    }

    fun applyTranslationResult(sessionId: Long, index: Int, text: String) {
        synchronized(trackingLock) {
            if (sessionId != trackingSessionId) return
            if (index !in trackedTranslations.indices) return
            trackedTranslations[index] = text
        }
    }

    fun hideOverlayOnly() {
        if (isFreezeFrameMode()) return
        if (!overlayEnabled) return

        overlayEnabled = false
        onGeneratedTextVisible(false)
        FrameOcrRepository.clearFrame()
        binding.root.post {
            binding.processedOverlay.setImageBitmap(null)
            binding.processedOverlay.visibility = View.INVISIBLE
        }
    }

    fun clearTrackingState(clearDisplayedOverlay: Boolean = true) {
        synchronized(trackingLock) {
            trackingSessionId++
            trackedResults = mutableListOf()
            trackedBaseResults = mutableListOf()
            trackedTranslations = mutableListOf()
            trackingAnchorGray?.release()
            trackingAnchorGray = null
            trackingAccumulatedTransform = TrackingTransform()
            trackingMissCount = 0
        }
        FrameOcrRepository.clearOcrSourceFrame()
        needsFreshInpaint = true
        onTrackingStateCleared()
        replacer.clearStableState()
        if (clearDisplayedOverlay) {
            FrameOcrRepository.clearFrame()
            overlayEnabled = false
            onGeneratedTextVisible(false)
            binding.root.post {
                binding.processedOverlay.setImageBitmap(null)
                binding.processedOverlay.visibility = View.INVISIBLE
            }
        }
    }

    private fun trackAndRender(frame: Mat) {
        if (isFreezeFrameMode()) {
            frame.release()
            return
        }

        val snapshot = synchronized(trackingLock) {
            if (trackedResults.isEmpty()) {
                return@synchronized null
            }
            val anchor = trackingAnchorGray?.clone()
            if (anchor == null || anchor.empty()) {
                anchor?.release()
                return@synchronized null
            }
            TrackingSnapshot(
                referenceResults = trackedResults.map { it.copy() },
                baseResults = trackedBaseResults.map { it.copy() },
                translations = trackedTranslations.toList(),
                anchorGray = anchor,
                accumulatedTransform = trackingAccumulatedTransform
            )
        }

        if (snapshot == null) {
            hideOverlayOnly()
            frame.release()
            return
        }

        val referencePivot = trackingEngine.computeTrackingPivot(
            snapshot.referenceResults,
            frame.cols(),
            frame.rows()
        )
        val currentGray = trackingEngine.toGray(frame)
        if (currentGray.empty() || currentGray.cols() <= 0 || currentGray.rows() <= 0) {
            currentGray.release()
            onTrackingLost()
            frame.release()
            return
        }
        val estimatedTransform = trackingEngine.estimateTrackingTransform(
            prev = snapshot.anchorGray,
            current = currentGray,
            referenceResults = snapshot.referenceResults,
            referencePivot = referencePivot
        )
        snapshot.anchorGray.release()

        val (stepTransform, accumulatedTransform, missCount) = synchronized(trackingLock) {
            if (estimatedTransform != null) {
                trackingAccumulatedTransform = trackingEngine.composeTransforms(
                    step = estimatedTransform,
                    accumulated = trackingAccumulatedTransform
                )
                trackingMissCount = 0
                Triple(estimatedTransform, trackingAccumulatedTransform, trackingMissCount)
            } else {
                trackingMissCount++
                Triple(TrackingTransform(), trackingAccumulatedTransform, trackingMissCount)
            }
        }

        if (estimatedTransform == null && missCount > MAX_TRACKING_MISSES) {
            currentGray.release()
            clearTrackingState(clearDisplayedOverlay = false)
            onTrackingLost()
            frame.release()
            return
        }

        if (trackingEngine.measureTransformMotion(stepTransform, referencePivot) > PAN_THRESHOLD_PX ||
            abs(stepTransform.scale - 1.0) > ZOOM_RESET_THRESHOLD
        ) {
            currentGray.release()
            clearTrackingState(clearDisplayedOverlay = false)
            onTrackingLost()
            frame.release()
            return
        }

        val shifted = trackingEngine.applyTrackingTransform(
            results = snapshot.baseResults,
            transform = accumulatedTransform,
            maxW = frame.cols(),
            maxH = frame.rows()
        )

        if (shifted.none { it.boundingBox != null }) {
            currentGray.release()
            clearTrackingState()
            onTrackingLost()
            frame.release()
            return
        }

        if (estimatedTransform != null) {
            advanceTrackingState(shifted, currentGray)
        } else {
            currentGray.release()
        }

        if (snapshot.translations.none { it.isNotBlank() }) {
            hideOverlayOnly()
            frame.release()
            return
        }

        onGeneratingActive(true)
        try {
            val updated = replacer.replaceText(
                frame = frame,
                detectionResults = shifted,
                translatedTexts = snapshot.translations,
                reuseTrackedInpaint = !needsFreshInpaint
            )
            try {
                if (updated.empty() || updated.cols() <= 0 || updated.rows() <= 0) {
                    return
                }
                FrameOcrRepository.updateFrame(updated)
                needsFreshInpaint = false
            } finally {
                updated.release()
            }
            overlayEnabled = true
            onGeneratedTextVisible(true)
            binding.root.post {
                binding.processedOverlay.visibility = View.VISIBLE
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed generating processed frame", e)
        } finally {
            onGeneratingActive(false)
            frame.release()
        }
    }

    private fun advanceTrackingState(
        shiftedResults: List<OcrService.DetectionResult>,
        nextAnchorGray: Mat
    ) {
        synchronized(trackingLock) {
            trackedResults = shiftedResults.map { it.copy() }.toMutableList()
            trackingAnchorGray?.release()
            trackingAnchorGray = nextAnchorGray
        }
    }

    private fun createTrackingConfig(): TrackingConfig {
        return TrackingConfig(
            minTrackShiftPx = 1.5,
            maxTrackStepPx = 96.0,
            minScaleDelta = 0.015,
            minTrackScale = 0.82,
            maxTrackScale = 1.22,
            trackingFeatureCount = 420,
            trackingFeatureQualityLevel = 0.005,
            trackingFeatureMinDistance = 4.0,
            minTrackingRoiSizePx = 180,
            minTrackingTargetTextSizePx = 56.0,
            maxTrackingUpscale = 4.0,
            trackingFlowWindowSize = Size(41.0, 41.0),
            trackingFlowMaxLevel = 5,
            minTrackingValidPoints = 6,
            minTrackingTranslationPoints = 3,
            minTrackingAffineInliers = 4,
            minTrackingRegionSizePx = 96,
            minTrackingScaleRadiusPx = 8.0,
            maxTrackRotationDegrees = 6.0,
            trackingFlowCriteria = TermCriteria(
                TermCriteria.COUNT or TermCriteria.EPS,
                20,
                0.03
            )
        )
    }

    private companion object {
        const val TAG = "TrackingSession"
        const val TRACK_INTERVAL_MS = 45L
        const val MAX_TRACKING_MISSES = 8
        const val PAN_THRESHOLD_PX = 220.0
        const val ZOOM_RESET_THRESHOLD = 0.28
    }
}
