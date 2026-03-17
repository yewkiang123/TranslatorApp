package com.example.translatorapp

import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.LifecycleCoroutineScope
import com.example.translatorapp.logic.LanguageUiLogic
import com.example.translatorapp.ocr.FrameOcrRepository
import com.example.translatorapp.ocr.OcrService
import com.example.translatorapp.translate.TranslationService
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

internal class MainOcrCoordinator(
    private val lifecycleScope: LifecycleCoroutineScope,
    private val ocrService: OcrService,
    private val translationService: TranslationService,
    private val languagePanelController: LanguagePanelController,
    private val processingIndicatorController: ProcessingIndicatorController,
    private val metricsController: MetricsController,
    private val trackingSessionController: TrackingSessionController,
    private val isFreezeFrameMode: () -> Boolean,
    private val isOpenCvReady: () -> Boolean
) {
    private var lastProcessTime = 0L
    private var lastTrackingSeedAtMs = 0L
    private var pendingRedetectJob: Job? = null
    private var nextStableDetectionId = 1L
    @Volatile private var cameraMotionStable = false
    @Volatile private var redetectAfterMotionStops = true
    @Volatile private var replaceTrackingOnNextOcr = false
    private val translationCache = HashMap<String, String>()

    fun setupOcrListener() {
        ocrService.analyzerFrames.onEach { analyzedAtMs ->
            if (!isFreezeFrameMode()) {
                metricsController.recordAnalyzerFrame(analyzedAtMs)
            }
        }.launchIn(lifecycleScope)

        ocrService.detectionEvents.onEach { event ->
            if (!isFreezeFrameMode()) {
                metricsController.recordOcrEvent(event)
            }
            val replaceTracking = !isFreezeFrameMode() && replaceTrackingOnNextOcr
            if (event.results.isNotEmpty()) {
                processingIndicatorController.setNoTextDetected(false)
                if (processOnInterval(event, replaceTracking)) {
                    replaceTrackingOnNextOcr = false
                }
            } else if (!isFreezeFrameMode()) {
                if (replaceTracking) {
                    replaceTrackingOnNextOcr = false
                    trackingSessionController.clearTrackingState()
                }
                processingIndicatorController.showNoTextDetected()
            }
        }.launchIn(lifecycleScope)

        ocrService.motionStable.onEach { stable ->
            if (!isFreezeFrameMode()) {
                cameraMotionStable = stable
                if (stable) {
                    if (redetectAfterMotionStops) {
                        scheduleRedetectionAfterSettle()
                    } else {
                        processingIndicatorController.setDetectingActive(false)
                    }
                } else {
                    cancelPendingRedetection()
                    replaceTrackingOnNextOcr = false
                    trackingSessionController.hideOverlayOnly()
                    redetectAfterMotionStops = true
                    processingIndicatorController.setNoTextDetected(false)
                    processingIndicatorController.setDetectingActive(false)
                }
            }
        }.launchIn(lifecycleScope)
    }

    fun requestRedetectionAfterTrackingLoss() {
        lastProcessTime = 0L
        processingIndicatorController.setNoTextDetected(false)
        if (cameraMotionStable) {
            redetectAfterMotionStops = true
            scheduleRedetectionAfterSettle()
        } else {
            cancelPendingRedetection()
            redetectAfterMotionStops = true
            processingIndicatorController.setDetectingActive(false)
        }
    }

    fun clearDetectedBlocks() {
        resetDetectionSession()
        Log.d(TAG, "Reset detection session due to language change")
    }

    fun clearTranslationCache() {
        translationCache.clear()
    }

    fun resetDetectionSession() {
        cancelPendingRedetection()
        replaceTrackingOnNextOcr = false
        clearTranslationCache()
        trackingSessionController.clearTrackingState()
        ocrService.resetSessionState()
        cameraMotionStable = false
        redetectAfterMotionStops = true
        lastProcessTime = 0L
        lastTrackingSeedAtMs = 0L
        processingIndicatorController.resetProcessingWork(
            keepDetecting = !isFreezeFrameMode() && isOpenCvReady()
        )
    }

    fun cancelPendingRedetection() {
        pendingRedetectJob?.cancel()
        pendingRedetectJob = null
    }

    private fun scheduleRedetectionAfterSettle() {
        pendingRedetectJob?.cancel()
        pendingRedetectJob = lifecycleScope.launch {
            delay(REDETECT_SETTLE_DELAY_MS)
            if (!isActive || isFreezeFrameMode() || !cameraMotionStable || !redetectAfterMotionStops) {
                return@launch
            }
            replaceTrackingOnNextOcr = trackingSessionController.hasActiveTrackingSession()
            ocrService.requestImmediateRefresh()
            lastProcessTime = 0L
            redetectAfterMotionStops = false
            processingIndicatorController.setDetectingActive(true)
        }
    }

    private fun processOnInterval(
        event: OcrService.OcrDetectionEvent,
        replaceTracking: Boolean = false
    ): Boolean {
        if (isFreezeFrameMode()) return false

        val nowAtMs = SystemClock.elapsedRealtime()
        if (!replaceTracking && nowAtMs - lastProcessTime <= PROCESS_INTERVAL_MS) return false

        val hasActiveTrackingSession = trackingSessionController.hasActiveTrackingSession()
        val shouldResyncTrackingSession = !replaceTracking &&
            hasActiveTrackingSession &&
            (lastTrackingSeedAtMs == 0L ||
                event.ocrCompletedAtMs - lastTrackingSeedAtMs >= TRACKING_RESYNC_INTERVAL_MS)

        if (!replaceTracking && hasActiveTrackingSession && !shouldResyncTrackingSession) {
            return false
        }
        lastProcessTime = nowAtMs

        val results = event.results.map { result ->
            result.copy(stableId = result.stableId ?: "det-${nextStableDetectionId++}")
        }
        val baseFrame = FrameOcrRepository.snapshotOcrSourceFrame(event.requestId)
            ?: FrameOcrRepository.snapshotLatestCameraFrame()
            ?: return false
        FrameOcrRepository.discardOcrSourceFrame(event.requestId)
        if (baseFrame.empty() || baseFrame.cols() <= 0 || baseFrame.rows() <= 0) {
            baseFrame.release()
            return false
        }

        val anchorGray = trackingSessionController.toGray(baseFrame)
        baseFrame.release()
        if (anchorGray.empty() || anchorGray.cols() <= 0 || anchorGray.rows() <= 0) {
            anchorGray.release()
            requestRedetectionAfterTrackingLoss()
            return false
        }

        val metricsSession = metricsController.startMetricsSession(event)
        val sessionId = trackingSessionController.beginSession(results, anchorGray, replaceTracking)
        lastTrackingSeedAtMs = event.ocrCompletedAtMs
        val selectedSourceCode = languagePanelController.selectedSourceLanguageCode
        val targetLanguage = languagePanelController.selectedTargetLanguageCode
        val pendingBatches = HashMap<String, MutableList<Pair<Int, String>>>()

        results.forEachIndexed { index, result ->
            val text = result.text

            translationCache[text]?.let { cached ->
                trackingSessionController.applyTranslationResult(sessionId, index, cached)
                return@forEachIndexed
            }

            val sourceLanguage = LanguageUiLogic.resolveSourceLanguage(selectedSourceCode, result.language)
            if (LanguageUiLogic.shouldBypassTranslation(sourceLanguage, targetLanguage, text)) {
                trackingSessionController.applyTranslationResult(sessionId, index, text)
                return@forEachIndexed
            }

            pendingBatches.getOrPut(sourceLanguage) { mutableListOf() }
                .add(index to text)
        }

        pendingBatches.forEach { (sourceLanguage, batch) ->
            val batchTexts = batch.map { it.second }
            metricsController.markTranslationBatchStarted(metricsSession)
            processingIndicatorController.markTranslationStarted()
            try {
                translationService.translateBatch(
                    texts = batchTexts,
                    sourceLanguage = sourceLanguage,
                    targetLanguage = targetLanguage,
                    onSuccess = { translatedBatch ->
                        try {
                            val count = minOf(batch.size, translatedBatch.size)
                            for (i in 0 until count) {
                                val (index, originalText) = batch[i]
                                val translated = translatedBatch[i]
                                translationCache[originalText] = translated
                                trackingSessionController.applyTranslationResult(
                                    sessionId,
                                    index,
                                    translated
                                )
                            }
                        } finally {
                            metricsController.markTranslationBatchFinished(metricsSession)
                            processingIndicatorController.markTranslationFinished()
                        }
                    },
                    onError = { error ->
                        Log.e(TAG, "Batch translation failed for $sourceLanguage->$targetLanguage", error)
                        metricsController.markTranslationBatchFinished(metricsSession)
                        processingIndicatorController.markTranslationFinished()
                    }
                )
            } catch (e: Exception) {
                Log.e(TAG, "Batch translation setup failed for $sourceLanguage->$targetLanguage", e)
                metricsController.markTranslationBatchFinished(metricsSession)
                processingIndicatorController.markTranslationFinished()
            }
        }

        if (pendingBatches.isEmpty()) {
            metricsController.finalizeTranslationMetrics(metricsSession, event.ocrCompletedAtMs)
        }
        return true
    }

    private companion object {
        const val TAG = "MainOcrCoordinator"
        const val PROCESS_INTERVAL_MS = 200L
        const val TRACKING_RESYNC_INTERVAL_MS = 1200L
        const val REDETECT_SETTLE_DELAY_MS = 350L
    }
}
