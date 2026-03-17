package com.example.translatorapp

import android.os.SystemClock
import com.example.translatorapp.databinding.ActivityMainBinding
import com.example.translatorapp.ocr.OcrService
import java.util.Locale

internal data class MetricsSession(
    val frameCapturedAtMs: Long,
    val ocrCompletedAtMs: Long,
    var pendingTranslationBatches: Int = 0,
    var translationCompletedAtMs: Long? = null,
    var overlayShownAtMs: Long? = null
)

internal class MetricsController(
    private val binding: ActivityMainBinding
) {
    private class FpsTracker {
        private var windowStartAtMs = 0L
        private var framesInWindow = 0

        fun reset(startAtMs: Long = SystemClock.elapsedRealtime()) {
            windowStartAtMs = startAtMs
            framesInWindow = 0
        }

        fun tick(nowAtMs: Long = SystemClock.elapsedRealtime()): Double? {
            if (windowStartAtMs == 0L) {
                windowStartAtMs = nowAtMs
            }
            framesInWindow++
            val elapsedMs = nowAtMs - windowStartAtMs
            if (elapsedMs < 1000L) return null

            val fps = framesInWindow * 1000.0 / elapsedMs
            windowStartAtMs = nowAtMs
            framesInWindow = 0
            return fps
        }
    }

    private val metricsLock = Any()
    private val analyzerFpsTracker = FpsTracker()
    private val ocrFpsTracker = FpsTracker()
    private val overlayFpsTracker = FpsTracker()

    private var analyzerFps = 0.0
    private var ocrFps = 0.0
    private var overlayFps = 0.0
    private var timeToFirstResultMs: Long? = null
    private var ocrLatencyMs: Long? = null
    private var translationLatencyMs: Long? = null
    private var endToEndLatencyMs: Long? = null
    private var cameraSessionStartedAtMs = 0L
    private var firstResultShown = false
    private var activeMetricsSession: MetricsSession? = null

    fun render(force: Boolean = false) {
        updateMetricsOverlay(force)
    }

    fun resetRuntimeMetrics() {
        val now = SystemClock.elapsedRealtime()
        synchronized(metricsLock) {
            analyzerFpsTracker.reset(now)
            ocrFpsTracker.reset(now)
            overlayFpsTracker.reset(now)
            analyzerFps = 0.0
            ocrFps = 0.0
            overlayFps = 0.0
            timeToFirstResultMs = null
            ocrLatencyMs = null
            translationLatencyMs = null
            endToEndLatencyMs = null
            cameraSessionStartedAtMs = now
            firstResultShown = false
            activeMetricsSession = null
        }
        updateMetricsOverlay(force = true)
    }

    fun recordAnalyzerFrame(analyzedAtMs: Long) {
        val fps = synchronized(metricsLock) { analyzerFpsTracker.tick(analyzedAtMs) } ?: return
        analyzerFps = fps
        updateMetricsOverlay()
    }

    fun recordOcrEvent(event: OcrService.OcrDetectionEvent) {
        val fps = synchronized(metricsLock) { ocrFpsTracker.tick(event.ocrCompletedAtMs) }
        if (fps != null) {
            ocrFps = fps
        }
        ocrLatencyMs = (event.ocrCompletedAtMs - event.stableAtMs).coerceAtLeast(0L)
        updateMetricsOverlay()
    }

    fun startMetricsSession(event: OcrService.OcrDetectionEvent): MetricsSession {
        val session = MetricsSession(
            frameCapturedAtMs = event.frameCapturedAtMs,
            ocrCompletedAtMs = event.ocrCompletedAtMs
        )
        synchronized(metricsLock) {
            translationLatencyMs = null
            endToEndLatencyMs = null
            activeMetricsSession = session
        }
        updateMetricsOverlay()
        return session
    }

    fun markTranslationBatchStarted(session: MetricsSession) {
        synchronized(metricsLock) {
            if (activeMetricsSession === session) {
                session.pendingTranslationBatches++
            }
        }
    }

    fun finalizeTranslationMetrics(session: MetricsSession, completedAtMs: Long) {
        val updated = synchronized(metricsLock) {
            if (activeMetricsSession !== session || session.translationCompletedAtMs != null) {
                false
            } else {
                session.translationCompletedAtMs = completedAtMs
                translationLatencyMs = (completedAtMs - session.ocrCompletedAtMs).coerceAtLeast(0L)
                true
            }
        }
        if (updated) {
            updateMetricsOverlay()
        }
    }

    fun markTranslationBatchFinished(session: MetricsSession) {
        val completedAtMs = SystemClock.elapsedRealtime()
        val shouldFinalize = synchronized(metricsLock) {
            if (activeMetricsSession !== session) {
                false
            } else {
                if (session.pendingTranslationBatches > 0) {
                    session.pendingTranslationBatches--
                }
                session.pendingTranslationBatches == 0 && session.translationCompletedAtMs == null
            }
        }
        if (shouldFinalize) {
            finalizeTranslationMetrics(session, completedAtMs)
        }
    }

    fun recordOverlayRendered() {
        val now = SystemClock.elapsedRealtime()
        val fps = synchronized(metricsLock) { overlayFpsTracker.tick(now) }
        var latenciesUpdated = false

        synchronized(metricsLock) {
            if (fps != null) {
                overlayFps = fps
            }

            val session = activeMetricsSession
            if (session != null && session.overlayShownAtMs == null) {
                session.overlayShownAtMs = now
                endToEndLatencyMs = (now - session.frameCapturedAtMs).coerceAtLeast(0L)
                if (!firstResultShown && cameraSessionStartedAtMs > 0L) {
                    timeToFirstResultMs = (now - cameraSessionStartedAtMs).coerceAtLeast(0L)
                    firstResultShown = true
                }
                latenciesUpdated = true
            }
        }

        if (fps != null || latenciesUpdated) {
            updateMetricsOverlay()
        }
    }

    fun clearActiveSession() {
        synchronized(metricsLock) {
            activeMetricsSession = null
        }
    }

    private fun updateMetricsOverlay(force: Boolean = false) {
        if (!force && !binding.root.isAttachedToWindow) return

        val metricsText = buildString {
            append("FPS: ")
            append(formatFps(analyzerFps))
            append('\n')
            append("First Result: ")
            append(formatDuration(timeToFirstResultMs))
            append('\n')
            append("OCR Latency: ")
            append(formatDuration(ocrLatencyMs))
            append('\n')
            append("Translation: ")
            append(formatDuration(translationLatencyMs))
            append('\n')
            append("End-to-End: ")
            append(formatDuration(endToEndLatencyMs))
        }

        binding.metricsText.post {
            binding.metricsText.text = metricsText
        }
    }

    private fun formatFps(value: Double): String {
        return if (value > 0.0) {
            String.format(Locale.US, "%.1f", value)
        } else {
            "--"
        }
    }

    private fun formatDuration(valueMs: Long?): String {
        return valueMs?.let { "${it} ms" } ?: "--"
    }
}
