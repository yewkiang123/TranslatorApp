package com.example.translatorapp

import android.Manifest
import android.animation.Animator
import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.animation.PropertyValuesHolder
import android.animation.ValueAnimator
import android.content.ContentValues
import android.content.res.ColorStateList
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.provider.MediaStore
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.view.ViewGroup
import android.view.animation.LinearInterpolator
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.interpolator.view.animation.FastOutSlowInInterpolator
import androidx.lifecycle.lifecycleScope
import com.example.translatorapp.camera.CameraController
import com.example.translatorapp.databinding.ActivityMainBinding
import com.example.translatorapp.logic.LanguageUiLogic
import com.example.translatorapp.ocr.OcrService
import com.example.translatorapp.translate.TranslationService
import com.google.mlkit.nl.translate.TranslateLanguage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.launch
import java.util.Locale
import java.util.concurrent.Executors
import com.example.translatorapp.ocr.FrameOcrRepository
import com.example.translatorapp.ocr.SceneTextReplacer
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.collectLatest
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import kotlinx.coroutines.flow.conflate
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.roundToInt
import kotlin.math.sin


class MainActivity : AppCompatActivity(), AdapterView.OnItemSelectedListener {
    private enum class ProcessingIndicatorState {
        IDLE,
        DETECTING,
        NO_TEXT,
        TRANSLATING,
        GENERATING,
        DISABLED
    }

    private data class ProcessingIndicatorStyle(
        val labelRes: Int,
        val backgroundColorRes: Int,
        val textColorRes: Int,
        val spinnerColorRes: Int,
        val spinnerVisible: Boolean
    )

    private data class LanguageOption(
        val label: String,
        val code: String
    )

    private data class TrackingTransform(
        val m00: Double = 1.0,
        val m01: Double = 0.0,
        val m02: Double = 0.0,
        val m10: Double = 0.0,
        val m11: Double = 1.0,
        val m12: Double = 0.0
    ) {
        val dx: Double
            get() = m02

        val dy: Double
            get() = m12

        val scale: Double
            get() = hypot(m00, m10)
    }

    private data class TrackingSnapshot(
        val referenceResults: List<OcrService.DetectionResult>,
        val baseResults: List<OcrService.DetectionResult>,
        val translations: List<String>,
        val anchorGray: Mat,
        val accumulatedTransform: TrackingTransform
    )

    private data class TrackingMotionStats(
        val prevCentroid: Point,
        val nextCentroid: Point,
        val scaleHint: Double?
    )

    private data class MetricsSession(
        val frameCapturedAtMs: Long,
        val ocrCompletedAtMs: Long,
        var pendingTranslationBatches: Int = 0,
        var translationCompletedAtMs: Long? = null,
        var overlayShownAtMs: Long? = null
    )

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

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraController: CameraController
    private lateinit var ocrService: OcrService
    private lateinit var translationService: TranslationService

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var lastProcessTime = 0L
    private val processInterval = 200L
    private var targetLanguageCode = "en"
    private var sourceLanguageDisplay = "Detected Language"
    private val targetLanguageOptions = mutableListOf<LanguageOption>()
    private val sourceLanguageOptions = mutableListOf<LanguageOption>()
    private val downloadedLanguageCodes = mutableSetOf<String>()
    private val downloadingLanguageCodes = mutableSetOf<String>()
    private val languageCodeByDisplay = HashMap<String, String>()
    private val translationCache = HashMap<String, String>()
    private lateinit var sourceSpinnerAdapter: SourceLanguageAdapter
    private lateinit var targetSpinnerAdapter: TargetLanguageAdapter

    private var renderJob: Job? = null
    private var trackingJob: Job? = null
    private var pendingRedetectJob: Job? = null
    @Volatile private var overlayEnabled = false
    @Volatile private var needsFreshInpaint = true
    @Volatile private var freezeFrameMode = false
    private var frozenFrameBitmap: Bitmap? = null
    private val trackingLock = Any()
    private var trackedResults: MutableList<OcrService.DetectionResult> = mutableListOf()
    private var trackedBaseResults: MutableList<OcrService.DetectionResult> = mutableListOf()
    private var trackedTranslations: MutableList<String> = mutableListOf()
    private var nextStableDetectionId = 1L
    private var trackingAnchorGray: Mat? = null
    private var trackingAccumulatedTransform = TrackingTransform()
    private var trackingMissCount = 0
    @Volatile private var cameraMotionStable = false
    @Volatile private var redetectAfterMotionStops = true
    private val trackIntervalMs = 45L
    private val redetectSettleDelayMs = 350L
    private val maxTrackingMisses = 8
    private val panThresholdPx = 220.0
    private val minTrackShiftPx = 1.5
    private val maxTrackStepPx = 96.0
    private val minScaleDelta = 0.015
    private val minTrackScale = 0.82
    private val maxTrackScale = 1.22
    private val zoomResetThreshold = 0.28
    private val trackingFeatureCount = 420
    private val trackingFeatureQualityLevel = 0.005
    private val trackingFeatureMinDistance = 4.0
    private val minTrackingRoiSizePx = 180
    private val minTrackingTargetTextSizePx = 56.0
    private val maxTrackingUpscale = 4.0
    private val trackingFlowWindowSize = Size(41.0, 41.0)
    private val trackingFlowMaxLevel = 5
    private val minTrackingValidPoints = 6
    private val minTrackingTranslationPoints = 3
    private val minTrackingAffineInliers = 4
    private val minTrackingRegionSizePx = 96
    private val minTrackingScaleRadiusPx = 8.0
    private val maxTrackRotationDegrees = 6.0
    private val trackingFlowCriteria = TermCriteria(
        TermCriteria.COUNT or TermCriteria.EPS,
        20,
        0.03
    )

    private val replacer = SceneTextReplacer()
    private lateinit var scaleGestureDetector: ScaleGestureDetector
    private lateinit var freezeGestureDetector: GestureDetector
    private val freezeImageMatrix = Matrix()
    private var freezeCurrentScale = 1f
    private var freezeMinScale = 1f
    private var freezeMaxScale = 1f
    private var freezeImageWidth = 0f
    private var freezeImageHeight = 0f
    private var lastTouchX = 0f
    private var lastTouchY = 0f
    private var isFreezeDragging = false
    private var isOpenCvReady = false
    private var hasShownOpenCvUnavailableWarning = false
    private val processingStateLock = Any()
    private val metricsLock = Any()
    private var detectingActive = false
    private var noTextDetected = false
    private var translatingJobsInFlight = 0
    private var generatingJobsInFlight = 0
    private var generatedTextVisible = false
    private var lastProcessingIndicatorState: ProcessingIndicatorState? = null
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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize OpenCV FIRST before anything else
        isOpenCvReady = initializeOpenCV()

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        updateFreezeFrameActionButtons()
        renderProcessingIndicator(force = true)
        updateMetricsOverlay(force = true)

        initServices()
        setupOcrListener()
        setupSpinners()
        setupCamera()
        setupClickListeners()
        setupBackPressHandler()
        setupFreezeFrameZoom()
        observeProcessedFrames()
        startTrackingLoop()

        if (allPermissionsGranted()) {
            startCameraWithOcr()
        } else {
            requestPermissions()
        }
    }

    private fun initializeOpenCV(): Boolean {
        return try {
            if (!OpenCVInitializer.init()) {
                Log.e("OpenCV", "OpenCV initialization failed!")
                Toast.makeText(this, "OpenCV failed to load", Toast.LENGTH_SHORT).show()
                false
            } else {
                Log.d("OpenCV", "OpenCV initialized successfully")
                true
            }
        } catch (e: UnsatisfiedLinkError) {
            Log.e("OpenCV", "OpenCV library not found", e)
            Toast.makeText(this, "OpenCV library missing", Toast.LENGTH_LONG).show()
            false
        }
    }

    private fun initServices() {
        ocrService = OcrService()
        ocrService.setSourceLanguageHint("auto")
        translationService = TranslationService()
    }

    private fun setupOcrListener() {
        ocrService.analyzerFrames.onEach { analyzedAtMs ->
            if (!freezeFrameMode) {
                recordAnalyzerFrame(analyzedAtMs)
            }
        }.launchIn(lifecycleScope)

        ocrService.detectionEvents.onEach { event ->
            if (!freezeFrameMode) {
                recordOcrEvent(event)
            }
            if (event.results.isNotEmpty()) {
                setNoTextDetected(false)
                processOnInterval(event)
            } else if (!freezeFrameMode) {
                showNoTextDetected()
            }
        }.launchIn(lifecycleScope)

        ocrService.motionStable.onEach { stable ->
            if (!freezeFrameMode) {
                cameraMotionStable = stable
                if (stable) {
                    if (redetectAfterMotionStops) {
                        scheduleRedetectionAfterSettle()
                    } else {
                        setDetectingActive(false)
                    }
                } else {
                    cancelPendingRedetection()
                    hideOverlayOnly()
                    redetectAfterMotionStops = true
                    setNoTextDetected(false)
                    setDetectingActive(false)
                }
            }
        }.launchIn(lifecycleScope)
    }

    private fun setDetectingActive(active: Boolean) {
        synchronized(processingStateLock) {
            detectingActive = active
            if (active) {
                noTextDetected = false
            }
        }
        renderProcessingIndicator()
    }

    private fun setNoTextDetected(active: Boolean) {
        synchronized(processingStateLock) {
            noTextDetected = active
            if (active) {
                detectingActive = false
            }
        }
        renderProcessingIndicator()
    }

    private fun scheduleRedetectionAfterSettle() {
        pendingRedetectJob?.cancel()
        pendingRedetectJob = lifecycleScope.launch {
            kotlinx.coroutines.delay(redetectSettleDelayMs)
            if (!isActive || freezeFrameMode || !cameraMotionStable || !redetectAfterMotionStops) {
                return@launch
            }
            ocrService.requestImmediateRefresh()
            lastProcessTime = 0L
            redetectAfterMotionStops = false
            setDetectingActive(true)
        }
    }

    private fun cancelPendingRedetection() {
        pendingRedetectJob?.cancel()
        pendingRedetectJob = null
    }

    private fun showNoTextDetected() {
        synchronized(processingStateLock) {
            noTextDetected = true
            detectingActive = false
        }
        renderProcessingIndicator(force = true)
    }

    private fun resetRuntimeMetrics() {
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

    private fun recordAnalyzerFrame(analyzedAtMs: Long) {
        val fps = synchronized(metricsLock) { analyzerFpsTracker.tick(analyzedAtMs) } ?: return
        analyzerFps = fps
        updateMetricsOverlay()
    }

    private fun recordOcrEvent(event: OcrService.OcrDetectionEvent) {
        val fps = synchronized(metricsLock) { ocrFpsTracker.tick(event.ocrCompletedAtMs) }
        if (fps != null) {
            ocrFps = fps
        }
        ocrLatencyMs = (event.ocrCompletedAtMs - event.stableAtMs).coerceAtLeast(0L)
        updateMetricsOverlay()
    }

    private fun startMetricsSession(event: OcrService.OcrDetectionEvent): MetricsSession {
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

    private fun markTranslationBatchStarted(session: MetricsSession) {
        synchronized(metricsLock) {
            if (activeMetricsSession === session) {
                session.pendingTranslationBatches++
            }
        }
    }

    private fun finalizeTranslationMetrics(session: MetricsSession, completedAtMs: Long) {
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

    private fun markTranslationBatchFinished(session: MetricsSession) {
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

    private fun recordOverlayRendered() {
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

    private fun updateMetricsOverlay(force: Boolean = false) {
        if (!force && !::binding.isInitialized) return

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

        runOnUiThread {
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

    private fun resetProcessingWork(keepDetecting: Boolean) {
        synchronized(processingStateLock) {
            translatingJobsInFlight = 0
            generatingJobsInFlight = 0
            generatedTextVisible = false
            detectingActive = keepDetecting
            noTextDetected = false
        }
        renderProcessingIndicator(force = true)
    }

    private fun markTranslationStarted() {
        synchronized(processingStateLock) {
            translatingJobsInFlight++
        }
        renderProcessingIndicator()
    }

    private fun markTranslationFinished() {
        synchronized(processingStateLock) {
            if (translatingJobsInFlight > 0) {
                translatingJobsInFlight--
            }
        }
        renderProcessingIndicator()
    }

    private fun setGeneratingActive(active: Boolean) {
        synchronized(processingStateLock) {
            if (active) {
                generatingJobsInFlight++
            } else if (generatingJobsInFlight > 0) {
                generatingJobsInFlight--
            }
        }
        renderProcessingIndicator()
    }

    private fun setGeneratedTextVisible(visible: Boolean) {
        synchronized(processingStateLock) {
            generatedTextVisible = visible
        }
        renderProcessingIndicator()
    }

    private fun resolveProcessingIndicatorState(): ProcessingIndicatorState {
        return synchronized(processingStateLock) {
            when {
                generatedTextVisible -> ProcessingIndicatorState.IDLE
                !isOpenCvReady -> ProcessingIndicatorState.DISABLED
                generatingJobsInFlight > 0 -> ProcessingIndicatorState.GENERATING
                translatingJobsInFlight > 0 -> ProcessingIndicatorState.TRANSLATING
                noTextDetected -> ProcessingIndicatorState.NO_TEXT
                detectingActive -> ProcessingIndicatorState.DETECTING
                else -> ProcessingIndicatorState.IDLE
            }
        }
    }

    private fun styleForProcessingState(state: ProcessingIndicatorState): ProcessingIndicatorStyle {
        return when (state) {
            ProcessingIndicatorState.IDLE -> ProcessingIndicatorStyle(
                labelRes = R.string.status_ready,
                backgroundColorRes = R.color.status_idle_bg,
                textColorRes = R.color.status_idle_text,
                spinnerColorRes = R.color.status_spinner_idle,
                spinnerVisible = false
            )

            ProcessingIndicatorState.DETECTING -> ProcessingIndicatorStyle(
                labelRes = R.string.status_detecting_text,
                backgroundColorRes = R.color.status_detect_bg,
                textColorRes = R.color.status_detect_text,
                spinnerColorRes = R.color.status_spinner_detect,
                spinnerVisible = true
            )

            ProcessingIndicatorState.NO_TEXT -> ProcessingIndicatorStyle(
                labelRes = R.string.status_no_text_detected,
                backgroundColorRes = R.color.status_idle_bg,
                textColorRes = R.color.status_idle_text,
                spinnerColorRes = R.color.status_spinner_idle,
                spinnerVisible = false
            )

            ProcessingIndicatorState.TRANSLATING -> ProcessingIndicatorStyle(
                labelRes = R.string.status_translating,
                backgroundColorRes = R.color.status_translate_bg,
                textColorRes = R.color.status_translate_text,
                spinnerColorRes = R.color.status_spinner_translate,
                spinnerVisible = true
            )

            ProcessingIndicatorState.GENERATING -> ProcessingIndicatorStyle(
                labelRes = R.string.status_generating_image,
                backgroundColorRes = R.color.status_generate_bg,
                textColorRes = R.color.status_generate_text,
                spinnerColorRes = R.color.status_spinner_generate,
                spinnerVisible = true
            )

            ProcessingIndicatorState.DISABLED -> ProcessingIndicatorStyle(
                labelRes = R.string.status_ocr_disabled,
                backgroundColorRes = R.color.status_disabled_bg,
                textColorRes = R.color.status_disabled_text,
                spinnerColorRes = R.color.status_spinner_idle,
                spinnerVisible = false
            )
        }
    }

    private fun renderProcessingIndicator(force: Boolean = false) {
        val state = resolveProcessingIndicatorState()
        if (!force && state == lastProcessingIndicatorState) return
        lastProcessingIndicatorState = state
        val style = styleForProcessingState(state)

        runOnUiThread {
            binding.processingIndicatorText.setText(style.labelRes)
            binding.processingIndicatorText.setTextColor(
                ContextCompat.getColor(this, style.textColorRes)
            )
            binding.processingIndicatorCard.setCardBackgroundColor(
                ContextCompat.getColor(this, style.backgroundColorRes)
            )
            binding.processingIndicatorProgress.visibility =
                if (style.spinnerVisible) View.VISIBLE else View.GONE
            binding.processingIndicatorProgress.indeterminateTintList =
                ColorStateList.valueOf(ContextCompat.getColor(this, style.spinnerColorRes))
        }
    }

    private var reusableBitmap: Bitmap? = null

    private fun observeProcessedFrames() {

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
                    if (freezeFrameMode) {
                        mat.release()
                        return@collectLatest
                    }

                    val safeMat = mat

                    val w = safeMat.cols()
                    val h = safeMat.rows()
                    if (w <= 0 || h <= 0) {
                        safeMat.release()
                        return@collectLatest
                    }

                    // Heavy work off main thread
                    val bmp = withContext(Dispatchers.Default) {
                        try {
                            val rgba = Mat()
                            when (safeMat.channels()) {
                                4 -> safeMat.copyTo(rgba)
                                3 -> Imgproc.cvtColor(safeMat, rgba, Imgproc.COLOR_BGR2RGBA)
                                1 -> Imgproc.cvtColor(safeMat, rgba, Imgproc.COLOR_GRAY2RGBA)
                                else -> {
                                    safeMat.release()
                                    rgba.release()
                                    return@withContext null
                                }
                            }

                            // Reuse bitmap to reduce GC + flicker
                            val out = reusableBitmap?.takeIf { it.width == w && it.height == h }
                                ?: Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also {
                                    reusableBitmap = it
                                }

                            org.opencv.android.Utils.matToBitmap(rgba, out)

                            rgba.release()
                            safeMat.release()

                            out
                        } catch (e: Exception) {
                            safeMat.release()
                            Log.e(TAG, "render failed", e)
                            null
                        }
                    }

                    if (bmp != null && overlayEnabled && !freezeFrameMode) {
                        binding.processedOverlay.setImageBitmap(bmp)
                        recordOverlayRendered()
                    }
                }
        }
    }

    private fun startTrackingLoop() {
        trackingJob?.cancel()
        trackingJob = lifecycleScope.launch(Dispatchers.Default) {
            while (isActive) {
                val loopStartedAt = SystemClock.elapsedRealtime()
                if (freezeFrameMode) {
                    kotlinx.coroutines.delay(trackIntervalMs)
                    continue
                }
                val frame = FrameOcrRepository.snapshotLatestCameraFrame()
                if (frame != null && !frame.empty()) {
                    trackAndRender(frame)
                } else {
                    frame?.release()
                }
                val remainingDelay = trackIntervalMs - (SystemClock.elapsedRealtime() - loopStartedAt)
                if (remainingDelay > 0L) {
                    kotlinx.coroutines.delay(remainingDelay)
                }
            }
        }
    }

    private fun trackAndRender(frame: Mat) {
        if (freezeFrameMode) {
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

        val referencePivot = computeTrackingPivot(
            snapshot.referenceResults,
            frame.cols(),
            frame.rows()
        )
        val currentGray = toGray(frame)
        val estimatedTransform = estimateTrackingTransform(
            prev = snapshot.anchorGray,
            current = currentGray,
            referenceResults = snapshot.referenceResults,
            referencePivot = referencePivot
        )
        snapshot.anchorGray.release()

        val (stepTransform, accumulatedTransform, missCount) = synchronized(trackingLock) {
            if (estimatedTransform != null) {
                trackingAccumulatedTransform = composeTransforms(
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

        if (estimatedTransform == null && missCount > maxTrackingMisses) {
            currentGray.release()
            clearTrackingState(clearDisplayedOverlay = false)
            requestRedetectionAfterTrackingLoss()
            frame.release()
            return
        }

        if (measureTransformMotion(stepTransform, referencePivot) > panThresholdPx ||
            abs(stepTransform.scale - 1.0) > zoomResetThreshold) {
            currentGray.release()
            clearTrackingState(clearDisplayedOverlay = false)
            requestRedetectionAfterTrackingLoss()
            frame.release()
            return
        }

        val shifted = applyTrackingTransform(
            results = snapshot.baseResults,
            transform = accumulatedTransform,
            maxW = frame.cols(),
            maxH = frame.rows()
        )

        if (shifted.none { it.boundingBox != null }) {
            currentGray.release()
            clearTrackingState()
            requestRedetectionAfterTrackingLoss()
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

        setGeneratingActive(true)
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
            setGeneratedTextVisible(true)
            runOnUiThread {
                binding.processedOverlay.visibility = View.VISIBLE
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed generating processed frame", e)
        } finally {
            setGeneratingActive(false)
            frame.release()
        }
    }

    private fun hideOverlayOnly() {
        if (freezeFrameMode) return
        if (!overlayEnabled) return

        overlayEnabled = false
        setGeneratedTextVisible(false)
        FrameOcrRepository.clearFrame()
        runOnUiThread {
            binding.processedOverlay.setImageBitmap(null)
            binding.processedOverlay.visibility = View.INVISIBLE
        }
    }

    private fun toGray(src: Mat): Mat {
        val gray = Mat()
        when (src.channels()) {
            4 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            3 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            1 -> src.copyTo(gray)
            else -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
        }
        return gray
    }

    private fun estimateTrackingTransform(
        prev: Mat?,
        current: Mat,
        referenceResults: List<OcrService.DetectionResult>,
        referencePivot: Point
    ): TrackingTransform? {
        if (prev == null || prev.empty()) return null

        val trackingRegion = computeTrackingRegion(prev.cols(), prev.rows(), referenceResults)
        val trackingScale = computeTrackingUpscale(referenceResults)
        val prevTracking = createTrackingInput(prev, trackingRegion, trackingScale)
        val currentTracking = createTrackingInput(current, trackingRegion, trackingScale)
        val points = MatOfPoint()
        val trackingMask = buildTrackingMask(
            prevTracking.cols(),
            prevTracking.rows(),
            referenceResults,
            trackingRegion,
            trackingScale
        )
        try {
            if (trackingMask != null) {
                Imgproc.goodFeaturesToTrack(
                    prevTracking,
                    points,
                    trackingFeatureCount,
                    trackingFeatureQualityLevel,
                    trackingFeatureMinDistance,
                    trackingMask
                )
            } else {
                Imgproc.goodFeaturesToTrack(
                    prevTracking,
                    points,
                    trackingFeatureCount,
                    trackingFeatureQualityLevel,
                    trackingFeatureMinDistance
                )
            }
            if (points.empty() && trackingMask != null) {
                Imgproc.goodFeaturesToTrack(
                    prevTracking,
                    points,
                    trackingFeatureCount,
                    trackingFeatureQualityLevel,
                    trackingFeatureMinDistance
                )
            }
        } finally {
            trackingMask?.release()
        }
        if (points.empty()) {
            prevTracking.release()
            currentTracking.release()
            points.release()
            return estimateGlobalTranslationFallback(prev, current, referencePivot)
        }

        val prevPts = MatOfPoint2f(*points.toArray())
        val nextPts = MatOfPoint2f()
        val status = MatOfByte()
        val err = MatOfFloat()

        Video.calcOpticalFlowPyrLK(
            prevTracking,
            currentTracking,
            prevPts,
            nextPts,
            status,
            err,
            trackingFlowWindowSize,
            trackingFlowMaxLevel,
            trackingFlowCriteria
        )

        val prevArr = prevPts.toArray()
        val nextArr = nextPts.toArray()
        val statusArr = status.toArray()

        prevTracking.release()
        currentTracking.release()
        prevPts.release()
        nextPts.release()
        status.release()
        err.release()
        points.release()

        val prevValid = ArrayList<org.opencv.core.Point>(statusArr.size)
        val nextValid = ArrayList<org.opencv.core.Point>(statusArr.size)

        for (i in statusArr.indices) {
            if (statusArr[i].toInt() != 1) continue
            val from = trackingPointToFrame(prevArr[i], trackingRegion, trackingScale)
            val to = trackingPointToFrame(nextArr[i], trackingRegion, trackingScale)
            if (!from.x.isFinite() || !from.y.isFinite() || !to.x.isFinite() || !to.y.isFinite()) continue
            prevValid.add(from)
            nextValid.add(to)
        }

        if (prevValid.size < minTrackingTranslationPoints) {
            return estimateGlobalTranslationFallback(prev, current, referencePivot)
        }

        val motionStats = computeTrackingMotionStats(prevValid, nextValid)
        if (prevValid.size < minTrackingValidPoints) {
            return estimateTranslationOnlyTransform(
                prevPoints = prevValid,
                nextPoints = nextValid,
                referencePivot = referencePivot,
                motionStats = motionStats
            )
        }

        val prevInliers = MatOfPoint2f(*prevValid.toTypedArray())
        val nextInliers = MatOfPoint2f(*nextValid.toTypedArray())
        val inlierMask = Mat()
        val affine = Calib3d.estimateAffinePartial2D(
            prevInliers,
            nextInliers,
            inlierMask,
            Calib3d.RANSAC,
            3.0,
            2000L,
            0.99,
            10L
        )

        val inlierCount = if (!inlierMask.empty()) {
            Core.countNonZero(inlierMask)
        } else {
            prevValid.size
        }

        prevInliers.release()
        nextInliers.release()
        inlierMask.release()

        if (affine.empty() || affine.rows() < 2 || affine.cols() < 3) {
            affine.release()
            return estimateTranslationOnlyTransform(
                prevPoints = prevValid,
                nextPoints = nextValid,
                referencePivot = referencePivot,
                motionStats = motionStats
            ) ?: estimateGlobalTranslationFallback(prev, current, referencePivot)
        }
        if (inlierCount < minTrackingAffineInliers) {
            affine.release()
            return estimateTranslationOnlyTransform(
                prevPoints = prevValid,
                nextPoints = nextValid,
                referencePivot = referencePivot,
                motionStats = motionStats
            ) ?: estimateGlobalTranslationFallback(prev, current, referencePivot)
        }

        val rawTransform = TrackingTransform(
            m00 = affine.readDouble(0, 0) ?: 1.0,
            m01 = affine.readDouble(0, 1) ?: 0.0,
            m02 = affine.readDouble(0, 2) ?: 0.0,
            m10 = affine.readDouble(1, 0) ?: 0.0,
            m11 = affine.readDouble(1, 1) ?: 1.0,
            m12 = affine.readDouble(1, 2) ?: 0.0
        )
        affine.release()

        return sanitizeTrackingTransform(
            raw = rawTransform,
            referencePivot = referencePivot,
            motionStats = motionStats
        ) ?: estimateGlobalTranslationFallback(prev, current, referencePivot)
    }

    private fun estimateGlobalTranslationFallback(
        prev: Mat,
        current: Mat,
        referencePivot: Point
    ): TrackingTransform? {
        val points = MatOfPoint()
        try {
            Imgproc.goodFeaturesToTrack(
                prev,
                points,
                maxOf(trackingFeatureCount / 2, 120),
                trackingFeatureQualityLevel,
                trackingFeatureMinDistance
            )
            if (points.empty()) {
                return null
            }

            val prevPts = MatOfPoint2f(*points.toArray())
            val nextPts = MatOfPoint2f()
            val status = MatOfByte()
            val err = MatOfFloat()

            try {
                Video.calcOpticalFlowPyrLK(
                    prev,
                    current,
                    prevPts,
                    nextPts,
                    status,
                    err,
                    Size(51.0, 51.0),
                    trackingFlowMaxLevel + 1,
                    trackingFlowCriteria
                )

                val prevArr = prevPts.toArray()
                val nextArr = nextPts.toArray()
                val statusArr = status.toArray()

                val prevValid = ArrayList<Point>(statusArr.size)
                val nextValid = ArrayList<Point>(statusArr.size)
                for (i in statusArr.indices) {
                    if (statusArr[i].toInt() != 1) continue
                    val from = prevArr[i]
                    val to = nextArr[i]
                    if (!from.x.isFinite() || !from.y.isFinite() || !to.x.isFinite() || !to.y.isFinite()) continue
                    prevValid.add(from)
                    nextValid.add(to)
                }

                if (prevValid.size < minTrackingTranslationPoints) {
                    return null
                }

                val motionStats = computeTrackingMotionStats(prevValid, nextValid)
                return estimateTranslationOnlyTransform(
                    prevPoints = prevValid,
                    nextPoints = nextValid,
                    referencePivot = referencePivot,
                    motionStats = motionStats
                )
            } finally {
                prevPts.release()
                nextPts.release()
                status.release()
                err.release()
            }
        } finally {
            points.release()
        }
    }

    private fun buildTrackingMask(
        width: Int,
        height: Int,
        referenceResults: List<OcrService.DetectionResult>,
        trackingRegion: android.graphics.Rect,
        trackingScale: Double
    ): Mat? {
        if (width <= 0 || height <= 0 || referenceResults.isEmpty()) return null

        val mask = Mat.zeros(height, width, CvType.CV_8UC1)
        val fill = Scalar(255.0)
        var regions = 0

        referenceResults.forEach { result ->
            val box = result.blockBoundingBox ?: result.boundingBox ?: return@forEach
            val centerX = ((box.left + box.right) / 2.0 - trackingRegion.left) * trackingScale
            val centerY = ((box.top + box.bottom) / 2.0 - trackingRegion.top) * trackingScale
            val halfWidth = maxOf(
                (box.width() * 0.5 + box.width() * 0.5) * trackingScale,
                minTrackingRegionSizePx * trackingScale / 2.0
            )
            val halfHeight = maxOf(
                (box.height() * 0.5 + box.height() * 0.7) * trackingScale,
                minTrackingRegionSizePx * trackingScale / 2.0
            )
            val left = floor(centerX - halfWidth).toInt().coerceIn(0, width - 1)
            val top = floor(centerY - halfHeight).toInt().coerceIn(0, height - 1)
            val right = ceil(centerX + halfWidth).toInt().coerceIn(1, width)
            val bottom = ceil(centerY + halfHeight).toInt().coerceIn(1, height)
            if (right <= left || bottom <= top) return@forEach

            Imgproc.rectangle(
                mask,
                Point(left.toDouble(), top.toDouble()),
                Point(right.toDouble(), bottom.toDouble()),
                fill,
                -1
            )
            regions++
        }

        if (regions == 0) {
            mask.release()
            return null
        }

        return mask
    }

    private fun computeTrackingRegion(
        width: Int,
        height: Int,
        referenceResults: List<OcrService.DetectionResult>
    ): android.graphics.Rect {
        val boxes = referenceResults.mapNotNull { it.blockBoundingBox ?: it.boundingBox }
        if (boxes.isEmpty()) {
            return android.graphics.Rect(0, 0, width, height)
        }

        val contentLeft = boxes.minOf { it.left }.coerceIn(0, width - 1)
        val contentTop = boxes.minOf { it.top }.coerceIn(0, height - 1)
        val contentRight = boxes.maxOf { it.right }.coerceIn(1, width)
        val contentBottom = boxes.maxOf { it.bottom }.coerceIn(1, height)
        val contentWidth = (contentRight - contentLeft).coerceAtLeast(1)
        val contentHeight = (contentBottom - contentTop).coerceAtLeast(1)

        val padX = maxOf(contentWidth * 1.8, 84.0)
        val padY = maxOf(contentHeight * 1.9, 84.0)
        val targetWidth = maxOf(contentWidth + padX * 2.0, minTrackingRoiSizePx.toDouble())
        val targetHeight = maxOf(contentHeight + padY * 2.0, minTrackingRoiSizePx.toDouble())
        val centerX = (contentLeft + contentRight) / 2.0
        val centerY = (contentTop + contentBottom) / 2.0

        var left = floor(centerX - targetWidth / 2.0).toInt()
        var top = floor(centerY - targetHeight / 2.0).toInt()
        var right = ceil(centerX + targetWidth / 2.0).toInt()
        var bottom = ceil(centerY + targetHeight / 2.0).toInt()

        if (left < 0) {
            right -= left
            left = 0
        }
        if (top < 0) {
            bottom -= top
            top = 0
        }
        if (right > width) {
            val overflow = right - width
            left -= overflow
            right = width
        }
        if (bottom > height) {
            val overflow = bottom - height
            top -= overflow
            bottom = height
        }

        left = left.coerceIn(0, width - 1)
        top = top.coerceIn(0, height - 1)
        right = right.coerceIn(left + 1, width)
        bottom = bottom.coerceIn(top + 1, height)
        return android.graphics.Rect(left, top, right, bottom)
    }

    private fun computeTrackingUpscale(
        referenceResults: List<OcrService.DetectionResult>
    ): Double {
        val boxes = referenceResults.mapNotNull { it.blockBoundingBox ?: it.boundingBox }
        if (boxes.isEmpty()) return 1.0

        val contentLeft = boxes.minOf { it.left }
        val contentTop = boxes.minOf { it.top }
        val contentRight = boxes.maxOf { it.right }
        val contentBottom = boxes.maxOf { it.bottom }
        val contentWidth = (contentRight - contentLeft).coerceAtLeast(1)
        val contentHeight = (contentBottom - contentTop).coerceAtLeast(1)
        val contentMinDimension = minOf(contentWidth, contentHeight).toDouble()
        if (contentMinDimension <= 0.0) return 1.0

        return (minTrackingTargetTextSizePx / contentMinDimension).coerceIn(1.0, maxTrackingUpscale)
    }

    private fun createTrackingInput(
        source: Mat,
        trackingRegion: android.graphics.Rect,
        trackingScale: Double
    ): Mat {
        val roi = Mat(
            source,
            org.opencv.core.Rect(
                trackingRegion.left,
                trackingRegion.top,
                trackingRegion.width(),
                trackingRegion.height()
            )
        )
        return try {
            if (trackingScale > 1.01) {
                val resized = Mat()
                Imgproc.resize(
                    roi,
                    resized,
                    Size(),
                    trackingScale,
                    trackingScale,
                    Imgproc.INTER_CUBIC
                )
                resized
            } else {
                roi.clone()
            }
        } finally {
            roi.release()
        }
    }

    private fun trackingPointToFrame(
        point: Point,
        trackingRegion: android.graphics.Rect,
        trackingScale: Double
    ): Point {
        return Point(
            trackingRegion.left + point.x / trackingScale,
            trackingRegion.top + point.y / trackingScale
        )
    }

    private fun estimateTranslationOnlyTransform(
        prevPoints: List<Point>,
        nextPoints: List<Point>,
        referencePivot: Point,
        motionStats: TrackingMotionStats
    ): TrackingTransform? {
        if (prevPoints.size != nextPoints.size || prevPoints.size < minTrackingTranslationPoints) {
            return null
        }

        val dxSamples = ArrayList<Double>(prevPoints.size)
        val dySamples = ArrayList<Double>(prevPoints.size)
        for (i in prevPoints.indices) {
            dxSamples.add(nextPoints[i].x - prevPoints[i].x)
            dySamples.add(nextPoints[i].y - prevPoints[i].y)
        }

        val medianDx = median(dxSamples) ?: return null
        val medianDy = median(dySamples) ?: return null
        val rawTransform = TrackingTransform(
            m02 = medianDx,
            m12 = medianDy
        )
        return sanitizeTrackingTransform(
            raw = rawTransform,
            referencePivot = referencePivot,
            motionStats = motionStats.copy(scaleHint = 1.0)
        )
    }

    private fun sanitizeTrackingTransform(
        raw: TrackingTransform,
        referencePivot: Point,
        motionStats: TrackingMotionStats
    ): TrackingTransform? {
        val rawScale = raw.scale
        val scale = sanitizeScale(
            motionStats.scaleHint
                ?.takeIf { it.isFinite() && it > 0.0 }
                ?: rawScale
        )
        val rotation = sanitizeRotation(atan2(raw.m10, raw.m00))
        val cosValue = cos(rotation)
        val sinValue = sin(rotation)
        val m00 = scale * cosValue
        val m01 = -scale * sinValue
        val m10 = scale * sinValue
        val m11 = scale * cosValue

        val prevCentroid = motionStats.prevCentroid
        val nextCentroid = motionStats.nextCentroid
        val rawTargetPivotX = nextCentroid.x -
            (m00 * (prevCentroid.x - referencePivot.x) + m01 * (prevCentroid.y - referencePivot.y))
        val rawTargetPivotY = nextCentroid.y -
            (m10 * (prevCentroid.x - referencePivot.x) + m11 * (prevCentroid.y - referencePivot.y))
        if (!rawTargetPivotX.isFinite() || !rawTargetPivotY.isFinite()) return null

        val pivotDx = sanitizeShift(rawTargetPivotX - referencePivot.x)
        val pivotDy = sanitizeShift(rawTargetPivotY - referencePivot.y)
        val targetPivotX = referencePivot.x + pivotDx
        val targetPivotY = referencePivot.y + pivotDy
        val m02 = targetPivotX - (m00 * referencePivot.x + m01 * referencePivot.y)
        val m12 = targetPivotY - (m10 * referencePivot.x + m11 * referencePivot.y)

        return TrackingTransform(
            m00 = m00,
            m01 = m01,
            m02 = m02,
            m10 = m10,
            m11 = m11,
            m12 = m12
        )
    }

    private fun sanitizeShift(delta: Double): Double {
        if (!delta.isFinite()) return 0.0
        if (abs(delta) < minTrackShiftPx) return 0.0
        return delta.coerceIn(-maxTrackStepPx, maxTrackStepPx)
    }

    private fun computeTrackingPivot(
        referenceResults: List<OcrService.DetectionResult>,
        width: Int,
        height: Int
    ): Point {
        val boxes = referenceResults.mapNotNull { it.blockBoundingBox ?: it.boundingBox }
        if (boxes.isEmpty()) {
            return Point(width / 2.0, height / 2.0)
        }

        val left = boxes.minOf { it.left }.coerceIn(0, width - 1)
        val top = boxes.minOf { it.top }.coerceIn(0, height - 1)
        val right = boxes.maxOf { it.right }.coerceIn(1, width)
        val bottom = boxes.maxOf { it.bottom }.coerceIn(1, height)
        return Point((left + right) / 2.0, (top + bottom) / 2.0)
    }

    private fun measureTransformMotion(transform: TrackingTransform, pivot: Point): Double {
        val mappedPivot = transformPoint(pivot.x, pivot.y, transform)
        if (!mappedPivot.x.isFinite() || !mappedPivot.y.isFinite()) {
            return Double.POSITIVE_INFINITY
        }
        return hypot(mappedPivot.x - pivot.x, mappedPivot.y - pivot.y)
    }

    private fun sanitizeScale(scale: Double): Double {
        if (!scale.isFinite() || scale <= 0.0) return 1.0
        val clamped = scale.coerceIn(minTrackScale, maxTrackScale)
        if (abs(clamped - 1.0) < minScaleDelta) return 1.0
        return clamped
    }

    private fun sanitizeRotation(rotation: Double): Double {
        if (!rotation.isFinite()) return 0.0
        val maxRotation = Math.toRadians(maxTrackRotationDegrees)
        return rotation.coerceIn(-maxRotation, maxRotation)
    }

    private fun computeTrackingMotionStats(
        prevPoints: List<Point>,
        nextPoints: List<Point>
    ): TrackingMotionStats {
        val prevCentroid = Point(
            prevPoints.sumOf { it.x } / prevPoints.size,
            prevPoints.sumOf { it.y } / prevPoints.size
        )
        val nextCentroid = Point(
            nextPoints.sumOf { it.x } / nextPoints.size,
            nextPoints.sumOf { it.y } / nextPoints.size
        )

        val scaleSamples = ArrayList<Double>(prevPoints.size)
        for (i in prevPoints.indices) {
            val prevRadius = hypot(
                prevPoints[i].x - prevCentroid.x,
                prevPoints[i].y - prevCentroid.y
            )
            val nextRadius = hypot(
                nextPoints[i].x - nextCentroid.x,
                nextPoints[i].y - nextCentroid.y
            )
            if (prevRadius < minTrackingScaleRadiusPx) continue
            if (!nextRadius.isFinite() || nextRadius <= 0.0) continue
            scaleSamples.add(nextRadius / prevRadius)
        }

        return TrackingMotionStats(
            prevCentroid = prevCentroid,
            nextCentroid = nextCentroid,
            scaleHint = median(scaleSamples)
        )
    }

    private fun median(values: List<Double>): Double? {
        if (values.isEmpty()) return null
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle]
        }
    }

    private fun Mat.readDouble(row: Int, col: Int): Double? {
        return get(row, col)?.getOrNull(0)
    }

    private fun composeTransforms(
        step: TrackingTransform,
        accumulated: TrackingTransform
    ): TrackingTransform {
        return TrackingTransform(
            m00 = step.m00 * accumulated.m00 + step.m01 * accumulated.m10,
            m01 = step.m00 * accumulated.m01 + step.m01 * accumulated.m11,
            m02 = step.m00 * accumulated.m02 + step.m01 * accumulated.m12 + step.m02,
            m10 = step.m10 * accumulated.m00 + step.m11 * accumulated.m10,
            m11 = step.m10 * accumulated.m01 + step.m11 * accumulated.m11,
            m12 = step.m10 * accumulated.m02 + step.m11 * accumulated.m12 + step.m12
        )
    }

    private fun applyTrackingTransform(
        results: List<OcrService.DetectionResult>,
        transform: TrackingTransform,
        maxW: Int,
        maxH: Int
    ): List<OcrService.DetectionResult> {
        return results.map { result ->
            val shiftedCorners = shiftPoints(result.cornerPoints, transform, maxW, maxH)
            val shiftedBox = shiftedCorners?.let { boundsFromPoints(it, maxW, maxH) }
                ?: result.boundingBox?.let { shiftRect(it, transform, maxW, maxH) }

            val shiftedBlockBox = (result.blockBoundingBox ?: result.boundingBox)?.let {
                shiftRectKeepingSize(it, transform, maxW, maxH)
            }
            val shiftedBlockCorners = shiftedBlockBox?.let { rectToPoints(it) }

            val shiftedTextRegions = result.textRegions.mapNotNull { region ->
                val shiftedRegionCorners = shiftPoints(region.cornerPoints, transform, maxW, maxH)
                val shiftedRegionBox = shiftedRegionCorners?.let { boundsFromPoints(it, maxW, maxH) }
                    ?: region.boundingBox?.let { shiftRect(it, transform, maxW, maxH) }

                if (shiftedRegionBox == null && shiftedRegionCorners == null) {
                    null
                } else {
                    region.copy(
                        boundingBox = shiftedRegionBox,
                        cornerPoints = shiftedRegionCorners
                    )
                }
            }

            result.copy(
                boundingBox = shiftedBox,
                cornerPoints = shiftedCorners,
                blockBoundingBox = shiftedBlockBox,
                blockCornerPoints = shiftedBlockCorners,
                textRegions = shiftedTextRegions
            )
        }
    }

    private fun shiftRectKeepingSize(
        rect: android.graphics.Rect,
        transform: TrackingTransform,
        maxW: Int,
        maxH: Int
    ): android.graphics.Rect? {
        val cx = (rect.left + rect.right) * 0.5
        val cy = (rect.top + rect.bottom) * 0.5
        val transformedCenter = transformPoint(cx, cy, transform)
        val halfWidth = rect.width() / 2.0
        val halfHeight = rect.height() / 2.0

        val shiftedLeft = floor(transformedCenter.x - halfWidth).toInt()
        val shiftedTop = floor(transformedCenter.y - halfHeight).toInt()
        val shiftedRight = ceil(transformedCenter.x + halfWidth).toInt()
        val shiftedBottom = ceil(transformedCenter.y + halfHeight).toInt()

        if (!TrackingFrameBounds.containsRect(
                left = shiftedLeft,
                top = shiftedTop,
                right = shiftedRight,
                bottom = shiftedBottom,
                maxWidth = maxW,
                maxHeight = maxH
            )
        ) {
            return null
        }

        return android.graphics.Rect(shiftedLeft, shiftedTop, shiftedRight, shiftedBottom)
    }

    private fun shiftRect(
        rect: android.graphics.Rect,
        transform: TrackingTransform,
        maxW: Int,
        maxH: Int
    ): android.graphics.Rect? {
        val corners = arrayOf(
            transformPoint(rect.left.toDouble(), rect.top.toDouble(), transform),
            transformPoint(rect.right.toDouble(), rect.top.toDouble(), transform),
            transformPoint(rect.left.toDouble(), rect.bottom.toDouble(), transform),
            transformPoint(rect.right.toDouble(), rect.bottom.toDouble(), transform)
        )
        val shiftedLeft = floor(corners.minOf { it.x }).toInt()
        val shiftedTop = floor(corners.minOf { it.y }).toInt()
        val shiftedRight = ceil(corners.maxOf { it.x }).toInt()
        val shiftedBottom = ceil(corners.maxOf { it.y }).toInt()

        if (!TrackingFrameBounds.containsRect(
                left = shiftedLeft,
                top = shiftedTop,
                right = shiftedRight,
                bottom = shiftedBottom,
                maxWidth = maxW,
                maxHeight = maxH
            )
        ) {
            return null
        }

        return android.graphics.Rect(shiftedLeft, shiftedTop, shiftedRight, shiftedBottom)
    }

    private fun shiftPoints(
        points: Array<android.graphics.Point>?,
        transform: TrackingTransform,
        maxW: Int,
        maxH: Int
    ): Array<android.graphics.Point>? {
        if (points == null) return null
        return points.map { p ->
            val transformed = transformPoint(p.x.toDouble(), p.y.toDouble(), transform)
            if (!TrackingFrameBounds.containsPoint(
                    x = transformed.x,
                    y = transformed.y,
                    maxWidth = maxW,
                    maxHeight = maxH
                )
            ) {
                return null
            }
            val x = transformed.x.roundToInt()
            val y = transformed.y.roundToInt()
            android.graphics.Point(x, y)
        }
            .toTypedArray()
    }

    private fun boundsFromPoints(
        points: Array<android.graphics.Point>?,
        maxW: Int,
        maxH: Int
    ): android.graphics.Rect? {
        if (points == null || points.isEmpty()) return null

        val rawLeft = points.minOf { it.x }
        val rawTop = points.minOf { it.y }
        val rawRight = points.maxOf { it.x }
        val rawBottom = points.maxOf { it.y }

        if (!TrackingFrameBounds.containsRect(
                left = rawLeft,
                top = rawTop,
                right = rawRight,
                bottom = rawBottom,
                maxWidth = maxW,
                maxHeight = maxH
            )
        ) {
            return null
        }

        return android.graphics.Rect(rawLeft, rawTop, rawRight, rawBottom)
    }

    private fun rectToPoints(rect: android.graphics.Rect): Array<android.graphics.Point> {
        return arrayOf(
            android.graphics.Point(rect.left, rect.top),
            android.graphics.Point(rect.right, rect.top),
            android.graphics.Point(rect.right, rect.bottom),
            android.graphics.Point(rect.left, rect.bottom)
        )
    }

    private fun transformPoint(x: Double, y: Double, transform: TrackingTransform): Point {
        return Point(
            transform.m00 * x + transform.m01 * y + transform.m02,
            transform.m10 * x + transform.m11 * y + transform.m12
        )
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

    private fun clearTrackingState(clearDisplayedOverlay: Boolean = true) {
        synchronized(trackingLock) {
            trackedResults.clear()
            trackedBaseResults.clear()
            trackedTranslations.clear()
            trackingAnchorGray?.release()
            trackingAnchorGray = null
            trackingAccumulatedTransform = TrackingTransform()
            trackingMissCount = 0
        }
        translationCache.clear()
        FrameOcrRepository.clearOcrSourceFrame()
        needsFreshInpaint = true
        synchronized(metricsLock) {
            activeMetricsSession = null
        }
        replacer.clearStableState()
        if (clearDisplayedOverlay) {
            FrameOcrRepository.clearFrame()
            overlayEnabled = false
            setGeneratedTextVisible(false)
            runOnUiThread {
                binding.processedOverlay.setImageBitmap(null)
                binding.processedOverlay.visibility = View.INVISIBLE
            }
        }
    }

    private fun requestRedetectionAfterTrackingLoss() {
        lastProcessTime = 0L
        setNoTextDetected(false)
        if (cameraMotionStable) {
            redetectAfterMotionStops = true
            scheduleRedetectionAfterSettle()
        } else {
            cancelPendingRedetection()
            redetectAfterMotionStops = true
            setDetectingActive(false)
        }
    }

    private fun clearDetectedBlocks() {
        resetDetectionSession()
        Log.d(TAG, "Reset detection session due to language change")
    }

    private fun resetDetectionSession() {
        cancelPendingRedetection()
        clearTrackingState()
        ocrService.resetSessionState()
        cameraMotionStable = false
        redetectAfterMotionStops = true
        lastProcessTime = 0L
        resetProcessingWork(keepDetecting = !freezeFrameMode && isOpenCvReady)
    }

    private fun processOnInterval(event: OcrService.OcrDetectionEvent) {
        if (freezeFrameMode) return

        val rawResults = event.results
        val now = System.currentTimeMillis()
        if (now - lastProcessTime <= processInterval) return
        lastProcessTime = now
        synchronized(trackingLock) {
            if (trackedResults.isNotEmpty()) {
                return
            }
        }
        val results = rawResults.map { result ->
            result.copy(stableId = result.stableId ?: "det-${nextStableDetectionId++}")
        }
        val baseFrame = FrameOcrRepository.snapshotOcrSourceFrame(event.requestId)
            ?: FrameOcrRepository.snapshotLatestCameraFrame()
            ?: return
        FrameOcrRepository.discardOcrSourceFrame(event.requestId)
        if (baseFrame.empty() || baseFrame.cols() <= 0 || baseFrame.rows() <= 0) {
            baseFrame.release()
            return
        }

        val translatedTexts = MutableList(results.size) { "" }
        val anchorGray = toGray(baseFrame)
        baseFrame.release()
        val metricsSession = startMetricsSession(event)

        synchronized(trackingLock) {
            trackedBaseResults = results.map { it.copy() }.toMutableList()
            trackedResults = results.map { it.copy() }.toMutableList()
            trackedTranslations = translatedTexts
            trackingAnchorGray?.release()
            trackingAnchorGray = anchorGray
            trackingAccumulatedTransform = TrackingTransform()
            trackingMissCount = 0
        }
        needsFreshInpaint = true

        val selectedSourceCode = getLanguageCode(sourceLanguageDisplay)
        val targetLanguage = targetLanguageCode
        val pendingBatches = HashMap<String, MutableList<Pair<Int, String>>>()

        results.forEachIndexed { index, result ->
            val text = result.text

            translationCache[text]?.let { cached ->
                applyTranslationResult(index, cached, translatedTexts)
                return@forEachIndexed
            }

            val sourceLanguage = LanguageUiLogic.resolveSourceLanguage(selectedSourceCode, result.language)
            if (LanguageUiLogic.shouldBypassTranslation(sourceLanguage, targetLanguage, text)) {
                applyTranslationResult(index, text, translatedTexts)
                return@forEachIndexed
            }

            pendingBatches.getOrPut(sourceLanguage) { mutableListOf() }
                .add(index to text)
        }

        pendingBatches.forEach { (sourceLanguage, batch) ->
            val batchTexts = batch.map { it.second }
            markTranslationBatchStarted(metricsSession)
            markTranslationStarted()
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
                                applyTranslationResult(index, translated, translatedTexts)
                            }
                        } finally {
                            markTranslationBatchFinished(metricsSession)
                            markTranslationFinished()
                        }
                    },
                    onError = { error ->
                        Log.e(TAG, "Batch translation failed for $sourceLanguage->$targetLanguage", error)
                        markTranslationBatchFinished(metricsSession)
                        markTranslationFinished()
                    }
                )
            } catch (e: Exception) {
                Log.e(TAG, "Batch translation setup failed for $sourceLanguage->$targetLanguage", e)
                markTranslationBatchFinished(metricsSession)
                markTranslationFinished()
            }
        }

        if (pendingBatches.isEmpty()) {
            finalizeTranslationMetrics(metricsSession, event.ocrCompletedAtMs)
        }
    }

    private fun applyTranslationResult(
        index: Int,
        text: String,
        translatedTexts: MutableList<String>
    ) {
        if (index >= translatedTexts.size) return
        translatedTexts[index] = text
        synchronized(trackingLock) {
            if (trackedTranslations === translatedTexts && index < trackedTranslations.size) {
                trackedTranslations[index] = text
            }
        }
    }


    private fun mapDetectedLanguage(detected: String): String {
        return LanguageUiLogic.mapDetectedLanguage(detected)
    }

    private fun setupSpinners() {
        languageCodeByDisplay.clear()
        sourceLanguageOptions.clear()
        targetLanguageOptions.clear()
        downloadingLanguageCodes.clear()
        downloadedLanguageCodes.clear()

        val usedLabels = HashSet<String>()
        val supportedLanguages = TranslateLanguage.getAllLanguages()
            .map { code ->
                LanguageOption(
                    label = buildLanguageLabel(code, usedLabels),
                    code = code
                )
            }
            .sortedBy { it.label }

        sourceLanguageOptions.add(LanguageOption("Detected Language", "auto"))
        languageCodeByDisplay["Detected Language"] = "auto"
        supportedLanguages.forEach { option ->
            sourceLanguageOptions.add(option)
            targetLanguageOptions.add(option)
            languageCodeByDisplay[option.label] = option.code
        }

        sourceSpinnerAdapter = SourceLanguageAdapter()
        binding.sourceLanguage.adapter = sourceSpinnerAdapter

        targetSpinnerAdapter = TargetLanguageAdapter()
        binding.targetLanguage.adapter = targetSpinnerAdapter

        binding.targetLanguage.onItemSelectedListener = this
        binding.sourceLanguage.onItemSelectedListener = this
        binding.targetLanguage.setOnTouchListener { _, _ ->
            refreshDownloadedLanguageStatus()
            false
        }
        binding.sourceLanguage.setOnTouchListener { _, _ ->
            refreshDownloadedLanguageStatus()
            false
        }

        val targetEnglishIndex = targetLanguageOptions.indexOfFirst { it.code == "en" }.coerceAtLeast(0)
        val sourceDetectedIndex = sourceLanguageOptions.indexOfFirst { it.code == "auto" }.coerceAtLeast(0)
        binding.targetLanguage.setSelection(targetEnglishIndex)
        binding.sourceLanguage.setSelection(sourceDetectedIndex)
        updateOcrSourceLanguageHint()

        refreshDownloadedLanguageStatus()
    }

    override fun onItemSelected(parent: AdapterView<*>, view: android.view.View?, pos: Int, id: Long) {
        when (parent.id) {
            R.id.targetLanguage -> {
                val selectedOption = targetLanguageOptions.getOrNull(pos) ?: return
                val newLang = selectedOption.code

                if (newLang != targetLanguageCode) {

                    targetLanguageCode = newLang

                    clearDetectedBlocks()

                    Log.d(TAG, "Target changed → cleared blocks")
                }
            }
            R.id.sourceLanguage -> {
                val selected = sourceLanguageOptions.getOrNull(pos)?.label ?: return
                if (selected != sourceLanguageDisplay) {

                    sourceLanguageDisplay = selected
                    updateOcrSourceLanguageHint()

                    clearDetectedBlocks()

                    Log.d(TAG, "Source changed → cleared blocks")
                }
            }
        }
    }

    override fun onNothingSelected(parent: AdapterView<*>) {
        when (parent.id) {
            R.id.targetLanguage -> targetLanguageCode = "en"
            R.id.sourceLanguage -> {
                sourceLanguageDisplay = "Detected Language"
                updateOcrSourceLanguageHint()
            }
        }
    }

    private fun getLanguageCode(displayName: String): String {
        return languageCodeByDisplay[displayName] ?: "en"
    }

    private fun updateOcrSourceLanguageHint() {
        val sourceCode = getLanguageCode(sourceLanguageDisplay)
        ocrService.setSourceLanguageHint(sourceCode)
    }

    private fun buildLanguageLabel(code: String, usedLabels: MutableSet<String>): String {
        return LanguageUiLogic.buildLanguageLabel(code, usedLabels)
    }

    private fun isLanguageModelDownloaded(languageCode: String): Boolean {
        return downloadedLanguageCodes.contains(languageCode)
    }

    private fun refreshDownloadedLanguageStatus() {
        translationService.getDownloadedLanguageModels(
            onSuccess = { codes ->
                runOnUiThread {
                    downloadedLanguageCodes.clear()
                    downloadedLanguageCodes.addAll(codes)
                    notifyLanguageAdaptersChanged()
                }
            },
            onError = { error ->
                Log.e(TAG, "Failed loading downloaded language models", error)
            }
        )
    }

    private fun requestLanguageModelDownload(option: LanguageOption) {
        val code = option.code
        if (isLanguageModelDownloaded(code) || downloadingLanguageCodes.contains(code)) return

        downloadingLanguageCodes.add(code)
        notifyLanguageAdaptersChanged()

        translationService.downloadLanguageModel(
            languageCode = code,
            onSuccess = {
                runOnUiThread {
                    downloadingLanguageCodes.remove(code)
                    downloadedLanguageCodes.add(code)
                    notifyLanguageAdaptersChanged()
                    showToast("${option.label} language pack downloaded")
                }
            },
            onError = { error ->
                runOnUiThread {
                    downloadingLanguageCodes.remove(code)
                    notifyLanguageAdaptersChanged()
                    showToast("Failed to download ${option.label}")
                }
                Log.e(TAG, "Model download failed for ${option.label}", error)
            }
        )
    }

    private fun notifyLanguageAdaptersChanged() {
        if (::sourceSpinnerAdapter.isInitialized) {
            sourceSpinnerAdapter.notifyDataSetChanged()
        }
        if (::targetSpinnerAdapter.isInitialized) {
            targetSpinnerAdapter.notifyDataSetChanged()
        }
    }

    private fun bindLanguageDropdownRow(view: View, option: LanguageOption, allowDownload: Boolean) {
        val label = view.findViewById<TextView>(R.id.languageLabel)
        val downloadIcon = view.findViewById<ImageButton>(R.id.downloadIcon)
        label.text = option.label

        if (!allowDownload) {
            stopDownloadIconAnimation(downloadIcon)
            downloadIcon.visibility = View.GONE
            downloadIcon.setOnClickListener(null)
            return
        }

        val presentation = LanguageUiLogic.computeDownloadPresentation(
            isDownloaded = isLanguageModelDownloaded(option.code),
            isDownloading = downloadingLanguageCodes.contains(option.code)
        )

        if (presentation.iconState == LanguageUiLogic.DownloadIconState.HIDDEN) {
            stopDownloadIconAnimation(downloadIcon)
            downloadIcon.visibility = View.GONE
            downloadIcon.setOnClickListener(null)
            return
        }

        downloadIcon.visibility = View.VISIBLE
        downloadIcon.setImageResource(
            if (presentation.iconState == LanguageUiLogic.DownloadIconState.DOWNLOADING) {
                R.drawable.ic_downloading_20
            } else {
                R.drawable.ic_download_20
            }
        )

        if (presentation.iconState == LanguageUiLogic.DownloadIconState.DOWNLOADING) {
            startDownloadIconAnimation(downloadIcon)
        } else {
            stopDownloadIconAnimation(downloadIcon)
        }

        downloadIcon.isEnabled = presentation.enabled
        downloadIcon.contentDescription = getString(
            if (presentation.iconState == LanguageUiLogic.DownloadIconState.DOWNLOADING) {
                R.string.downloading_language_pack
            } else {
                R.string.download_language_pack
            }
        )
        downloadIcon.setOnClickListener {
            requestLanguageModelDownload(option)
        }
    }

    private inner class SourceLanguageAdapter : ArrayAdapter<LanguageOption>(
        this@MainActivity,
        R.layout.item_spinner_selected,
        sourceLanguageOptions
    ) {
        override fun getCount(): Int = sourceLanguageOptions.size

        override fun getItem(position: Int): LanguageOption = sourceLanguageOptions[position]

        override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
            val view = convertView ?: layoutInflater.inflate(R.layout.item_spinner_selected, parent, false)
            val text = view.findViewById<TextView>(R.id.spinnerText)
            text.text = getItem(position).label
            return view
        }

        override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View {
            val view = convertView
                ?: layoutInflater.inflate(R.layout.item_target_language_dropdown, parent, false)

            val option = getItem(position)
            bindLanguageDropdownRow(
                view = view,
                option = option,
                allowDownload = option.code != "auto"
            )
            return view
        }
    }

    private inner class TargetLanguageAdapter : ArrayAdapter<LanguageOption>(
        this@MainActivity,
        R.layout.item_spinner_selected,
        targetLanguageOptions
    ) {
        override fun getCount(): Int = targetLanguageOptions.size

        override fun getItem(position: Int): LanguageOption = targetLanguageOptions[position]

        override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
            val view = convertView ?: layoutInflater.inflate(R.layout.item_spinner_selected, parent, false)
            val text = view.findViewById<TextView>(R.id.spinnerText)
            text.text = getItem(position).label
            return view
        }

        override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View {
            val view = convertView
                ?: layoutInflater.inflate(R.layout.item_target_language_dropdown, parent, false)

            val option = getItem(position)
            bindLanguageDropdownRow(view = view, option = option, allowDownload = true)

            return view
        }
    }

    private fun startDownloadIconAnimation(icon: ImageButton) {
        val existing = icon.tag as? Animator
        if (existing?.isRunning == true) return
        existing?.cancel()

        val spin = ObjectAnimator.ofFloat(icon, View.ROTATION, 0f, 360f).apply {
            duration = 1050L
            repeatCount = ValueAnimator.INFINITE
            interpolator = LinearInterpolator()
        }
        val pulse = ObjectAnimator.ofPropertyValuesHolder(
            icon,
            PropertyValuesHolder.ofFloat(View.SCALE_X, 1f, 0.9f, 1f),
            PropertyValuesHolder.ofFloat(View.SCALE_Y, 1f, 0.9f, 1f),
            PropertyValuesHolder.ofFloat(View.ALPHA, 1f, 0.7f, 1f)
        ).apply {
            duration = 820L
            repeatCount = ValueAnimator.INFINITE
            interpolator = FastOutSlowInInterpolator()
        }

        val animatorSet = AnimatorSet().apply {
            playTogether(spin, pulse)
        }
        icon.tag = animatorSet
        animatorSet.start()
    }

    private fun stopDownloadIconAnimation(icon: ImageButton) {
        (icon.tag as? Animator)?.cancel()
        icon.tag = null
        icon.rotation = 0f
        icon.scaleX = 1f
        icon.scaleY = 1f
        icon.alpha = 1f
    }

    private fun setupCamera() {
        cameraController = CameraController(this, this, cameraExecutor, binding)
    }

    private fun startCameraWithOcr() {
        resetRuntimeMetrics()
        redetectAfterMotionStops = true
        if (!isOpenCvReady) {
            cameraController.startCamera()
            setDetectingActive(false)
            if (!hasShownOpenCvUnavailableWarning) {
                hasShownOpenCvUnavailableWarning = true
                Toast.makeText(
                    this,
                    "OpenCV unavailable: camera preview only, OCR disabled",
                    Toast.LENGTH_LONG
                ).show()
            }
            Log.e(TAG, "Skipping OCR analyzer because OpenCV is not ready")
            return
        }
        cameraController.startCamera(ocrService)
        setDetectingActive(true)
    }

    private fun setupClickListeners() {
        binding.imageCaptureButton.setOnClickListener {
            if (!freezeFrameMode) {
                enterFreezeFrameMode()
            }
        }
        binding.savePhotoButton.setOnClickListener {
            saveFrozenFrameToGallery()
        }
        binding.returnToCameraButton.setOnClickListener {
            exitFreezeFrameMode()
        }
    }

    private fun setupBackPressHandler() {
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (freezeFrameMode) {
                    exitFreezeFrameMode()
                    return
                }

                isEnabled = false
                onBackPressedDispatcher.onBackPressed()
                isEnabled = true
            }
        })
    }

    private fun setupFreezeFrameZoom() {
        scaleGestureDetector = ScaleGestureDetector(
            this,
            object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
                override fun onScale(detector: ScaleGestureDetector): Boolean {
                    if (!freezeFrameMode) return false
                    applyFreezeScale(detector.scaleFactor, detector.focusX, detector.focusY)
                    return true
                }
            }
        )

        freezeGestureDetector = GestureDetector(
            this,
            object : GestureDetector.SimpleOnGestureListener() {
                override fun onDoubleTap(e: MotionEvent): Boolean {
                    if (!freezeFrameMode) return false
                    val target = if (freezeCurrentScale <= freezeMinScale * 1.2f) {
                        (freezeMinScale * 2.0f).coerceAtMost(freezeMaxScale)
                    } else {
                        freezeMinScale
                    }
                    scaleFreezeTo(target, e.x, e.y)
                    return true
                }
            }
        )

        binding.processedOverlay.setOnTouchListener { _, event ->
            if (!freezeFrameMode) return@setOnTouchListener false

            freezeGestureDetector.onTouchEvent(event)
            scaleGestureDetector.onTouchEvent(event)

            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    lastTouchX = event.x
                    lastTouchY = event.y
                    isFreezeDragging = true
                }

                MotionEvent.ACTION_MOVE -> {
                    if (!scaleGestureDetector.isInProgress && isFreezeDragging && event.pointerCount == 1) {
                        val dx = event.x - lastTouchX
                        val dy = event.y - lastTouchY
                        freezeImageMatrix.postTranslate(dx, dy)
                        constrainFreezeMatrix()
                        binding.processedOverlay.imageMatrix = freezeImageMatrix
                        lastTouchX = event.x
                        lastTouchY = event.y
                    }
                }

                MotionEvent.ACTION_POINTER_DOWN -> {
                    isFreezeDragging = false
                }

                MotionEvent.ACTION_UP,
                MotionEvent.ACTION_CANCEL -> {
                    isFreezeDragging = false
                }
            }

            true
        }
    }

    private fun enterFreezeFrameMode() {
        if (freezeFrameMode) return

        val snapshot = snapshotCurrentFrame() ?: run {
            showToast("No frame available yet")
            return
        }

        lifecycleScope.launch(Dispatchers.Default) {
            val frozenBitmap = matToDisplayBitmap(snapshot)
            snapshot.release()

            withContext(Dispatchers.Main) {
                if (frozenBitmap == null) {
                    showToast("Failed to capture frame")
                    return@withContext
                }

                freezeFrameMode = true
                cameraController.stopCamera()
                resetProcessingWork(keepDetecting = false)
                frozenFrameBitmap?.recycle()
                frozenFrameBitmap = frozenBitmap
                updateFreezeFrameActionButtons()

                binding.processedOverlay.scaleType = ImageView.ScaleType.MATRIX
                binding.processedOverlay.setImageBitmap(frozenBitmap)
                binding.processedOverlay.post { resetFreezeImageMatrix() }
                binding.processedOverlay.visibility = View.VISIBLE
                overlayEnabled = true
            }
        }
    }

    private fun exitFreezeFrameMode() {
        if (!freezeFrameMode) return

        freezeFrameMode = false
        resetDetectionSession()
        frozenFrameBitmap?.recycle()
        frozenFrameBitmap = null
        updateFreezeFrameActionButtons()
        resetFreezeState()
        binding.processedOverlay.scaleType = ImageView.ScaleType.FIT_CENTER
        startCameraWithOcr()
    }

    private fun updateFreezeFrameActionButtons() {
        val freezeVisible = if (freezeFrameMode) View.VISIBLE else View.GONE
        val captureVisible = if (freezeFrameMode) View.GONE else View.VISIBLE

        binding.imageCaptureButton.visibility = captureVisible
        binding.imageCaptureButton.isEnabled = !freezeFrameMode
        binding.savePhotoButton.visibility = freezeVisible
        binding.savePhotoButton.isEnabled = freezeFrameMode
        binding.returnToCameraButton.visibility = freezeVisible
        binding.returnToCameraButton.isEnabled = freezeFrameMode

        if (freezeFrameMode) {
            binding.savePhotoButton.bringToFront()
            binding.returnToCameraButton.bringToFront()
            binding.controlPanel.invalidate()
        }
    }

    private fun saveFrozenFrameToGallery() {
        val sourceBitmap = frozenFrameBitmap ?: run {
            showToast(getString(R.string.photo_unavailable))
            return
        }
        val bitmapToSave = Bitmap.createBitmap(sourceBitmap)
        binding.savePhotoButton.isEnabled = false

        lifecycleScope.launch(Dispatchers.IO) {
            val resolver = applicationContext.contentResolver
            val displayName = "TextLens_${System.currentTimeMillis()}.jpg"
            var savedUri = resolver.insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, displayName)
                    put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/TextLens")
                        put(MediaStore.Images.Media.IS_PENDING, 1)
                    }
                }
            )

            val saveSucceeded = try {
                val targetUri = savedUri ?: error("Unable to create gallery entry")
                val outputStream = resolver.openOutputStream(targetUri)
                    ?: error("Unable to open output stream")
                outputStream.use { stream ->
                    if (!bitmapToSave.compress(Bitmap.CompressFormat.JPEG, 95, stream)) {
                        error("Bitmap compression failed")
                    }
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    resolver.update(
                        targetUri,
                        ContentValues().apply {
                            put(MediaStore.Images.Media.IS_PENDING, 0)
                        },
                        null,
                        null
                    )
                }
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to save frozen frame", e)
                savedUri?.let { resolver.delete(it, null, null) }
                savedUri = null
                false
            } finally {
                bitmapToSave.recycle()
            }

            withContext(Dispatchers.Main) {
                binding.savePhotoButton.isEnabled = freezeFrameMode
                showToast(
                    if (saveSucceeded) {
                        getString(R.string.photo_saved)
                    } else {
                        getString(R.string.photo_save_failed)
                    }
                )
            }
        }
    }

    private fun resetFreezeImageMatrix() {
        val drawable = binding.processedOverlay.drawable ?: return
        val viewWidth = binding.processedOverlay.width.toFloat()
        val viewHeight = binding.processedOverlay.height.toFloat()
        if (viewWidth <= 0f || viewHeight <= 0f) return

        val imageWidth = drawable.intrinsicWidth.toFloat()
        val imageHeight = drawable.intrinsicHeight.toFloat()
        if (imageWidth <= 0f || imageHeight <= 0f) return

        freezeImageWidth = imageWidth
        freezeImageHeight = imageHeight
        freezeMinScale = minOf(viewWidth / imageWidth, viewHeight / imageHeight)
        freezeMaxScale = freezeMinScale * 5f
        freezeCurrentScale = freezeMinScale

        val offsetX = (viewWidth - imageWidth * freezeMinScale) * 0.5f
        val offsetY = (viewHeight - imageHeight * freezeMinScale) * 0.5f

        freezeImageMatrix.reset()
        freezeImageMatrix.postScale(freezeMinScale, freezeMinScale)
        freezeImageMatrix.postTranslate(offsetX, offsetY)
        binding.processedOverlay.imageMatrix = freezeImageMatrix
    }

    private fun applyFreezeScale(scaleFactor: Float, focusX: Float, focusY: Float) {
        if (scaleFactor <= 0f) return
        val targetScale = (freezeCurrentScale * scaleFactor).coerceIn(freezeMinScale, freezeMaxScale)
        scaleFreezeTo(targetScale, focusX, focusY)
    }

    private fun scaleFreezeTo(targetScale: Float, focusX: Float, focusY: Float) {
        val appliedFactor = targetScale / freezeCurrentScale
        if (appliedFactor == 1f) return
        freezeImageMatrix.postScale(appliedFactor, appliedFactor, focusX, focusY)
        freezeCurrentScale = targetScale
        constrainFreezeMatrix()
        binding.processedOverlay.imageMatrix = freezeImageMatrix
    }

    private fun constrainFreezeMatrix() {
        if (freezeImageWidth <= 0f || freezeImageHeight <= 0f) return
        val viewWidth = binding.processedOverlay.width.toFloat()
        val viewHeight = binding.processedOverlay.height.toFloat()
        if (viewWidth <= 0f || viewHeight <= 0f) return

        val mapped = RectF(0f, 0f, freezeImageWidth, freezeImageHeight)
        freezeImageMatrix.mapRect(mapped)

        val dx = when {
            mapped.width() <= viewWidth -> viewWidth * 0.5f - mapped.centerX()
            mapped.left > 0f -> -mapped.left
            mapped.right < viewWidth -> viewWidth - mapped.right
            else -> 0f
        }

        val dy = when {
            mapped.height() <= viewHeight -> viewHeight * 0.5f - mapped.centerY()
            mapped.top > 0f -> -mapped.top
            mapped.bottom < viewHeight -> viewHeight - mapped.bottom
            else -> 0f
        }

        if (dx != 0f || dy != 0f) {
            freezeImageMatrix.postTranslate(dx, dy)
        }
    }

    private fun resetFreezeState() {
        freezeImageMatrix.reset()
        freezeCurrentScale = 1f
        freezeMinScale = 1f
        freezeMaxScale = 1f
        freezeImageWidth = 0f
        freezeImageHeight = 0f
        isFreezeDragging = false
    }

    private fun snapshotCurrentFrame(): Mat? {
        FrameOcrRepository.snapshotCurrentFrame()?.let { processed ->
            return processed
        }

        FrameOcrRepository.snapshotLatestCameraFrame()?.let { raw ->
            return raw
        }

        return null
    }

    private fun matToDisplayBitmap(source: Mat): Bitmap? {
        if (source.empty() || source.cols() <= 0 || source.rows() <= 0) return null

        val rgba = Mat()
        return try {
            when (source.channels()) {
                4 -> source.copyTo(rgba)
                3 -> Imgproc.cvtColor(source, rgba, Imgproc.COLOR_BGR2RGBA)
                1 -> Imgproc.cvtColor(source, rgba, Imgproc.COLOR_GRAY2RGBA)
                else -> return null
            }

            Bitmap.createBitmap(source.cols(), source.rows(), Bitmap.Config.ARGB_8888).also { bmp ->
                org.opencv.android.Utils.matToBitmap(rgba, bmp)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to convert Mat to Bitmap", e)
            null
        } finally {
            rgba.release()
        }
    }

    private fun showToast(message: String) {
        runOnUiThread {
            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraController.stopCamera()
        ocrService.cleanup()
        translationService.cleanup()
        cancelPendingRedetection()
        trackingJob?.cancel()
        synchronized(trackingLock) {
            trackingAnchorGray?.release()
            trackingAnchorGray = null
        }
        frozenFrameBitmap?.recycle()
        frozenFrameBitmap = null
        cameraExecutor.shutdown()
    }

    companion object {
        const val TAG = "MainActivity"
        private val REQUIRED_PERMISSIONS = mutableListOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        ).apply {
            if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            }
        }.toTypedArray()
    }

    private val activityResultLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            val granted = permissions.entries.all { it.value }
            if (granted) {
                startCameraWithOcr()
            } else {
                Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show()
            }
        }
}
