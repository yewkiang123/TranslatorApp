package com.example.translatorapp

import android.Manifest
import android.animation.Animator
import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.animation.PropertyValuesHolder
import android.animation.ValueAnimator
import android.content.res.ColorStateList
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.Build
import android.os.Bundle
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
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import kotlinx.coroutines.flow.conflate
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import kotlin.math.abs
import kotlin.math.hypot


class MainActivity : AppCompatActivity(), AdapterView.OnItemSelectedListener {
    private enum class ProcessingIndicatorState {
        IDLE,
        DETECTING,
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
        val dx: Double = 0.0,
        val dy: Double = 0.0,
        val scale: Double = 1.0
    )

    private data class TrackingSnapshot(
        val results: List<OcrService.DetectionResult>,
        val translations: List<String>,
        val anchorGray: Mat
    )

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
    @Volatile private var overlayEnabled = false
    @Volatile private var freezeFrameMode = false
    private var frozenFrameBitmap: Bitmap? = null
    private val trackingLock = Any()
    private var trackedResults: MutableList<OcrService.DetectionResult> = mutableListOf()
    private var trackedTranslations: MutableList<String> = mutableListOf()
    private var trackingAnchorGray: Mat? = null
    private var lastTrackingTransform = TrackingTransform()
    private val trackIntervalMs = 120L
    private val panThresholdPx = 80.0
    private val minTrackShiftPx = 1.5
    private val maxTrackStepPx = 30.0
    private val minScaleDelta = 0.015
    private val minTrackScale = 0.90
    private val maxTrackScale = 1.10
    private val zoomResetThreshold = 0.22

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
    private var detectingActive = false
    private var translatingJobsInFlight = 0
    private var generatingJobsInFlight = 0
    private var generatedTextVisible = false
    private var lastProcessingIndicatorState: ProcessingIndicatorState? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize OpenCV FIRST before anything else
        isOpenCvReady = initializeOpenCV()

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        renderProcessingIndicator(force = true)

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
        try {
            // Try to load OpenCV
            if (!org.opencv.android.OpenCVLoader.initDebug()) {
                Log.e("OpenCV", "OpenCV initialization failed!")
                Toast.makeText(this, "OpenCV failed to load", Toast.LENGTH_SHORT).show()
                return false
            } else {
                Log.d("OpenCV", "OpenCV initialized successfully")
                return true
            }
        } catch (e: UnsatisfiedLinkError) {
            Log.e("OpenCV", "OpenCV library not found", e)
            Toast.makeText(this, "OpenCV library missing", Toast.LENGTH_LONG).show()
            return false
        }
    }

    private fun initServices() {
        ocrService = OcrService()
        ocrService.setSourceLanguageHint("auto")
        translationService = TranslationService()
    }

    private fun setupOcrListener() {
        ocrService.detections.onEach { results ->
            if (results.isNotEmpty()) {
                processOnInterval(results)
            }
        }.launchIn(lifecycleScope)

        ocrService.motionStable.onEach { stable ->
            if (!stable && !freezeFrameMode) {
                clearTrackingState()
            }
        }.launchIn(lifecycleScope)
    }

    private fun setDetectingActive(active: Boolean) {
        synchronized(processingStateLock) {
            detectingActive = active
        }
        renderProcessingIndicator()
    }

    private fun resetProcessingWork(keepDetecting: Boolean) {
        synchronized(processingStateLock) {
            translatingJobsInFlight = 0
            generatingJobsInFlight = 0
            generatedTextVisible = false
            detectingActive = keepDetecting
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
                .collectLatest { mat ->

                    if (mat.empty()) return@collectLatest
                    if (freezeFrameMode) return@collectLatest

                    // Clone immediately (Mat may change later)
                    val safeMat = mat.clone()

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
                    }
                }
        }
    }

    private fun startTrackingLoop() {
        trackingJob?.cancel()
        trackingJob = lifecycleScope.launch(Dispatchers.Default) {
            while (isActive) {
                if (freezeFrameMode) {
                    kotlinx.coroutines.delay(trackIntervalMs)
                    continue
                }
                val frame = FrameOcrRepository.latestCameraFrame.value?.clone()
                if (frame != null && !frame.empty()) {
                    trackAndRender(frame)
                } else {
                    frame?.release()
                }
                kotlinx.coroutines.delay(trackIntervalMs)
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
                results = trackedResults.map { it.copy() },
                translations = trackedTranslations.toList(),
                anchorGray = anchor
            )
        }

        if (snapshot == null) {
            hideOverlayOnly()
            frame.release()
            return
        }

        val currentGray = toGray(frame)
        val estimatedTransform = estimateTrackingTransform(snapshot.anchorGray, currentGray)
        currentGray.release()
        snapshot.anchorGray.release()

        val transform = synchronized(trackingLock) {
            if (estimatedTransform != null) {
                lastTrackingTransform = estimatedTransform
            }
            lastTrackingTransform
        }

        if (abs(transform.dx) + abs(transform.dy) > panThresholdPx ||
            abs(transform.scale - 1.0) > zoomResetThreshold) {
            clearTrackingState()
            frame.release()
            return
        }

        val shifted = snapshot.results.map { result ->
            val box = result.boundingBox ?: return@map result
            val shiftedBox = shiftRect(box, transform, frame.cols(), frame.rows())
            val shiftedCorners = if (shiftedBox != null) {
                shiftPoints(result.cornerPoints, transform, frame.cols(), frame.rows())
            } else {
                null
            }
            val shiftedBlockBox = result.blockBoundingBox?.let { block ->
                shiftRect(block, transform, frame.cols(), frame.rows())
            }
            val shiftedBlockCorners = if (shiftedBlockBox != null) {
                shiftPoints(result.blockCornerPoints, transform, frame.cols(), frame.rows())
            } else {
                null
            }
            result.copy(
                boundingBox = shiftedBox,
                cornerPoints = shiftedCorners,
                blockBoundingBox = shiftedBlockBox,
                blockCornerPoints = shiftedBlockCorners
            )
        }

        if (shifted.none { it.boundingBox != null }) {
            clearTrackingState()
            frame.release()
            return
        }

        if (snapshot.translations.none { it.isNotBlank() }) {
            hideOverlayOnly()
            frame.release()
            return
        }

        setGeneratingActive(true)
        try {
            val updated = replacer.replaceText(frame, shifted, snapshot.translations)
            try {
                FrameOcrRepository.updateFrame(updated)
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
            hideOverlayOnly()
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

    private fun estimateTrackingTransform(prev: Mat?, current: Mat): TrackingTransform? {
        if (prev == null || prev.empty()) return null

        val points = MatOfPoint()
        Imgproc.goodFeaturesToTrack(prev, points, 120, 0.01, 8.0)
        if (points.empty()) {
            points.release()
            return null
        }

        val prevPts = MatOfPoint2f(*points.toArray())
        val nextPts = MatOfPoint2f()
        val status = MatOfByte()
        val err = MatOfFloat()

        Video.calcOpticalFlowPyrLK(prev, current, prevPts, nextPts, status, err)

        val prevArr = prevPts.toArray()
        val nextArr = nextPts.toArray()
        val statusArr = status.toArray()

        prevPts.release()
        nextPts.release()
        status.release()
        err.release()
        points.release()

        val prevValid = ArrayList<org.opencv.core.Point>(statusArr.size)
        val nextValid = ArrayList<org.opencv.core.Point>(statusArr.size)

        for (i in statusArr.indices) {
            if (statusArr[i].toInt() != 1) continue
            val from = prevArr[i]
            val to = nextArr[i]
            if (!from.x.isFinite() || !from.y.isFinite() || !to.x.isFinite() || !to.y.isFinite()) continue
            prevValid.add(from)
            nextValid.add(to)
        }

        if (prevValid.size < 6) {
            return null
        }

        val dxSamples = ArrayList<Double>(prevValid.size)
        val dySamples = ArrayList<Double>(prevValid.size)
        for (i in prevValid.indices) {
            dxSamples.add(nextValid[i].x - prevValid[i].x)
            dySamples.add(nextValid[i].y - prevValid[i].y)
        }

        val medianDx = medianOrZero(dxSamples)
        val medianDy = medianOrZero(dySamples)

        val prevCenterX = prevValid.sumOf { it.x } / prevValid.size
        val prevCenterY = prevValid.sumOf { it.y } / prevValid.size
        val nextCenterX = nextValid.sumOf { it.x } / nextValid.size
        val nextCenterY = nextValid.sumOf { it.y } / nextValid.size

        val scaleSamples = ArrayList<Double>(prevValid.size)
        for (i in prevValid.indices) {
            val prevRadius = hypot(prevValid[i].x - prevCenterX, prevValid[i].y - prevCenterY)
            val nextRadius = hypot(nextValid[i].x - nextCenterX, nextValid[i].y - nextCenterY)
            if (prevRadius > 6.0 && nextRadius.isFinite()) {
                scaleSamples.add(nextRadius / prevRadius)
            }
        }

        val rawScale = if (scaleSamples.size >= 4) {
            medianOrZero(scaleSamples)
        } else {
            1.0
        }

        return TrackingTransform(
            dx = sanitizeShift(medianDx),
            dy = sanitizeShift(medianDy),
            scale = sanitizeScale(rawScale)
        )
    }

    private fun sanitizeShift(delta: Double): Double {
        if (!delta.isFinite()) return 0.0
        if (abs(delta) < minTrackShiftPx) return 0.0
        return delta.coerceIn(-maxTrackStepPx, maxTrackStepPx)
    }

    private fun sanitizeScale(scale: Double): Double {
        if (!scale.isFinite() || scale <= 0.0) return 1.0
        val clamped = scale.coerceIn(minTrackScale, maxTrackScale)
        if (abs(clamped - 1.0) < minScaleDelta) return 1.0
        return clamped
    }

    private fun medianOrZero(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val mid = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    private fun shiftRect(
        rect: android.graphics.Rect,
        transform: TrackingTransform,
        maxW: Int,
        maxH: Int
    ): android.graphics.Rect? {
        val centerX = (maxW - 1) / 2.0
        val centerY = (maxH - 1) / 2.0

        val shiftedLeft = (centerX + (rect.left - centerX) * transform.scale + transform.dx).toInt()
        val shiftedTop = (centerY + (rect.top - centerY) * transform.scale + transform.dy).toInt()
        val shiftedRight = (centerX + (rect.right - centerX) * transform.scale + transform.dx).toInt()
        val shiftedBottom = (centerY + (rect.bottom - centerY) * transform.scale + transform.dy).toInt()

        if (shiftedRight <= 0 || shiftedBottom <= 0 || shiftedLeft >= maxW || shiftedTop >= maxH) {
            return null
        }

        val left = shiftedLeft.coerceIn(0, maxW - 1)
        val top = shiftedTop.coerceIn(0, maxH - 1)
        val right = shiftedRight.coerceIn(1, maxW)
        val bottom = shiftedBottom.coerceIn(1, maxH)

        if (right <= left || bottom <= top) {
            return null
        }

        return android.graphics.Rect(left, top, right, bottom)
    }

    private fun shiftPoints(
        points: Array<android.graphics.Point>?,
        transform: TrackingTransform,
        maxW: Int,
        maxH: Int
    ): Array<android.graphics.Point>? {
        if (points == null) return null
        val centerX = (maxW - 1) / 2.0
        val centerY = (maxH - 1) / 2.0
        return points.map { p ->
            val x = (centerX + (p.x - centerX) * transform.scale + transform.dx).toInt()
                .coerceIn(0, maxW - 1)
            val y = (centerY + (p.y - centerY) * transform.scale + transform.dy).toInt()
                .coerceIn(0, maxH - 1)
            android.graphics.Point(x, y)
        }.toTypedArray()
    }


    private fun clearTrackingState() {
        synchronized(trackingLock) {
            trackedResults.clear()
            trackedTranslations.clear()
            trackingAnchorGray?.release()
            trackingAnchorGray = null
            lastTrackingTransform = TrackingTransform()
        }
        translationCache.clear()
        FrameOcrRepository.clearFrame()
        overlayEnabled = false
        setGeneratedTextVisible(false)
        runOnUiThread {
            binding.processedOverlay.setImageBitmap(null)
            binding.processedOverlay.visibility = View.INVISIBLE
        }
    }
    private fun clearDetectedBlocks() {
        resetDetectionSession()
        Log.d(TAG, "Reset detection session due to language change")
    }

    private fun resetDetectionSession() {
        clearTrackingState()
        ocrService.resetSessionState()
        lastProcessTime = 0L
        resetProcessingWork(keepDetecting = !freezeFrameMode && isOpenCvReady)
    }
    private fun processOnInterval(results: List<OcrService.DetectionResult>) {
        if (freezeFrameMode) return

        val now = System.currentTimeMillis()
        if (now - lastProcessTime <= processInterval) return
        lastProcessTime = now
        synchronized(trackingLock) {
            if (trackedResults.isNotEmpty()) {
                return
            }
        }
        val baseFrame = FrameOcrRepository.latestCameraFrame.value ?: return
        if (baseFrame.empty() || baseFrame.cols() <= 0 || baseFrame.rows() <= 0) {
            return
        }

        val translatedTexts = MutableList(results.size) { "" }
        val anchorGray = toGray(baseFrame)

        synchronized(trackingLock) {
            trackedResults = results.map { it.copy() }.toMutableList()
            trackedTranslations = translatedTexts
            trackingAnchorGray?.release()
            trackingAnchorGray = anchorGray
            lastTrackingTransform = TrackingTransform()
        }

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
                            markTranslationFinished()
                        }
                    },
                    onError = { error ->
                        Log.e(TAG, "Batch translation failed for $sourceLanguage->$targetLanguage", error)
                        markTranslationFinished()
                    }
                )
            } catch (e: Exception) {
                Log.e(TAG, "Batch translation setup failed for $sourceLanguage->$targetLanguage", e)
                markTranslationFinished()
            }
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
            if (freezeFrameMode) {
                exitFreezeFrameMode()
            } else {
                enterFreezeFrameMode()
            }
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
        resetFreezeState()
        binding.processedOverlay.scaleType = ImageView.ScaleType.FIT_CENTER
        startCameraWithOcr()
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
        val processed = FrameOcrRepository.currentFrame.value
        if (processed != null && !processed.empty()) {
            return processed.clone()
        }

        val raw = FrameOcrRepository.latestCameraFrame.value
        if (raw != null && !raw.empty()) {
            return raw.clone()
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
