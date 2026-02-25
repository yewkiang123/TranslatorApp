package com.example.translatorapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.translatorapp.camera.CameraController
import com.example.translatorapp.databinding.ActivityMainBinding
import com.example.translatorapp.ocr.OcrService
import com.example.translatorapp.translate.TranslationService
import com.google.mlkit.nl.translate.TranslateLanguage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import java.util.concurrent.Executors
import androidx.core.graphics.createBitmap
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
    private data class TrackingTransform(
        val dx: Double = 0.0,
        val dy: Double = 0.0,
        val scale: Double = 1.0
    )

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraController: CameraController
    private lateinit var ocrService: OcrService
    private lateinit var translationService: TranslationService

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var lastProcessTime = 0L
    private val processInterval = 200L
    private var targetLanguageCode = "en"
    private var sourceLanguageDisplay = "auto"
    private val translationCache = HashMap<String, String>()

    private var renderJob: Job? = null
    private var trackingJob: Job? = null
    @Volatile private var overlayEnabled = false
    private val trackingLock = Any()
    private var trackedResults: MutableList<OcrService.DetectionResult> = mutableListOf()
    private var trackedTranslations: MutableList<String> = mutableListOf()
    private var prevGrayTrack: Mat? = null
    private val trackIntervalMs = 120L
    private val panThresholdPx = 80.0
    private val minTrackShiftPx = 1.5
    private val maxTrackStepPx = 30.0
    private val minScaleDelta = 0.015
    private val minTrackScale = 0.90
    private val maxTrackScale = 1.10
    private val zoomResetThreshold = 0.22

    private val replacer = SceneTextReplacer()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize OpenCV FIRST before anything else
        initializeOpenCV()

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initServices()
        setupOcrListener()
        setupSpinners()
        setupCamera()
        setupClickListeners()
        observeProcessedFrames()
        startTrackingLoop()

        if (allPermissionsGranted()) {
            startCameraWithOcr()
        } else {
            requestPermissions()
        }
    }

    private fun initializeOpenCV() {
        try {
            // Try to load OpenCV
            if (!org.opencv.android.OpenCVLoader.initDebug()) {
                Log.e("OpenCV", "OpenCV initialization failed!")
                Toast.makeText(this, "OpenCV failed to load", Toast.LENGTH_SHORT).show()
            } else {
                Log.d("OpenCV", "OpenCV initialized successfully")
            }
        } catch (e: UnsatisfiedLinkError) {
            Log.e("OpenCV", "OpenCV library not found", e)
            Toast.makeText(this, "OpenCV library missing", Toast.LENGTH_LONG).show()
        }
    }

    private fun initServices() {
        ocrService = OcrService()
        translationService = TranslationService()
    }

    private fun setupOcrListener() {
        ocrService.detections.onEach { results ->
            if (results.isNotEmpty()) {
                processOnInterval(results)
            }
        }.launchIn(lifecycleScope)
    }

    private var reusableBitmap: Bitmap? = null

    private fun observeProcessedFrames() {

        renderJob?.cancel()

        renderJob = lifecycleScope.launch {
            FrameOcrRepository.currentFrame
                .filterNotNull()
                .conflate() // keep only the latest if we're slow
                .collectLatest { mat ->  // CANCEL previous render if a new frame arrives

                    if (mat.empty()) return@collectLatest

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

                    if (bmp != null && overlayEnabled) {
                        binding.processedOverlay.setImageBitmap(bmp)
                    }
                }
        }
    }

    private fun startTrackingLoop() {
        trackingJob?.cancel()
        trackingJob = lifecycleScope.launch(Dispatchers.Default) {
            while (isActive) {
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
        val (results, translations) = synchronized(trackingLock) {
            if (trackedResults.isEmpty()) {
                return@synchronized null to null
            }
            trackedResults.map { it.copy() } to trackedTranslations.toList()
        }

        if (results == null || translations == null) {
            frame.release()
            return
        }

        val currentGray = toGray(frame)
        val previousGray = prevGrayTrack
        val transform = estimateTrackingTransform(previousGray, currentGray)
        prevGrayTrack = currentGray
        previousGray?.release()

        if (abs(transform.dx) + abs(transform.dy) > panThresholdPx ||
            abs(transform.scale - 1.0) > zoomResetThreshold) {
            clearTrackingState()
            frame.release()
            return
        }

        val shifted = results.map { result ->
            val box = result.boundingBox ?: return@map result
            val shiftedBox = shiftRect(box, transform, frame.cols(), frame.rows())
            val shiftedCorners = if (shiftedBox != null) {
                shiftPoints(result.cornerPoints, transform, frame.cols(), frame.rows())
            } else {
                null
            }
            result.copy(boundingBox = shiftedBox, cornerPoints = shiftedCorners)
        }

        if (shifted.none { it.boundingBox != null }) {
            clearTrackingState()
            frame.release()
            return
        }

        synchronized(trackingLock) {
            trackedResults = shifted.toMutableList()
        }

        if (translations.none { it.isNotBlank() }) {
            frame.release()
            return
        }

        val updated = replacer.replaceText(frame, shifted, translations)
        FrameOcrRepository.updateFrame(updated)
        updated.release()
        overlayEnabled = true
        runOnUiThread {
            binding.processedOverlay.visibility = View.VISIBLE
        }
        frame.release()
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

    private fun estimateTrackingTransform(prev: Mat?, current: Mat): TrackingTransform {
        if (prev == null || prev.empty()) return TrackingTransform()

        val points = MatOfPoint()
        Imgproc.goodFeaturesToTrack(prev, points, 120, 0.01, 8.0)
        if (points.empty()) {
            points.release()
            return TrackingTransform()
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
            return TrackingTransform()
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
        }
        translationCache.clear()
        FrameOcrRepository.clearFrame()
        overlayEnabled = false
        prevGrayTrack?.release()
        prevGrayTrack = null
        runOnUiThread {
            binding.processedOverlay.setImageBitmap(null)
            binding.processedOverlay.visibility = View.INVISIBLE
        }
    }
    private fun clearDetectedBlocks() {

        // 1️⃣ Clear OCR detection cache
        ocrService.resetCache()

        // 2️⃣ Clear displayed frame
        FrameOcrRepository.clearFrame()

        // 3️⃣ Force immediate reprocessing
        lastProcessTime = 0L

        Log.d(TAG, "Cleared detected blocks due to language change")
    }
    private fun processOnInterval(results: List<OcrService.DetectionResult>) {

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

        // Make mutable translated list
        val translatedTexts = MutableList(results.size) { "" }

        synchronized(trackingLock) {
            trackedResults = results.map { it.copy() }.toMutableList()
            trackedTranslations = translatedTexts
        }

        results.forEachIndexed { index, result ->

            // Skip if cached
            translationCache[result.text]?.let { cached ->

                if (index < translatedTexts.size) {
                    translatedTexts[index] = cached
                } else {
                    return@forEachIndexed
                }

                synchronized(trackingLock) {
                    if (index < trackedTranslations.size) {
                        trackedTranslations[index] = cached
                    }
                }

                return@forEachIndexed
            }

            // Launch translation WITHOUT blocking
            lifecycleScope.launch(Dispatchers.IO) {

                translateText(result.text, result.language) { translated ->

                    translationCache[result.text] = translated
                    if (index >= translatedTexts.size) return@translateText
                    translatedTexts[index] = translated

                    synchronized(trackingLock) {
                        if (index < trackedTranslations.size) {
                            trackedTranslations[index] = translated
                        }
                    }
                }

            }
        }

    }

    // Rendering is handled in trackAndRender using tracked patches + translations.

    private fun saveFrameToFile(frameMat: Mat) {

        lifecycleScope.launch(Dispatchers.IO) {

            try {

                val safeMat = frameMat.clone()   // ✅ CRITICAL FIX

                if (safeMat.empty()) {
                    Log.e("FrameSave", "Mat is empty — skipping save")
                    return@launch
                }

                val timestamp = System.currentTimeMillis()
                val filename = "frame_${timestamp}.jpg"

                val directory = getExternalFilesDir(android.os.Environment.DIRECTORY_PICTURES)
                val file = File(directory, filename)

                val bitmap = createBitmap(safeMat.cols(), safeMat.rows())

                org.opencv.android.Utils.matToBitmap(safeMat, bitmap)

                FileOutputStream(file).use {
                    bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 95, it)
                }

                bitmap.recycle()
                safeMat.release()

                Log.d("FrameSave", "Saved: ${file.absolutePath}")

            } catch (e: Exception) {

                Log.e("FrameSave", "Save failed", e)

            }
        }
    }
    private fun saveRoiToFile(roiMat: org.opencv.core.Mat?, originalText: String) {
        if (roiMat == null || roiMat.empty()) return

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Create filename with timestamp
                val timestamp = System.currentTimeMillis()
                val safeText = originalText.replace("[^a-zA-Z0-9]".toRegex(), "_").take(20)
                val filename = "roi_${timestamp}_${safeText}.jpg"

                // Save to app's external files directory
                val directory = getExternalFilesDir(android.os.Environment.DIRECTORY_PICTURES)
                val file = File(directory, filename)

                // Convert Mat to Bitmap and save as JPEG
                val bitmap = createBitmap(roiMat.cols(), roiMat.rows())
                org.opencv.android.Utils.matToBitmap(roiMat, bitmap)

                FileOutputStream(file).use { out ->
                    bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, out)
                }

                bitmap.recycle()

                Log.d("ROISave", "Saved ROI to: ${file.absolutePath}")

                // Also log the path for easy ADB pull
                val adbPath = "/storage/emulated/0/Android/data/${packageName}/files/Pictures/$filename"
                Log.d("ADB Pull", "To pull: adb pull \"$adbPath\"")

            } catch (e: Exception) {
                Log.e("ROI Save", "Failed to save ROI", e)
            }
        }
    }

    private fun saveRoiWithText(roiMat: org.opencv.core.Mat?, translatedText: String, originalText: String) {
        if (roiMat == null || roiMat.empty()) return

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val timestamp = System.currentTimeMillis()
                val filename = "translated_${timestamp}.jpg"
                val directory = getExternalFilesDir(android.os.Environment.DIRECTORY_PICTURES)
                val file = File(directory, filename)

                // Create a new Mat with text overlay
                val displayMat = roiMat.clone()

                // Add text to the image
                val text = "Orig: $originalText\nTrans: $translatedText"
                org.opencv.imgproc.Imgproc.putText(
                    displayMat,
                    text,
                    org.opencv.core.Point(10.0, 30.0),
                    org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    org.opencv.core.Scalar(255.0, 0.0, 0.0),
                    1
                )

                // Save as JPEG
                org.opencv.imgcodecs.Imgcodecs.imwrite(file.absolutePath, displayMat)

                displayMat.release()

                Log.d("ROI Save", "Saved translated ROI: ${file.absolutePath}")

            } catch (e: Exception) {
                Log.e("ROI Save", "Failed to save translated ROI", e)
            }
        }
    }

    private fun translateText(
        text: String,
        detectedLanguage: String,
        onResult: (String) -> Unit
    ) {
        val sourceLang = when {
            sourceLanguageDisplay == "auto" -> mapDetectedLanguage(detectedLanguage)
            sourceLanguageDisplay != "null" -> getLanguageCode(sourceLanguageDisplay)
            else -> "en"
        }

        if (sourceLang == targetLanguageCode || text.trim().isEmpty()) {
            showToast("Languages are same:\n$text")
            onResult(text)   // still return something
            return
        }

        translationService.translate(
            text = text,
            sourceLanguage = sourceLang,
            targetLanguage = targetLanguageCode,
            onSuccess = { translated ->
                showToast("Translated: $translated")
                onResult(translated)      // ✅ send result back
            },
            onError = { error ->
                Log.e("Translation", "Failed", error)
            }
        )
    }


    private fun mapDetectedLanguage(detected: String): String {
        return when (detected) {
            "zh", "ja", "ko" -> detected
            else -> "en"
        }
    }

    private fun setupSpinners() {
        val languages = TranslateLanguage.getAllLanguages()
            .map { Locale(it).getDisplayName(Locale.ENGLISH) }
            .toMutableList()
            .apply { add(0, "Detected Language") }

        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, languages)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)

        binding.targetLanguage.adapter = adapter
        binding.sourceLanguage.adapter = adapter

        binding.targetLanguage.onItemSelectedListener = this
        binding.sourceLanguage.onItemSelectedListener = this

        binding.targetLanguage.setSelection(languages.indexOf("English"))
        binding.sourceLanguage.setSelection(languages.indexOf("Detected Language"))
    }

    override fun onItemSelected(parent: AdapterView<*>, view: android.view.View?, pos: Int, id: Long) {
        val selected = parent.getItemAtPosition(pos) as String

        when (parent.id) {
            R.id.targetLanguage -> {
                val newLang = getLanguageCode(selected)

                if (newLang != targetLanguageCode) {

                    targetLanguageCode = newLang

                    clearDetectedBlocks()

                    Log.d(TAG, "Target changed → cleared blocks")
                }
            }
            R.id.sourceLanguage -> {
                if (selected != sourceLanguageDisplay) {

                    sourceLanguageDisplay = selected

                    clearDetectedBlocks()

                    Log.d(TAG, "Source changed → cleared blocks")
                }
            }
        }
    }

    override fun onNothingSelected(parent: AdapterView<*>) {
        when (parent.id) {
            R.id.targetLanguage -> targetLanguageCode = "en"
            R.id.sourceLanguage -> sourceLanguageDisplay = "auto"
        }
    }

    private fun getLanguageCode(displayName: String): String {
        return when (displayName) {
            "Detected Language" -> "auto"
            else -> Locale.getAvailableLocales()
                .find { it.getDisplayName(Locale.ENGLISH).equals(displayName, ignoreCase = true) }
                ?.language ?: "en"
        }
    }

    private fun setupCamera() {
        cameraController = CameraController(this, this, cameraExecutor, binding)
    }

    private fun startCameraWithOcr() {
        cameraController.startCamera(ocrService)
    }

    private fun setupClickListeners() {
        binding.imageCaptureButton.setOnClickListener {
            cameraController.takePhoto()
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
        ocrService.cleanup()
        trackingJob?.cancel()
        prevGrayTrack?.release()
        prevGrayTrack = null
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
