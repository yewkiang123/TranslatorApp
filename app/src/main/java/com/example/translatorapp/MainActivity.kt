package com.example.translatorapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.util.Log
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
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.suspendCancellableCoroutine
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlinx.coroutines.flow.conflate
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.withContext


class MainActivity : AppCompatActivity(), AdapterView.OnItemSelectedListener {

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

                    if (bmp != null) {
                        binding.processedOverlay.setImageBitmap(bmp)
                    }
                }
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

        val currentFrame = FrameOcrRepository.currentFrame.value?.clone() ?: return

        // Make mutable translated list
        val translatedTexts = MutableList(results.size) { "" }

        // Show original frame immediately
        FrameOcrRepository.updateFrame(currentFrame.clone())

        results.forEachIndexed { index, result ->

            // Skip if cached
            translationCache[result.text]?.let { cached ->

                translatedTexts[index] = cached

                lifecycleScope.launch(Dispatchers.Default) {
                    updateFrameProgressively(
                        currentFrame.clone(),   // ✅ CRITICAL FIX
                        results,
                        translatedTexts
                    )

                }

                return@forEachIndexed
            }

            // Launch translation WITHOUT blocking
            lifecycleScope.launch(Dispatchers.IO) {

                translateText(result.text, result.language) { translated ->

                    translationCache[result.text] = translated
                    translatedTexts[index] = translated

                    lifecycleScope.launch(Dispatchers.Default) {

                        updateFrameProgressively(
                            currentFrame,
                            results,
                            translatedTexts
                        )

                    }
                }

            }
        }
    }

    private fun updateFrameProgressively(
        originalFrame: Mat,
        results: List<OcrService.DetectionResult>,
        translatedTexts: List<String>
    ) {

        val safeFrame = originalFrame.clone()

        val updatedFrame = replacer.replaceText(
            safeFrame.clone(),   // ✅ CRITICAL FIX
            results,
            translatedTexts
        )

        lifecycleScope.launch(Dispatchers.Main) {

            FrameOcrRepository.updateFrame(updatedFrame)

        }

        //saveFrameToFile(updatedFrame.clone())
    }

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