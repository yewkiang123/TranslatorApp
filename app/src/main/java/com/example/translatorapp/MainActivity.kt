package com.example.translatorapp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.translatorapp.camera.CameraController
import com.example.translatorapp.databinding.ActivityMainBinding
import com.example.translatorapp.ocr.OcrService
import com.example.translatorapp.translate.TranslationService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraController: CameraController
    private lateinit var ocrService: OcrService
    private lateinit var translationService: TranslationService
    private lateinit var languagePanelController: LanguagePanelController
    private lateinit var freezeFrameController: FreezeFrameController
    private lateinit var metricsController: MetricsController
    private lateinit var processingIndicatorController: ProcessingIndicatorController
    private lateinit var trackingSessionController: TrackingSessionController
    private lateinit var ocrCoordinator: MainOcrCoordinator

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var isOpenCvReady = false
    private var hasShownOpenCvUnavailableWarning = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        isOpenCvReady = initializeOpenCV()
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initServices()
        initControllers()
        setupCamera()
        setupBackPressHandler()

        processingIndicatorController.render(force = true)
        metricsController.render(force = true)
        ocrCoordinator.setupOcrListener()
        languagePanelController.setup()
        freezeFrameController.setup()
        trackingSessionController.bindOverlayRenderer()
        trackingSessionController.startTrackingLoop()

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
        ocrService.setSourceLanguageHint("en")
        translationService = TranslationService()
    }

    private fun initControllers() {
        metricsController = MetricsController(binding)
        processingIndicatorController = ProcessingIndicatorController(
            binding = binding,
            isOpenCvReady = { isOpenCvReady }
        )
        languagePanelController = LanguagePanelController(
            binding = binding,
            translationService = translationService,
            ocrService = ocrService,
            onLanguageChanged = { ocrCoordinator.clearDetectedBlocks() },
            showToast = ::showToast
        )
        trackingSessionController = TrackingSessionController(
            binding = binding,
            lifecycleScope = lifecycleScope,
            isFreezeFrameMode = { freezeFrameController.isFreezeFrameMode },
            onTrackingLost = { ocrCoordinator.requestRedetectionAfterTrackingLoss() },
            onTrackingStateCleared = {
                metricsController.clearActiveSession()
                ocrCoordinator.clearTranslationCache()
            },
            onGeneratingActive = { active -> processingIndicatorController.setGeneratingActive(active) },
            onGeneratedTextVisible = { visible ->
                processingIndicatorController.setGeneratedTextVisible(visible)
            },
            onOverlayRendered = { metricsController.recordOverlayRendered() }
        )
        ocrCoordinator = MainOcrCoordinator(
            lifecycleScope = lifecycleScope,
            ocrService = ocrService,
            translationService = translationService,
            languagePanelController = languagePanelController,
            processingIndicatorController = processingIndicatorController,
            metricsController = metricsController,
            trackingSessionController = trackingSessionController,
            isFreezeFrameMode = { freezeFrameController.isFreezeFrameMode },
            isOpenCvReady = { isOpenCvReady }
        )
        freezeFrameController = FreezeFrameController(
            binding = binding,
            lifecycleScope = lifecycleScope,
            onEnterFreezeFrameMode = {
                cameraController.stopCamera()
                processingIndicatorController.resetProcessingWork(keepDetecting = false)
            },
            onExitFreezeFrameMode = {
                ocrCoordinator.resetDetectionSession()
                startCameraWithOcr()
            },
            showToast = ::showToast
        )
    }

    private fun setupCamera() {
        cameraController = CameraController(this, this, cameraExecutor, binding)
    }

    private fun startCameraWithOcr() {
        metricsController.resetRuntimeMetrics()
        if (!isOpenCvReady) {
            cameraController.startCamera()
            processingIndicatorController.setDetectingActive(false)
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
        processingIndicatorController.setDetectingActive(true)
    }

    private fun setupBackPressHandler() {
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (freezeFrameController.isFreezeFrameMode) {
                    freezeFrameController.exitFreezeFrameMode()
                    return
                }

                isEnabled = false
                onBackPressedDispatcher.onBackPressed()
                isEnabled = true
            }
        })
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
        ocrCoordinator.cancelPendingRedetection()
        trackingSessionController.stop()
        freezeFrameController.release()
        ocrService.cleanup()
        translationService.cleanup()
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
