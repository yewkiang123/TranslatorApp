package com.example.translatorapp.camera

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.example.translatorapp.databinding.ActivityMainBinding
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService

class CameraController(
    private val lifecycleOwner: LifecycleOwner,
    private val context: Context,
    private val cameraExecutor: ExecutorService,
    private val viewBinding: ActivityMainBinding
) {
    private val fileFormat = "yyyy-MM-dd-HH-mm-ss-SSS"
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null

    fun startCamera(analyzer: ImageAnalysis.Analyzer? = null) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            val boundCameraProvider = cameraProviderFuture.get()
            cameraProvider = boundCameraProvider

            // Preview
            val preview = Preview.Builder().build().also {
                it.surfaceProvider = viewBinding.viewFinder.surfaceProvider
            }

            // ImageCapture
            imageCapture = ImageCapture.Builder().build()

            // ImageAnalysis (only if analyzer provided)
            if (analyzer != null) {
                imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        // FIX: Use cameraExecutor, NOT main executor
                        it.setAnalyzer(cameraExecutor, analyzer)
                    }
            }

            // Camera selector
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                boundCameraProvider.unbindAll()

                // Bind use cases
                val useCases = mutableListOf<UseCase>(preview)
                if (analyzer != null) {
                    useCases.add(imageAnalysis!!)
                }

                boundCameraProvider.bindToLifecycle(
                    lifecycleOwner, cameraSelector,
                    *useCases.toTypedArray()
                )
            } catch (e: Exception) {
                Log.e("CAMERA", "Use case binding failed", e)
            }

        }, ContextCompat.getMainExecutor(context))
    }

    fun stopCamera() {
        try {
            imageAnalysis?.clearAnalyzer()
            cameraProvider?.unbindAll()
        } catch (e: Exception) {
            Log.e("CAMERA", "Failed to stop camera", e)
        }
    }

    fun takePhoto() {
        val imageCapture = imageCapture ?: return

        val name = SimpleDateFormat(fileFormat, Locale.US)
            .format(System.currentTimeMillis())

        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/CameraX-Image")
            }
        }

        val outputOptions = ImageCapture.OutputFileOptions.Builder(
            context.contentResolver,
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            contentValues
        ).build()

        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e("CAMERA", "Photo capture failed: ${exc.message}", exc)
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val msg = "Photo capture succeeded: ${output.savedUri}"
                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                    Log.d("CAMERA", msg)
                }
            }
        )
    }
}
