package com.example.translatorapp

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.widget.ImageView
import androidx.lifecycle.LifecycleCoroutineScope
import com.example.translatorapp.databinding.ActivityMainBinding
import com.example.translatorapp.ocr.FrameOcrRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

internal class FreezeFrameController(
    private val binding: ActivityMainBinding,
    private val lifecycleScope: LifecycleCoroutineScope,
    private val onEnterFreezeFrameMode: () -> Unit,
    private val onExitFreezeFrameMode: () -> Unit,
    private val showToast: (String) -> Unit
) {
    private var freezeFrameMode = false
    private var frozenFrameBitmap: Bitmap? = null
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

    val isFreezeFrameMode: Boolean
        get() = freezeFrameMode

    fun setup() {
        updateFreezeFrameActionButtons()
        setupClickListeners()
        setupFreezeFrameZoom()
    }

    fun enterFreezeFrameMode() {
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
                onEnterFreezeFrameMode()
                frozenFrameBitmap = frozenBitmap
                updateFreezeFrameActionButtons()

                binding.processedOverlay.scaleType = ImageView.ScaleType.MATRIX
                binding.processedOverlay.setImageBitmap(frozenBitmap)
                binding.processedOverlay.post { resetFreezeImageMatrix() }
                binding.processedOverlay.visibility = View.VISIBLE
            }
        }
    }

    fun exitFreezeFrameMode() {
        if (!freezeFrameMode) return

        freezeFrameMode = false
        clearFreezeFrameDisplay()
        frozenFrameBitmap = null
        updateFreezeFrameActionButtons()
        resetFreezeState()
        binding.processedOverlay.scaleType = ImageView.ScaleType.FIT_CENTER
        onExitFreezeFrameMode()
    }

    fun release() {
        clearFreezeFrameDisplay()
        frozenFrameBitmap = null
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

    private fun setupFreezeFrameZoom() {
        val context = binding.root.context
        scaleGestureDetector = ScaleGestureDetector(
            context,
            object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
                override fun onScale(detector: ScaleGestureDetector): Boolean {
                    if (!freezeFrameMode) return false
                    applyFreezeScale(detector.scaleFactor, detector.focusX, detector.focusY)
                    return true
                }
            }
        )

        freezeGestureDetector = GestureDetector(
            context,
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
            showToast(binding.root.context.getString(R.string.photo_unavailable))
            return
        }
        val bitmapToSave = Bitmap.createBitmap(sourceBitmap)
        binding.savePhotoButton.isEnabled = false

        lifecycleScope.launch(Dispatchers.IO) {
            val resolver = binding.root.context.applicationContext.contentResolver
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
                        binding.root.context.getString(R.string.photo_saved)
                    } else {
                        binding.root.context.getString(R.string.photo_save_failed)
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

    private fun clearFreezeFrameDisplay() {
        binding.processedOverlay.setImageDrawable(null)
        binding.processedOverlay.imageMatrix = Matrix()
        binding.processedOverlay.visibility = View.INVISIBLE
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
            if (rgba.empty() || rgba.cols() <= 0 || rgba.rows() <= 0) {
                return null
            }

            Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888).also { bmp ->
                org.opencv.android.Utils.matToBitmap(rgba, bmp)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to convert Mat to Bitmap", e)
            null
        } finally {
            rgba.release()
        }
    }

    private companion object {
        const val TAG = "FreezeFrame"
    }
}
