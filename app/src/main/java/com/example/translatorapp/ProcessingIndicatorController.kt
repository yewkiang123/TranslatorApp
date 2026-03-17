package com.example.translatorapp

import android.content.res.ColorStateList
import android.view.View
import androidx.core.content.ContextCompat
import com.example.translatorapp.databinding.ActivityMainBinding

internal class ProcessingIndicatorController(
    private val binding: ActivityMainBinding,
    private val isOpenCvReady: () -> Boolean
) {
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

    private val context = binding.root.context
    private val processingStateLock = Any()
    private var detectingActive = false
    private var noTextDetected = false
    private var translatingJobsInFlight = 0
    private var generatingJobsInFlight = 0
    private var generatedTextVisible = false
    private var lastProcessingIndicatorState: ProcessingIndicatorState? = null

    fun render(force: Boolean = false) {
        renderProcessingIndicator(force)
    }

    fun setDetectingActive(active: Boolean) {
        synchronized(processingStateLock) {
            detectingActive = active
            if (active) {
                noTextDetected = false
            }
        }
        renderProcessingIndicator()
    }

    fun setNoTextDetected(active: Boolean) {
        synchronized(processingStateLock) {
            noTextDetected = active
            if (active) {
                detectingActive = false
            }
        }
        renderProcessingIndicator()
    }

    fun showNoTextDetected() {
        synchronized(processingStateLock) {
            noTextDetected = true
            detectingActive = false
        }
        renderProcessingIndicator(force = true)
    }

    fun resetProcessingWork(keepDetecting: Boolean) {
        synchronized(processingStateLock) {
            translatingJobsInFlight = 0
            generatingJobsInFlight = 0
            generatedTextVisible = false
            detectingActive = keepDetecting
            noTextDetected = false
        }
        renderProcessingIndicator(force = true)
    }

    fun markTranslationStarted() {
        synchronized(processingStateLock) {
            translatingJobsInFlight++
        }
        renderProcessingIndicator()
    }

    fun markTranslationFinished() {
        synchronized(processingStateLock) {
            if (translatingJobsInFlight > 0) {
                translatingJobsInFlight--
            }
        }
        renderProcessingIndicator()
    }

    fun setGeneratingActive(active: Boolean) {
        synchronized(processingStateLock) {
            if (active) {
                generatingJobsInFlight++
            } else if (generatingJobsInFlight > 0) {
                generatingJobsInFlight--
            }
        }
        renderProcessingIndicator()
    }

    fun setGeneratedTextVisible(visible: Boolean) {
        synchronized(processingStateLock) {
            generatedTextVisible = visible
        }
        renderProcessingIndicator()
    }

    private fun resolveProcessingIndicatorState(): ProcessingIndicatorState {
        return synchronized(processingStateLock) {
            when {
                generatedTextVisible -> ProcessingIndicatorState.IDLE
                !isOpenCvReady() -> ProcessingIndicatorState.DISABLED
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

        binding.root.post {
            binding.processingIndicatorText.setText(style.labelRes)
            binding.processingIndicatorText.setTextColor(
                ContextCompat.getColor(context, style.textColorRes)
            )
            binding.processingIndicatorCard.setCardBackgroundColor(
                ContextCompat.getColor(context, style.backgroundColorRes)
            )
            binding.processingIndicatorProgress.visibility =
                if (style.spinnerVisible) View.VISIBLE else View.GONE
            binding.processingIndicatorProgress.indeterminateTintList =
                ColorStateList.valueOf(ContextCompat.getColor(context, style.spinnerColorRes))
        }
    }
}
