package com.example.translatorapp

import android.animation.Animator
import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.animation.PropertyValuesHolder
import android.animation.ValueAnimator
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.animation.LinearInterpolator
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.ImageButton
import android.widget.TextView
import androidx.interpolator.view.animation.FastOutSlowInInterpolator
import com.example.translatorapp.databinding.ActivityMainBinding
import com.example.translatorapp.logic.LanguageUiLogic
import com.example.translatorapp.ocr.OcrService
import com.example.translatorapp.translate.TranslationService
import com.google.mlkit.nl.translate.TranslateLanguage

internal data class LanguageOption(
    val label: String,
    val code: String
)

internal class LanguagePanelController(
    private val binding: ActivityMainBinding,
    private val translationService: TranslationService,
    private val ocrService: OcrService,
    private val onLanguageChanged: () -> Unit,
    private val showToast: (String) -> Unit
) : AdapterView.OnItemSelectedListener {

    private val context = binding.root.context
    private val layoutInflater = LayoutInflater.from(context)
    private val targetLanguageOptions = mutableListOf<LanguageOption>()
    private val sourceLanguageOptions = mutableListOf<LanguageOption>()
    private val downloadedLanguageCodes = mutableSetOf<String>()
    private val downloadingLanguageCodes = mutableSetOf<String>()
    private val languageCodeByDisplay = HashMap<String, String>()

    private lateinit var sourceSpinnerAdapter: SourceLanguageAdapter
    private lateinit var targetSpinnerAdapter: TargetLanguageAdapter

    private var targetLanguageCode = DEFAULT_LANGUAGE_CODE
    private var sourceLanguageDisplay = ""

    val selectedTargetLanguageCode: String
        get() = targetLanguageCode

    val selectedSourceLanguageCode: String
        get() = getLanguageCode(sourceLanguageDisplay)

    fun setup() {
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

        supportedLanguages.forEach { option ->
            sourceLanguageOptions.add(option)
            targetLanguageOptions.add(option)
            languageCodeByDisplay[option.label] = option.code
        }

        sourceSpinnerAdapter = SourceLanguageAdapter()
        targetSpinnerAdapter = TargetLanguageAdapter()

        binding.sourceLanguage.adapter = sourceSpinnerAdapter
        binding.targetLanguage.adapter = targetSpinnerAdapter
        binding.sourceLanguage.onItemSelectedListener = this
        binding.targetLanguage.onItemSelectedListener = this
        binding.targetLanguage.setOnTouchListener { _, _ ->
            refreshDownloadedLanguageStatus()
            false
        }
        binding.sourceLanguage.setOnTouchListener { _, _ ->
            refreshDownloadedLanguageStatus()
            false
        }

        val targetEnglishIndex = targetLanguageOptions.indexOfFirst { it.code == DEFAULT_LANGUAGE_CODE }
            .coerceAtLeast(0)
        val sourceEnglishIndex = sourceLanguageOptions.indexOfFirst { it.code == DEFAULT_LANGUAGE_CODE }
            .coerceAtLeast(0)
        targetLanguageCode = targetLanguageOptions
            .getOrNull(targetEnglishIndex)
            ?.code
            ?: DEFAULT_LANGUAGE_CODE
        sourceLanguageDisplay = sourceLanguageOptions
            .getOrNull(sourceEnglishIndex)
            ?.label
            ?: sourceLanguageOptions.firstOrNull()?.label.orEmpty()
        binding.targetLanguage.setSelection(targetEnglishIndex)
        binding.sourceLanguage.setSelection(sourceEnglishIndex)
        updateOcrSourceLanguageHint()
        refreshDownloadedLanguageStatus()
    }

    override fun onItemSelected(parent: AdapterView<*>, view: View?, pos: Int, id: Long) {
        when (parent.id) {
            R.id.targetLanguage -> {
                val selectedOption = targetLanguageOptions.getOrNull(pos) ?: return
                val newLang = selectedOption.code
                if (newLang != targetLanguageCode) {
                    targetLanguageCode = newLang
                    onLanguageChanged()
                    Log.d(TAG, "Target changed -> cleared blocks")
                }
            }

            R.id.sourceLanguage -> {
                val selected = sourceLanguageOptions.getOrNull(pos)?.label ?: return
                if (selected != sourceLanguageDisplay) {
                    sourceLanguageDisplay = selected
                    updateOcrSourceLanguageHint()
                    onLanguageChanged()
                    Log.d(TAG, "Source changed -> cleared blocks")
                }
            }
        }
    }

    override fun onNothingSelected(parent: AdapterView<*>) {
        when (parent.id) {
            R.id.targetLanguage -> targetLanguageCode = DEFAULT_LANGUAGE_CODE
            R.id.sourceLanguage -> {
                sourceLanguageDisplay = sourceLanguageOptions
                    .firstOrNull { it.code == DEFAULT_LANGUAGE_CODE }
                    ?.label
                    ?: sourceLanguageOptions.firstOrNull()?.label.orEmpty()
                updateOcrSourceLanguageHint()
            }
        }
    }

    private fun getLanguageCode(displayName: String): String {
        return languageCodeByDisplay[displayName] ?: DEFAULT_LANGUAGE_CODE
    }

    private fun updateOcrSourceLanguageHint() {
        ocrService.setSourceLanguageHint(selectedSourceLanguageCode)
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
                binding.root.post {
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
                binding.root.post {
                    downloadingLanguageCodes.remove(code)
                    downloadedLanguageCodes.add(code)
                    notifyLanguageAdaptersChanged()
                    showToast("${option.label} language pack downloaded")
                }
            },
            onError = { error ->
                binding.root.post {
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
        downloadIcon.contentDescription = context.getString(
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
        context,
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
                allowDownload = true
            )
            return view
        }
    }

    private inner class TargetLanguageAdapter : ArrayAdapter<LanguageOption>(
        context,
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
            bindLanguageDropdownRow(view = view, option = getItem(position), allowDownload = true)
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

    private companion object {
        const val TAG = "LanguagePanel"
        const val DEFAULT_LANGUAGE_CODE = "en"
    }
}
