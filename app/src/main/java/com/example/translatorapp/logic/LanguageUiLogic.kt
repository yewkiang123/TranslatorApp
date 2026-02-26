package com.example.translatorapp.logic

import java.util.Locale

object LanguageUiLogic {

    enum class DownloadIconState {
        HIDDEN,
        DOWNLOAD,
        DOWNLOADING
    }

    data class DownloadPresentation(
        val iconState: DownloadIconState,
        val enabled: Boolean
    )

    fun buildLanguageLabel(
        code: String,
        usedLabels: MutableSet<String>,
        displayLocale: Locale = Locale.ENGLISH
    ): String {
        val baseLabel = Locale(code).getDisplayName(displayLocale).ifBlank { code }
        var label = baseLabel
        if (usedLabels.contains(label)) {
            label = "$baseLabel ($code)"
        }
        usedLabels.add(label)
        return label
    }

    fun mapDetectedLanguage(detected: String): String {
        return when (detected) {
            "zh", "ja", "ko" -> detected
            else -> "en"
        }
    }

    fun resolveSourceLanguage(selectedSourceCode: String, detectedLanguage: String): String {
        return when {
            selectedSourceCode == "auto" -> mapDetectedLanguage(detectedLanguage)
            selectedSourceCode != "null" -> selectedSourceCode
            else -> "en"
        }
    }

    fun shouldBypassTranslation(sourceLanguage: String, targetLanguage: String, text: String): Boolean {
        return sourceLanguage == targetLanguage || text.trim().isEmpty()
    }

    fun computeDownloadPresentation(
        isDownloaded: Boolean,
        isDownloading: Boolean
    ): DownloadPresentation {
        if (isDownloaded) {
            return DownloadPresentation(
                iconState = DownloadIconState.HIDDEN,
                enabled = false
            )
        }

        return if (isDownloading) {
            DownloadPresentation(
                iconState = DownloadIconState.DOWNLOADING,
                enabled = false
            )
        } else {
            DownloadPresentation(
                iconState = DownloadIconState.DOWNLOAD,
                enabled = true
            )
        }
    }
}
