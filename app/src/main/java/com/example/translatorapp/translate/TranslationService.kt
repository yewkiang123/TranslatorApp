package com.example.translatorapp.translate

import android.util.Log
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.common.model.RemoteModelManager
import com.google.mlkit.nl.languageid.LanguageIdentification
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.TranslateRemoteModel
import com.google.mlkit.nl.translate.Translator
import com.google.mlkit.nl.translate.TranslatorOptions

class TranslationService {

    private data class TranslatorKey(
        val sourceLanguage: String,
        val targetLanguage: String
    )

    private data class TranslatorEntry(
        val translator: Translator,
        var modelReady: Boolean = false
    )

    private val tag = "TranslationService"
    private val languageIdentifier = LanguageIdentification.getClient()
    private val modelManager = RemoteModelManager.getInstance()
    private val downloadConditions = DownloadConditions.Builder()
        .requireWifi()
        .build()
    private val translatorLock = Any()
    private val translators = HashMap<TranslatorKey, TranslatorEntry>()
    private val pendingDownloads = HashMap<TranslatorKey, com.google.android.gms.tasks.Task<Void>>()

    fun translate(
        text: String,
        sourceLanguage: String,
        targetLanguage: String,
        onSuccess: (String) -> Unit,
        onError: (Exception) -> Unit
    ) {
        // If source language is "auto", detect it first
        if (sourceLanguage == "auto") {
            detectLanguage(text) { detectedLang ->
                if (detectedLang != null) {
                    translateWithLanguage(text, detectedLang, targetLanguage, onSuccess, onError)
                } else {
                    onError(Exception("Could not detect language"))
                }
            }
        } else {
            translateWithLanguage(text, sourceLanguage, targetLanguage, onSuccess, onError)
        }
    }

    fun translateBatch(
        texts: List<String>,
        sourceLanguage: String,
        targetLanguage: String,
        onSuccess: (List<String>) -> Unit,
        onError: (Exception) -> Unit
    ) {
        if (texts.isEmpty()) {
            onSuccess(emptyList())
            return
        }
        if (sourceLanguage == "auto") {
            onError(IllegalArgumentException("Batch translation requires explicit source language"))
            return
        }
        translateTextsWithLanguage(
            texts = texts,
            sourceLanguage = sourceLanguage,
            targetLanguage = targetLanguage,
            onSuccess = onSuccess,
            onError = onError
        )
    }

    private fun translateWithLanguage(
        text: String,
        sourceLanguage: String,
        targetLanguage: String,
        onSuccess: (String) -> Unit,
        onError: (Exception) -> Unit
    ) {
        translateTextsWithLanguage(
            texts = listOf(text),
            sourceLanguage = sourceLanguage,
            targetLanguage = targetLanguage,
            onSuccess = { translated ->
                onSuccess(translated.firstOrNull() ?: text)
            },
            onError = onError
        )
    }

    private fun translateTextsWithLanguage(
        texts: List<String>,
        sourceLanguage: String,
        targetLanguage: String,
        onSuccess: (List<String>) -> Unit,
        onError: (Exception) -> Unit
    ) {
        if (sourceLanguage == targetLanguage) {
            onSuccess(texts)
            return
        }

        val (key, entry) = getOrCreateTranslator(sourceLanguage, targetLanguage)
        ensureModelDownloaded(
            key = key,
            entry = entry,
            onReady = {
                translateSequentially(
                    translator = entry.translator,
                    texts = texts,
                    index = 0,
                    translated = MutableList(texts.size) { "" },
                    sourceLanguage = sourceLanguage,
                    targetLanguage = targetLanguage,
                    onSuccess = onSuccess,
                    onError = onError
                )
            },
            onError = onError
        )
    }

    private fun getOrCreateTranslator(
        sourceLanguage: String,
        targetLanguage: String
    ): Pair<TranslatorKey, TranslatorEntry> {
        val key = TranslatorKey(sourceLanguage, targetLanguage)
        synchronized(translatorLock) {
            translators[key]?.let { cached ->
                return key to cached
            }
        }

        val options = TranslatorOptions.Builder()
            .setSourceLanguage(sourceLanguage)
            .setTargetLanguage(targetLanguage)
            .build()

        val created = TranslatorEntry(translator = Translation.getClient(options))
        synchronized(translatorLock) {
            val existing = translators[key]
            if (existing != null) {
                created.translator.close()
                return key to existing
            }
            translators[key] = created
            return key to created
        }
    }

    private fun ensureModelDownloaded(
        key: TranslatorKey,
        entry: TranslatorEntry,
        onReady: () -> Unit,
        onError: (Exception) -> Unit
    ) {
        val existingTask: com.google.android.gms.tasks.Task<Void>? = synchronized(translatorLock) {
            if (entry.modelReady) {
                return onReady()
            }
            pendingDownloads[key]?.let { return@synchronized it }

            entry.translator.downloadModelIfNeeded(downloadConditions).also { task ->
                pendingDownloads[key] = task
            }
        }

        existingTask
            ?.addOnSuccessListener {
                synchronized(translatorLock) {
                    entry.modelReady = true
                    pendingDownloads.remove(key)
                }
                onReady()
            }
            ?.addOnFailureListener { e ->
                synchronized(translatorLock) {
                    pendingDownloads.remove(key)
                }
                Log.e(tag, "Error downloading model for ${key.sourceLanguage}->${key.targetLanguage}", e)
                onError(e)
            }
    }

    private fun translateSequentially(
        translator: Translator,
        texts: List<String>,
        index: Int,
        translated: MutableList<String>,
        sourceLanguage: String,
        targetLanguage: String,
        onSuccess: (List<String>) -> Unit,
        onError: (Exception) -> Unit
    ) {
        if (index >= texts.size) {
            onSuccess(translated.toList())
            return
        }

        val text = texts[index]
        if (text.isBlank()) {
            translated[index] = text
            translateSequentially(
                translator = translator,
                texts = texts,
                index = index + 1,
                translated = translated,
                sourceLanguage = sourceLanguage,
                targetLanguage = targetLanguage,
                onSuccess = onSuccess,
                onError = onError
            )
            return
        }

        translator.translate(text)
            .addOnSuccessListener { translatedText ->
                Log.d(tag, "Translated from $sourceLanguage to $targetLanguage: $translatedText")
                translated[index] = translatedText
                translateSequentially(
                    translator = translator,
                    texts = texts,
                    index = index + 1,
                    translated = translated,
                    sourceLanguage = sourceLanguage,
                    targetLanguage = targetLanguage,
                    onSuccess = onSuccess,
                    onError = onError
                )
            }
            .addOnFailureListener { e ->
                Log.e(tag, "Error translating text", e)
                onError(e)
            }
    }

    private fun detectLanguage(text: String, callback: (String?) -> Unit) {
        languageIdentifier.identifyLanguage(text)
            .addOnSuccessListener { languageCode ->
                if (languageCode != "und") { // "und" means undetermined
                    Log.d(tag, "Detected language: $languageCode")
                    callback(languageCode)
                } else {
                    Log.d(tag, "Could not detect language")
                    callback(null)
                }
            }
            .addOnFailureListener { e ->
                Log.e(tag, "Language detection failed", e)
                callback(null)
            }
    }

    fun getDownloadedLanguageModels(
        onSuccess: (Set<String>) -> Unit,
        onError: (Exception) -> Unit
    ) {
        modelManager.getDownloadedModels(TranslateRemoteModel::class.java)
            .addOnSuccessListener { models ->
                onSuccess(models.mapNotNull { it.language }.toSet())
            }
            .addOnFailureListener { e ->
                Log.e(tag, "Failed to fetch downloaded language models", e)
                onError(e)
            }
    }

    fun downloadLanguageModel(
        languageCode: String,
        onSuccess: () -> Unit,
        onError: (Exception) -> Unit
    ) {
        val model = TranslateRemoteModel.Builder(languageCode).build()
        val conditions = DownloadConditions.Builder()
            .requireWifi()
            .build()

        modelManager.download(model, conditions)
            .addOnSuccessListener {
                Log.d(tag, "Downloaded model: $languageCode")
                onSuccess()
            }
            .addOnFailureListener { e ->
                Log.e(tag, "Failed downloading model: $languageCode", e)
                onError(e)
            }
    }

    fun cleanup() {
        val translatorsToClose = synchronized(translatorLock) {
            val clients = translators.values.map { it.translator }
            translators.clear()
            pendingDownloads.clear()
            clients
        }
        translatorsToClose.forEach { it.close() }
        languageIdentifier.close()
    }
}
