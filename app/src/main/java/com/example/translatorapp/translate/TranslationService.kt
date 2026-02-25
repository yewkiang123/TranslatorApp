package com.example.translatorapp.translate

import android.util.Log
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.common.model.RemoteModelManager
import com.google.mlkit.nl.languageid.LanguageIdentification
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.TranslateRemoteModel
import com.google.mlkit.nl.translate.TranslatorOptions

class TranslationService {

    private val tag = "TranslationService"
    private val languageIdentifier = LanguageIdentification.getClient()
    private val modelManager = RemoteModelManager.getInstance()

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

    private fun translateWithLanguage(
        text: String,
        sourceLanguage: String,
        targetLanguage: String,
        onSuccess: (String) -> Unit,
        onError: (Exception) -> Unit
    ) {
        // Check if source and target are the same
        if (sourceLanguage == targetLanguage) {
            onSuccess(text) // Return original text
            return
        }

        val options = TranslatorOptions.Builder()
            .setSourceLanguage(sourceLanguage)
            .setTargetLanguage(targetLanguage)
            .build()

        val translator = Translation.getClient(options)

        val conditions = DownloadConditions.Builder()
            .requireWifi()
            .build()

        translator.downloadModelIfNeeded(conditions)
            .addOnSuccessListener {
                translator.translate(text)
                    .addOnSuccessListener { translatedText ->
                        Log.d(tag, "Translated from $sourceLanguage to $targetLanguage: $translatedText")
                        onSuccess(translatedText)
                        translator.close()
                    }
                    .addOnFailureListener { e ->
                        Log.e(tag, "Error translating text", e)
                        onError(e)
                        translator.close()
                    }
            }
            .addOnFailureListener { e ->
                Log.e(tag, "Error downloading model", e)
                onError(e)
                translator.close()
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
}
