package com.example.translatorapp.logic

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Locale

class LanguageUiLogicTest {

    @Test
    fun buildLanguageLabel_keepsOriginalWhenUnique() {
        val used = mutableSetOf<String>()

        val label = LanguageUiLogic.buildLanguageLabel("en", used, Locale.ENGLISH)

        assertEquals("English", label)
        assertTrue(used.contains("English"))
    }

    @Test
    fun buildLanguageLabel_appendsCodeWhenDuplicate() {
        val used = mutableSetOf("English")

        val label = LanguageUiLogic.buildLanguageLabel("en", used, Locale.ENGLISH)

        assertEquals("English (en)", label)
        assertTrue(used.contains("English (en)"))
    }

    @Test
    fun mapDetectedLanguage_keepsSupportedCjkCodes() {
        assertEquals("zh", LanguageUiLogic.mapDetectedLanguage("zh"))
        assertEquals("ja", LanguageUiLogic.mapDetectedLanguage("ja"))
        assertEquals("ko", LanguageUiLogic.mapDetectedLanguage("ko"))
    }

    @Test
    fun mapDetectedLanguage_defaultsUnknownToEnglish() {
        assertEquals("en", LanguageUiLogic.mapDetectedLanguage("fr"))
        assertEquals("en", LanguageUiLogic.mapDetectedLanguage("und"))
    }

    @Test
    fun resolveSourceLanguage_autoUsesDetectedMapping() {
        assertEquals("ja", LanguageUiLogic.resolveSourceLanguage("auto", "ja"))
        assertEquals("en", LanguageUiLogic.resolveSourceLanguage("auto", "fr"))
    }

    @Test
    fun resolveSourceLanguage_prefersSelectedCodeWhenNotAuto() {
        assertEquals("de", LanguageUiLogic.resolveSourceLanguage("de", "ja"))
        assertEquals("en", LanguageUiLogic.resolveSourceLanguage("null", "ja"))
    }

    @Test
    fun shouldBypassTranslation_trueWhenSameLanguage() {
        assertTrue(LanguageUiLogic.shouldBypassTranslation("en", "en", "hello"))
    }

    @Test
    fun shouldBypassTranslation_trueWhenTextBlank() {
        assertTrue(LanguageUiLogic.shouldBypassTranslation("en", "ja", "   "))
    }

    @Test
    fun shouldBypassTranslation_falseForDifferentLanguageAndNonBlankText() {
        assertFalse(LanguageUiLogic.shouldBypassTranslation("en", "ja", "hello"))
    }

    @Test
    fun computeDownloadPresentation_hiddenWhenDownloaded() {
        val state = LanguageUiLogic.computeDownloadPresentation(
            isDownloaded = true,
            isDownloading = false
        )

        assertEquals(LanguageUiLogic.DownloadIconState.HIDDEN, state.iconState)
        assertFalse(state.enabled)
    }

    @Test
    fun computeDownloadPresentation_downloadingWhenInProgress() {
        val state = LanguageUiLogic.computeDownloadPresentation(
            isDownloaded = false,
            isDownloading = true
        )

        assertEquals(LanguageUiLogic.DownloadIconState.DOWNLOADING, state.iconState)
        assertFalse(state.enabled)
    }

    @Test
    fun computeDownloadPresentation_downloadWhenMissingAndIdle() {
        val state = LanguageUiLogic.computeDownloadPresentation(
            isDownloaded = false,
            isDownloading = false
        )

        assertEquals(LanguageUiLogic.DownloadIconState.DOWNLOAD, state.iconState)
        assertTrue(state.enabled)
    }
}
