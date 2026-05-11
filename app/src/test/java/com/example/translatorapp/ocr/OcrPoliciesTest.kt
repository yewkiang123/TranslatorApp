package com.example.translatorapp.ocr

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OcrPoliciesTest {

    @Test
    fun timeoutForRecognizer_usesPrimaryForNonAutoAndFirstAutoRecognizer() {
        assertEquals(
            700L,
            OcrRecognizerPolicy.timeoutForRecognizer(
                isAutoSource = false,
                recognizerIndex = 2,
                primaryTimeoutMs = 700L,
                fallbackTimeoutMs = 500L
            )
        )

        assertEquals(
            700L,
            OcrRecognizerPolicy.timeoutForRecognizer(
                isAutoSource = true,
                recognizerIndex = 0,
                primaryTimeoutMs = 700L,
                fallbackTimeoutMs = 500L
            )
        )
    }

    @Test
    fun timeoutForRecognizer_usesFallbackForLaterAutoRecognizers() {
        assertEquals(
            500L,
            OcrRecognizerPolicy.timeoutForRecognizer(
                isAutoSource = true,
                recognizerIndex = 1,
                primaryTimeoutMs = 700L,
                fallbackTimeoutMs = 500L
            )
        )
    }

    @Test
    fun shouldStopAfterRecognizer_respectsAutoSourceRules() {
        assertTrue(
            OcrRecognizerPolicy.shouldStopAfterRecognizer(
                isAutoSource = false,
                recognizerFoundCount = 0
            )
        )
        assertFalse(
            OcrRecognizerPolicy.shouldStopAfterRecognizer(
                isAutoSource = true,
                recognizerFoundCount = 0
            )
        )
        assertTrue(
            OcrRecognizerPolicy.shouldStopAfterRecognizer(
                isAutoSource = true,
                recognizerFoundCount = 1
            )
        )
    }

    @Test
    fun evaluate_hardMotionBlocksAndInvalidatesInFlightRequests() {
        val decision = OcrMotionPolicy.evaluate(
            motion = 20.0,
            nowAtMs = 1_000L,
            lastMotionAtMs = 0L,
            unstableSinceMs = 0L,
            lastStableState = false,
            blockOcrUntilMs = 0L,
            motionThreshold = 10.5,
            activationBlockThreshold = 13.0,
            hardMotionThreshold = 19.0,
            stableHoldMs = 700L,
            unstableHoldMs = 1_000L,
            motionCooldownMs = 350L
        )

        assertFalse(decision.stable)
        assertEquals(1_000L, decision.lastMotionAtMs)
        assertEquals(1_000L, decision.unstableSinceMs)
        assertEquals(1_350L, decision.blockOcrUntilMs)
        assertTrue(decision.invalidateInFlightRequests)
    }

    @Test
    fun evaluate_becomesStableAfterHoldAndCooldown() {
        val decision = OcrMotionPolicy.evaluate(
            motion = 0.0,
            nowAtMs = 2_000L,
            lastMotionAtMs = 1_000L,
            unstableSinceMs = 1_000L,
            lastStableState = false,
            blockOcrUntilMs = 0L,
            motionThreshold = 10.5,
            activationBlockThreshold = 13.0,
            hardMotionThreshold = 19.0,
            stableHoldMs = 700L,
            unstableHoldMs = 1_000L,
            motionCooldownMs = 350L
        )

        assertTrue(decision.stable)
        assertEquals(1_000L, decision.lastMotionAtMs)
        assertEquals(0L, decision.unstableSinceMs)
        assertEquals(0L, decision.blockOcrUntilMs)
        assertFalse(decision.invalidateInFlightRequests)
    }

    @Test
    fun shouldAcceptResult_requiresMatchingRevisionGenerationAndStability() {
        assertTrue(
            OcrMotionPolicy.shouldAcceptResult(
                requestMotionRevision = 2L,
                currentMotionRevision = 2L,
                requestScanGeneration = 5L,
                currentScanGeneration = 5L,
                nowAtMs = 8_000L,
                blockOcrUntilMs = 7_900L,
                isStable = true
            )
        )

        assertFalse(
            OcrMotionPolicy.shouldAcceptResult(
                requestMotionRevision = 2L,
                currentMotionRevision = 3L,
                requestScanGeneration = 5L,
                currentScanGeneration = 5L,
                nowAtMs = 8_000L,
                blockOcrUntilMs = 7_900L,
                isStable = true
            )
        )
        assertFalse(
            OcrMotionPolicy.shouldAcceptResult(
                requestMotionRevision = 2L,
                currentMotionRevision = 2L,
                requestScanGeneration = 5L,
                currentScanGeneration = 4L,
                nowAtMs = 8_000L,
                blockOcrUntilMs = 7_900L,
                isStable = true
            )
        )
        assertFalse(
            OcrMotionPolicy.shouldAcceptResult(
                requestMotionRevision = 2L,
                currentMotionRevision = 2L,
                requestScanGeneration = 5L,
                currentScanGeneration = 5L,
                nowAtMs = 7_000L,
                blockOcrUntilMs = 7_900L,
                isStable = true
            )
        )
        assertFalse(
            OcrMotionPolicy.shouldAcceptResult(
                requestMotionRevision = 2L,
                currentMotionRevision = 2L,
                requestScanGeneration = 5L,
                currentScanGeneration = 5L,
                nowAtMs = 8_000L,
                blockOcrUntilMs = 7_900L,
                isStable = false
            )
        )
    }

    @Test
    fun shouldRunPeriodicRefresh_respectsArmedIntervalAndElapsedTime() {
        assertFalse(
            OcrRefreshPolicy.shouldRunPeriodicRefresh(
                isArmed = true,
                lastRearmAtMs = 0L,
                nowAtMs = 1_000L,
                refreshIntervalMs = 500L
            )
        )
        assertFalse(
            OcrRefreshPolicy.shouldRunPeriodicRefresh(
                isArmed = false,
                lastRearmAtMs = 100L,
                nowAtMs = 1_000L,
                refreshIntervalMs = 0L
            )
        )
        assertTrue(
            OcrRefreshPolicy.shouldRunPeriodicRefresh(
                isArmed = false,
                lastRearmAtMs = 0L,
                nowAtMs = 1_000L,
                refreshIntervalMs = 500L
            )
        )
        assertTrue(
            OcrRefreshPolicy.shouldRunPeriodicRefresh(
                isArmed = false,
                lastRearmAtMs = 1_000L,
                nowAtMs = 1_600L,
                refreshIntervalMs = 500L
            )
        )
        assertFalse(
            OcrRefreshPolicy.shouldRunPeriodicRefresh(
                isArmed = false,
                lastRearmAtMs = 1_000L,
                nowAtMs = 1_300L,
                refreshIntervalMs = 500L
            )
        )
    }
}
