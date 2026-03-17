package com.example.translatorapp.ocr

import android.graphics.Rect
import kotlin.math.abs
import kotlin.math.max

internal data class SeenDetection(
    val normalizedText: String,
    val boundingBox: Rect
)

internal object OcrRecognizerPolicy {
    fun timeoutForRecognizer(
        isAutoSource: Boolean,
        recognizerIndex: Int,
        primaryTimeoutMs: Long,
        fallbackTimeoutMs: Long
    ): Long {
        return if (!isAutoSource || recognizerIndex == 0) {
            primaryTimeoutMs
        } else {
            fallbackTimeoutMs
        }
    }

    fun shouldStopAfterRecognizer(
        isAutoSource: Boolean,
        recognizerFoundCount: Int
    ): Boolean {
        return !isAutoSource || recognizerFoundCount > 0
    }
}

internal object OcrDuplicatePolicy {
    private const val MIN_OVERLAP_RATIO = 0.72
    private const val MAX_CENTER_DELTA_PX = 20
    private const val MAX_SIZE_DELTA_PX = 24

    fun createSeenDetection(text: String, box: Rect): SeenDetection {
        return SeenDetection(
            normalizedText = normalizeText(text),
            boundingBox = Rect(box)
        )
    }

    fun isDuplicate(
        text: String,
        box: Rect,
        seenDetections: List<SeenDetection>
    ): Boolean {
        val normalizedText = normalizeText(text)
        return seenDetections.any { seen ->
            seen.normalizedText == normalizedText && boxesMatch(seen.boundingBox, box)
        }
    }

    private fun normalizeText(text: String): String {
        return text.trim().lowercase().replace("\\s+".toRegex(), " ")
    }

    private fun boxesMatch(left: Rect, right: Rect): Boolean {
        val overlap = overlapRatio(left, right)
        if (overlap >= MIN_OVERLAP_RATIO) return true

        val leftCenterX = (left.left + left.right) / 2
        val leftCenterY = (left.top + left.bottom) / 2
        val rightCenterX = (right.left + right.right) / 2
        val rightCenterY = (right.top + right.bottom) / 2

        val centerDeltaX = abs(leftCenterX - rightCenterX)
        val centerDeltaY = abs(leftCenterY - rightCenterY)
        val widthDelta = abs(left.width() - right.width())
        val heightDelta = abs(left.height() - right.height())

        return centerDeltaX <= MAX_CENTER_DELTA_PX &&
            centerDeltaY <= MAX_CENTER_DELTA_PX &&
            widthDelta <= MAX_SIZE_DELTA_PX &&
            heightDelta <= MAX_SIZE_DELTA_PX
    }

    private fun overlapRatio(left: Rect, right: Rect): Double {
        val overlapLeft = max(left.left, right.left)
        val overlapTop = max(left.top, right.top)
        val overlapRight = minOf(left.right, right.right)
        val overlapBottom = minOf(left.bottom, right.bottom)
        val overlapWidth = (overlapRight - overlapLeft).coerceAtLeast(0)
        val overlapHeight = (overlapBottom - overlapTop).coerceAtLeast(0)
        val overlapArea = overlapWidth.toDouble() * overlapHeight.toDouble()
        if (overlapArea <= 0.0) return 0.0

        val leftArea = left.width().coerceAtLeast(0).toDouble() * left.height().coerceAtLeast(0)
        val rightArea = right.width().coerceAtLeast(0).toDouble() * right.height().coerceAtLeast(0)
        val referenceArea = max(leftArea, rightArea)
        if (referenceArea <= 0.0) return 0.0

        return overlapArea / referenceArea
    }
}

internal data class OcrMotionDecision(
    val stable: Boolean,
    val lastMotionAtMs: Long,
    val unstableSinceMs: Long,
    val blockOcrUntilMs: Long,
    val invalidateInFlightRequests: Boolean
)

internal object OcrMotionPolicy {
    fun evaluate(
        motion: Double,
        nowAtMs: Long,
        lastMotionAtMs: Long,
        unstableSinceMs: Long,
        lastStableState: Boolean?,
        blockOcrUntilMs: Long,
        motionThreshold: Double,
        activationBlockThreshold: Double,
        hardMotionThreshold: Double,
        stableHoldMs: Long,
        unstableHoldMs: Long,
        motionCooldownMs: Long
    ): OcrMotionDecision {
        var nextLastMotionAtMs = lastMotionAtMs
        var nextUnstableSinceMs = unstableSinceMs
        var nextBlockOcrUntilMs = blockOcrUntilMs
        var invalidateInFlightRequests = false

        if (motion >= motionThreshold) {
            nextLastMotionAtMs = nowAtMs
            if (nextUnstableSinceMs == 0L) {
                nextUnstableSinceMs = nowAtMs
            }
            if (motion >= activationBlockThreshold) {
                nextBlockOcrUntilMs = max(nextBlockOcrUntilMs, nowAtMs + motionCooldownMs)
            }
            if (motion >= hardMotionThreshold) {
                nextBlockOcrUntilMs = max(nextBlockOcrUntilMs, nowAtMs + motionCooldownMs)
                invalidateInFlightRequests = true
            }
        } else if (nextUnstableSinceMs != 0L && nowAtMs - nextLastMotionAtMs >= unstableHoldMs) {
            nextUnstableSinceMs = 0L
        }

        val holdMs = if (lastStableState == true) stableHoldMs else max(stableHoldMs, unstableHoldMs)
        val stable = nextLastMotionAtMs != 0L &&
            nowAtMs >= nextBlockOcrUntilMs &&
            nowAtMs - nextLastMotionAtMs >= holdMs

        return OcrMotionDecision(
            stable = stable,
            lastMotionAtMs = nextLastMotionAtMs,
            unstableSinceMs = nextUnstableSinceMs,
            blockOcrUntilMs = nextBlockOcrUntilMs,
            invalidateInFlightRequests = invalidateInFlightRequests
        )
    }

    fun shouldAcceptResult(
        requestMotionRevision: Long,
        currentMotionRevision: Long,
        requestScanGeneration: Long,
        currentScanGeneration: Long,
        nowAtMs: Long,
        blockOcrUntilMs: Long,
        isStable: Boolean
    ): Boolean {
        return requestMotionRevision == currentMotionRevision &&
            requestScanGeneration == currentScanGeneration &&
            isStable &&
            nowAtMs >= blockOcrUntilMs
    }
}

internal object OcrRefreshPolicy {
    fun shouldRunPeriodicRefresh(
        isArmed: Boolean,
        lastRearmAtMs: Long,
        nowAtMs: Long,
        refreshIntervalMs: Long
    ): Boolean {
        if (isArmed) return false
        if (refreshIntervalMs <= 0L) return false
        if (lastRearmAtMs == 0L) return true
        return nowAtMs - lastRearmAtMs >= refreshIntervalMs
    }
}
