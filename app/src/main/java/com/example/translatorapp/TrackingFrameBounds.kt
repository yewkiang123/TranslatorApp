package com.example.translatorapp

object TrackingFrameBounds {
    fun containsPoint(
        x: Double,
        y: Double,
        maxWidth: Int,
        maxHeight: Int
    ): Boolean {
        if (!x.isFinite() || !y.isFinite()) return false
        if (maxWidth <= 0 || maxHeight <= 0) return false
        return x >= 0.0 &&
            y >= 0.0 &&
            x < maxWidth.toDouble() &&
            y < maxHeight.toDouble()
    }

    fun containsRect(
        left: Int,
        top: Int,
        right: Int,
        bottom: Int,
        maxWidth: Int,
        maxHeight: Int
    ): Boolean {
        if (maxWidth <= 0 || maxHeight <= 0) return false
        if (right <= left || bottom <= top) return false
        return left >= 0 &&
            top >= 0 &&
            right <= maxWidth &&
            bottom <= maxHeight
    }
}
