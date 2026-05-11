package com.example.translatorapp

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TrackingFrameBoundsTest {

    @Test
    fun containsPoint_returnsTrueForPointInsideBounds() {
        assertTrue(
            TrackingFrameBounds.containsPoint(
                x = 10.5,
                y = 20.0,
                maxWidth = 200,
                maxHeight = 300
            )
        )
    }

    @Test
    fun containsPoint_returnsFalseForNonFiniteCoordinates() {
        assertFalse(TrackingFrameBounds.containsPoint(Double.NaN, 0.0, 100, 100))
        assertFalse(TrackingFrameBounds.containsPoint(0.0, Double.POSITIVE_INFINITY, 100, 100))
    }

    @Test
    fun containsPoint_returnsFalseForOutOfBoundsCoordinates() {
        assertFalse(TrackingFrameBounds.containsPoint(-0.1, 10.0, 100, 100))
        assertFalse(TrackingFrameBounds.containsPoint(10.0, -0.1, 100, 100))
        assertFalse(TrackingFrameBounds.containsPoint(100.0, 10.0, 100, 100))
        assertFalse(TrackingFrameBounds.containsPoint(10.0, 100.0, 100, 100))
    }

    @Test
    fun containsRect_returnsTrueWhenRectIsFullyInsideBounds() {
        assertTrue(
            TrackingFrameBounds.containsRect(
                left = 1,
                top = 2,
                right = 99,
                bottom = 88,
                maxWidth = 100,
                maxHeight = 100
            )
        )
    }

    @Test
    fun containsRect_returnsFalseForInvalidOrOutOfBoundsRects() {
        assertFalse(TrackingFrameBounds.containsRect(10, 10, 10, 20, 100, 100))
        assertFalse(TrackingFrameBounds.containsRect(10, 10, 20, 10, 100, 100))
        assertFalse(TrackingFrameBounds.containsRect(-1, 0, 20, 20, 100, 100))
        assertFalse(TrackingFrameBounds.containsRect(0, 0, 101, 20, 100, 100))
        assertFalse(TrackingFrameBounds.containsRect(0, 0, 20, 101, 100, 100))
        assertFalse(TrackingFrameBounds.containsRect(0, 0, 20, 20, 0, 100))
    }
}
