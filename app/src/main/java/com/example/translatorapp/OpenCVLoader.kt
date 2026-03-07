package com.example.translatorapp

import android.util.Log
import org.opencv.android.OpenCVLoader

object OpenCVInitializer {
    private const val TAG = "OpenCVInitializer"
    @Volatile private var initialized = false
    @Volatile private var attempted = false

    fun init(): Boolean {
        if (initialized) return true
        synchronized(this) {
            if (initialized) return true
            if (attempted) return false
            attempted = true

            initialized = try {
                System.loadLibrary("opencv_java4")
                OpenCVLoader.initDebug()
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load OpenCV native library", e)
                false
            } catch (e: Throwable) {
                Log.e(TAG, "Failed to initialize OpenCV", e)
                false
            }

            return initialized
        }
    }

    fun isReady(): Boolean {
        return initialized
    }
}
