package com.example.translatorapp

import android.content.Context
import org.opencv.android.OpenCVLoader

object OpenCVInitializer {
    init {
        System.loadLibrary("opencv_java4")
    }

    fun init(context: Context): Boolean {
        return OpenCVLoader.initDebug()
    }
}