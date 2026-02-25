package com.example.translatorapp.ocr

import android.graphics.*
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import androidx.core.graphics.createBitmap
import org.opencv.core.Point

class SceneTextReplacer(
    private val typeface: Typeface = Typeface.DEFAULT_BOLD
) {

    companion object {
        private const val TAG = "SceneTextReplacer"
    }

    fun replaceText(
        frame: Mat,
        detectionResults: List<OcrService.DetectionResult>,
        translatedTexts: List<String>
    ): Mat {

        // 1️⃣ Ensure BGR format
        val working = Mat()
        when (frame.type()) {
            CvType.CV_8UC3 -> frame.copyTo(working)
            CvType.CV_8UC4 -> Imgproc.cvtColor(frame, working, Imgproc.COLOR_RGBA2BGR)
            CvType.CV_8UC1 -> Imgproc.cvtColor(frame, working, Imgproc.COLOR_GRAY2BGR)
            else -> frame.convertTo(working, CvType.CV_8UC3)
        }

        try {
            // 2️⃣ Create mask for inpainting
            val mask = Mat.zeros(working.size(), CvType.CV_8UC1)
            detectionResults.forEach { result ->
                val box = result.boundingBox ?: return@forEach
                Imgproc.rectangle(mask,
                    Point(box.left.toDouble(), box.top.toDouble()),
                    Point(box.right.toDouble(), box.bottom.toDouble()),
                    Scalar(255.0),
                    -1
                )
            }

            // 3️⃣ Inpaint original text
            Photo.inpaint(working, mask, working, 3.0, Photo.INPAINT_TELEA)
            mask.release()

            // 4️⃣ Convert BGR → RGB for drawing
            Imgproc.cvtColor(working, working, Imgproc.COLOR_BGR2RGB)

            // 5️⃣ Convert to Bitmap
            val bitmap = createBitmap(working.cols(), working.rows())
            Utils.matToBitmap(working, bitmap)
            val canvas = Canvas(bitmap)

            // 6️⃣ Draw translated text (always black)
            detectionResults.forEachIndexed { i, result ->
                val text = translatedTexts.getOrNull(i) ?: return@forEachIndexed
                val box = result.boundingBox ?: return@forEachIndexed

                val x1 = box.left.coerceAtLeast(0)
                val y1 = box.top.coerceAtLeast(0)
                val x2 = box.right.coerceAtMost(working.cols())
                val y2 = box.bottom.coerceAtMost(working.rows())
                val boxWidth = (x2 - x1).toFloat()
                val boxHeight = (y2 - y1).toFloat()
                if (boxWidth <= 0 || boxHeight <= 0) return@forEachIndexed

                // Create paint
                val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = Color.BLACK       // ✅ always black
                    typeface = typeface
                    style = Paint.Style.FILL
                    isFakeBoldText = true
                }

                // Dynamic font sizing
                var fontSize = boxHeight.coerceAtMost(60f) // max font cap
                paint.textSize = fontSize

                val bounds = android.graphics.Rect()
                paint.getTextBounds(text, 0, text.length, bounds)

                if (bounds.width() > boxWidth) {
                    fontSize *= boxWidth / bounds.width()
                    paint.textSize = fontSize
                    paint.getTextBounds(text, 0, text.length, bounds)
                }

                // Center text in bounding box
                val drawX = x1 + (boxWidth - bounds.width()) / 2f
                val drawY = y1 + (boxHeight + bounds.height()) / 2f

                canvas.drawText(text, drawX, drawY, paint)
            }

            // Convert back to Mat
            Utils.bitmapToMat(bitmap, working)
            bitmap.recycle()
            Imgproc.cvtColor(working, working, Imgproc.COLOR_RGB2BGR)

        } catch (e: Exception) {
            Log.e(TAG, "SceneTextReplacer failed", e)
        }

        return working
    }
}