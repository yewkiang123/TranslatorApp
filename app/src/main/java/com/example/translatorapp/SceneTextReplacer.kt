package com.example.translatorapp.ocr

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.util.Log
import androidx.core.graphics.createBitmap
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import kotlin.math.atan2
import kotlin.math.sqrt

class SceneTextReplacer(
    private val typeface: Typeface = Typeface.DEFAULT_BOLD
) {

    companion object {
        private const val TAG = "SceneTextReplacer"
        private const val GENERATED_TEXT_SCALE = 0.7f
    }

    private data class DrawItem(
        val result: OcrService.DetectionResult,
        val box: android.graphics.Rect,
        val text: String,
        val color: Int
    )

    private data class RgbColor(
        val r: Double,
        val g: Double,
        val b: Double
    )

    private var reusableMask: Mat? = null
    private var reusableBitmap: Bitmap? = null
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.BLACK
        this.typeface = this@SceneTextReplacer.typeface
        style = Paint.Style.FILL
        isFakeBoldText = true
    }
    private val textBounds = android.graphics.Rect()

    fun replaceText(
        frame: Mat,
        detectionResults: List<OcrService.DetectionResult>,
        translatedTexts: List<String>
    ): Mat {
        if (frame.empty() || frame.cols() <= 0 || frame.rows() <= 0) {
            return frame
        }

        val working = ensureBgr(frame)

        try {
            val drawItems = collectDrawItems(working, detectionResults, translatedTexts)
            if (drawItems.isEmpty()) {
                return working
            }

            val mask = getOrCreateMask(working.cols(), working.rows())
            mask.setTo(Scalar(0.0))
            drawItems.forEach { item ->
                Imgproc.rectangle(
                    mask,
                    Point(item.box.left.toDouble(), item.box.top.toDouble()),
                    Point(item.box.right.toDouble(), item.box.bottom.toDouble()),
                    Scalar(255.0),
                    -1
                )
            }

            Photo.inpaint(working, mask, working, 3.0, Photo.INPAINT_TELEA)

            Imgproc.cvtColor(working, working, Imgproc.COLOR_BGR2RGB)
            val bitmap = getOrCreateBitmap(working.cols(), working.rows())
            Utils.matToBitmap(working, bitmap)
            val canvas = Canvas(bitmap)

            drawItems.forEach { item ->
                drawTranslatedText(canvas, working.cols(), working.rows(), item)
            }

            Utils.bitmapToMat(bitmap, working)
            Imgproc.cvtColor(working, working, Imgproc.COLOR_RGB2BGR)
        } catch (e: Exception) {
            Log.e(TAG, "SceneTextReplacer failed", e)
        }

        return working
    }

    fun drawTextOnly(
        frame: Mat,
        detectionResults: List<OcrService.DetectionResult>,
        translatedTexts: List<String>
    ): Mat {
        if (frame.empty() || frame.cols() <= 0 || frame.rows() <= 0) {
            return frame
        }

        val working = ensureBgr(frame)

        try {
            val drawItems = collectDrawItems(working, detectionResults, translatedTexts)
            if (drawItems.isEmpty()) {
                return working
            }

            Imgproc.cvtColor(working, working, Imgproc.COLOR_BGR2RGB)
            val bitmap = getOrCreateBitmap(working.cols(), working.rows())
            Utils.matToBitmap(working, bitmap)
            val canvas = Canvas(bitmap)

            drawItems.forEach { item ->
                drawTranslatedText(canvas, working.cols(), working.rows(), item)
            }

            Utils.bitmapToMat(bitmap, working)
            Imgproc.cvtColor(working, working, Imgproc.COLOR_RGB2BGR)
        } catch (e: Exception) {
            Log.e(TAG, "drawTextOnly failed", e)
        }

        return working
    }

    private fun ensureBgr(frame: Mat): Mat {
        val working = Mat()
        when (frame.type()) {
            CvType.CV_8UC3 -> frame.copyTo(working)
            CvType.CV_8UC4 -> Imgproc.cvtColor(frame, working, Imgproc.COLOR_RGBA2BGR)
            CvType.CV_8UC1 -> Imgproc.cvtColor(frame, working, Imgproc.COLOR_GRAY2BGR)
            else -> frame.convertTo(working, CvType.CV_8UC3)
        }
        return working
    }

    private fun collectDrawItems(
        sourceBgr: Mat,
        detectionResults: List<OcrService.DetectionResult>,
        translatedTexts: List<String>
    ): List<DrawItem> {
        val items = ArrayList<DrawItem>(detectionResults.size)
        detectionResults.forEachIndexed { index, result ->
            val text = translatedTexts.getOrNull(index)?.trim().orEmpty()
            val box = result.boundingBox ?: return@forEachIndexed
            if (text.isEmpty()) return@forEachIndexed
            val color = estimateTextColor(sourceBgr, box)
            items.add(DrawItem(result, box, text, color))
        }
        return items
    }

    private fun estimateTextColor(sourceBgr: Mat, box: android.graphics.Rect): Int {
        val x1 = box.left.coerceAtLeast(0)
        val y1 = box.top.coerceAtLeast(0)
        val x2 = box.right.coerceAtMost(sourceBgr.cols())
        val y2 = box.bottom.coerceAtMost(sourceBgr.rows())
        val width = x2 - x1
        val height = y2 - y1
        if (width <= 1 || height <= 1) return Color.BLACK

        val area = width * height
        val sampleStep = when {
            area > 40_000 -> 4
            area > 12_000 -> 3
            area > 4_000 -> 2
            else -> 1
        }

        val luminances = ArrayList<Double>()
        for (y in y1 until y2 step sampleStep) {
            for (x in x1 until x2 step sampleStep) {
                val bgr = sourceBgr.get(y, x) ?: continue
                if (bgr.size < 3) continue
                val lum = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
                luminances.add(lum)
            }
        }

        if (luminances.size < 10) return Color.BLACK

        val sortedLuma = luminances.sorted()
        val lowThreshold = sortedLuma[((sortedLuma.size - 1) * 0.20).toInt()]
        val highThreshold = sortedLuma[((sortedLuma.size - 1) * 0.80).toInt()]

        var allR = 0.0
        var allG = 0.0
        var allB = 0.0
        var allCount = 0

        var lowR = 0.0
        var lowG = 0.0
        var lowB = 0.0
        var lowCount = 0

        var highR = 0.0
        var highG = 0.0
        var highB = 0.0
        var highCount = 0

        for (y in y1 until y2 step sampleStep) {
            for (x in x1 until x2 step sampleStep) {
                val bgr = sourceBgr.get(y, x) ?: continue
                if (bgr.size < 3) continue

                val b = bgr[0]
                val g = bgr[1]
                val r = bgr[2]
                val lum = 0.114 * b + 0.587 * g + 0.299 * r

                allR += r
                allG += g
                allB += b
                allCount++

                if (lum <= lowThreshold) {
                    lowR += r
                    lowG += g
                    lowB += b
                    lowCount++
                }
                if (lum >= highThreshold) {
                    highR += r
                    highG += g
                    highB += b
                    highCount++
                }
            }
        }

        if (allCount == 0) return Color.BLACK

        val avg = RgbColor(
            r = allR / allCount,
            g = allG / allCount,
            b = allB / allCount
        )
        val dark = if (lowCount > 0) {
            RgbColor(r = lowR / lowCount, g = lowG / lowCount, b = lowB / lowCount)
        } else {
            avg
        }
        val light = if (highCount > 0) {
            RgbColor(r = highR / highCount, g = highG / highCount, b = highB / highCount)
        } else {
            avg
        }

        val darkDistance = colorDistance(dark, avg)
        val lightDistance = colorDistance(light, avg)
        val selected = if (darkDistance >= lightDistance) dark else light
        val selectedDistance = if (darkDistance >= lightDistance) darkDistance else lightDistance

        if (selectedDistance < 18.0) {
            return if (luminance(avg) > 150.0) {
                Color.rgb(20, 20, 20)
            } else {
                Color.rgb(235, 235, 235)
            }
        }

        return Color.rgb(
            selected.r.toInt().coerceIn(0, 255),
            selected.g.toInt().coerceIn(0, 255),
            selected.b.toInt().coerceIn(0, 255)
        )
    }

    private fun luminance(color: RgbColor): Double {
        return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b
    }

    private fun colorDistance(a: RgbColor, b: RgbColor): Double {
        val dr = a.r - b.r
        val dg = a.g - b.g
        val db = a.b - b.b
        return sqrt(dr * dr + dg * dg + db * db)
    }

    private fun getOrCreateMask(width: Int, height: Int): Mat {
        val existing = reusableMask
        if (existing != null && existing.cols() == width && existing.rows() == height) {
            return existing
        }

        existing?.release()
        val created = Mat.zeros(height, width, CvType.CV_8UC1)
        reusableMask = created
        return created
    }

    private fun getOrCreateBitmap(width: Int, height: Int): Bitmap {
        val existing = reusableBitmap
        if (existing != null && existing.width == width && existing.height == height) {
            return existing
        }

        existing?.recycle()
        val created = createBitmap(width, height)
        reusableBitmap = created
        return created
    }

    private fun drawTranslatedText(
        canvas: Canvas,
        maxWidth: Int,
        maxHeight: Int,
        item: DrawItem
    ) {
        val x1 = item.box.left.coerceAtLeast(0)
        val y1 = item.box.top.coerceAtLeast(0)
        val x2 = item.box.right.coerceAtMost(maxWidth)
        val y2 = item.box.bottom.coerceAtMost(maxHeight)
        val boxWidth = (x2 - x1).toFloat()
        val boxHeight = (y2 - y1).toFloat()
        if (boxWidth <= 0 || boxHeight <= 0) return

        textPaint.color = item.color
        var fontSize = boxHeight.coerceAtMost(60f)
        textPaint.textSize = fontSize
        textPaint.getTextBounds(item.text, 0, item.text.length, textBounds)

        if (textBounds.width() > boxWidth && textBounds.width() > 0) {
            fontSize *= boxWidth / textBounds.width()
            textPaint.textSize = fontSize
            textPaint.getTextBounds(item.text, 0, item.text.length, textBounds)
        }

        fontSize = (fontSize * GENERATED_TEXT_SCALE).coerceAtLeast(1f)
        textPaint.textSize = fontSize
        textPaint.getTextBounds(item.text, 0, item.text.length, textBounds)

        val centerX = x1 + boxWidth / 2f
        val centerY = y1 + boxHeight / 2f
        val drawX = centerX - textPaint.measureText(item.text) / 2f
        val fontMetrics = textPaint.fontMetrics
        val drawY = centerY - (fontMetrics.ascent + fontMetrics.descent) / 2f

        val angle = computeTextAngle(item.result)
        if (angle != 0f) {
            canvas.save()
            canvas.rotate(angle, centerX, centerY)
            canvas.drawText(item.text, drawX, drawY, textPaint)
            canvas.restore()
        } else {
            canvas.drawText(item.text, drawX, drawY, textPaint)
        }
    }

    private fun computeTextAngle(result: OcrService.DetectionResult): Float {
        val points = result.cornerPoints ?: return 0f
        if (points.size < 2) return 0f
        val p0 = points[0]
        val p1 = points[1]
        val dx = (p1.x - p0.x).toFloat()
        val dy = (p1.y - p0.y).toFloat()
        if (dx == 0f && dy == 0f) return 0f
        return Math.toDegrees(atan2(dy, dx).toDouble()).toFloat()
    }
}
