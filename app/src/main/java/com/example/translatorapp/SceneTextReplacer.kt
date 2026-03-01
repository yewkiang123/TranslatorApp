package com.example.translatorapp.ocr

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.util.Log
import androidx.core.graphics.createBitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.roundToInt

class SceneTextReplacer(
    private val typeface: Typeface = Typeface.DEFAULT_BOLD
) {

    companion object {
        private const val TAG = "SceneTextReplacer"
        private const val TELEA_INPAINT_RADIUS = 3.0
        private const val INPAINT_MASK_DILATE_SIZE = 3
        private const val GENERATED_TEXT_SCALE = 0.8f
        private const val ANGLE_SNAP_DEGREES = 1.25f
        private const val MIN_TEXT_BACKGROUND_DISTANCE = 48.0
        private const val COLOR_SWITCH_HYSTERESIS = 36.0
        private const val COLOR_HISTORY_LIMIT = 256
    }

    private data class DrawItem(
        val stableKey: String,
        val result: OcrService.DetectionResult,
        val box: android.graphics.Rect,
        val text: String,
        val color: Int = Color.BLACK
    )

    private data class RgbColor(
        val r: Double,
        val g: Double,
        val b: Double
    )

    private var reusableBitmap: Bitmap? = null
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.BLACK
        this.typeface = this@SceneTextReplacer.typeface
        style = Paint.Style.FILL
        isFakeBoldText = true
        isSubpixelText = true
        isLinearText = true
        hinting = Paint.HINTING_ON
    }
    private val textBounds = android.graphics.Rect()
    private val stableTextColors = LinkedHashMap<String, Int>()

    fun replaceText(
        frame: Mat,
        detectionResults: List<OcrService.DetectionResult>,
        translatedTexts: List<String>
    ): Mat {
        if (frame.empty() || frame.cols() <= 0 || frame.rows() <= 0) {
            return frame
        }

        var working = ensureBgr(frame)
        val originalBgr = working.clone()

        try {
            val baseItems = collectDrawItems(detectionResults, translatedTexts)
            if (baseItems.isEmpty()) {
                stableTextColors.clear()
                return working
            }
            retainOnlyActiveColorKeys(baseItems.mapTo(HashSet()) { it.stableKey })

            val inpaintMask = buildInpaintMask(working.cols(), working.rows(), baseItems)
            if (Core.countNonZero(inpaintMask) > 0) {
                val inpainted = Mat()
                Photo.inpaint(
                    working,
                    inpaintMask,
                    inpainted,
                    TELEA_INPAINT_RADIUS,
                    Photo.INPAINT_TELEA
                )
                working.release()
                working = inpainted
            }
            inpaintMask.release()

            val drawItems = applyTextColors(baseItems, originalBgr, working)

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
        } finally {
            originalBgr.release()
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
        detectionResults: List<OcrService.DetectionResult>,
        translatedTexts: List<String>
    ): List<DrawItem> {
        val items = ArrayList<DrawItem>(detectionResults.size)
        detectionResults.forEachIndexed { index, result ->
            val text = translatedTexts.getOrNull(index)?.trim().orEmpty()
            val box = result.boundingBox ?: return@forEachIndexed
            if (text.isEmpty()) return@forEachIndexed
            items.add(
                DrawItem(
                    stableKey = "$index:${text.hashCode()}",
                    result = result,
                    box = box,
                    text = text
                )
            )
        }
        return items
    }

    private fun buildInpaintMask(
        width: Int,
        height: Int,
        drawItems: List<DrawItem>
    ): Mat {
        val mask = Mat.zeros(height, width, CvType.CV_8UC1)
        val white = Scalar(255.0)

        drawItems.forEach { item ->
            val polygon = toClampedPolygon(item.result.cornerPoints, width, height)
            if (polygon != null) {
                Imgproc.fillConvexPoly(mask, polygon, white)
                polygon.release()
            } else {
                val left = item.box.left.coerceAtLeast(0)
                val top = item.box.top.coerceAtLeast(0)
                val right = item.box.right.coerceAtMost(width)
                val bottom = item.box.bottom.coerceAtMost(height)
                Imgproc.rectangle(
                    mask,
                    Point(left.toDouble(), top.toDouble()),
                    Point(right.toDouble(), bottom.toDouble()),
                    white,
                    -1
                )
            }
        }

        val kernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            Size(INPAINT_MASK_DILATE_SIZE.toDouble(), INPAINT_MASK_DILATE_SIZE.toDouble())
        )
        Imgproc.dilate(mask, mask, kernel)
        kernel.release()
        return mask
    }

    private fun applyTextColors(
        items: List<DrawItem>,
        originalBgr: Mat,
        inpaintedBgr: Mat
    ): List<DrawItem> {
        return items.map { item ->
            val backgroundColor = estimateBackgroundColor(inpaintedBgr, item.box)
            val rawTextColor = estimateTextColor(originalBgr, item.box)
            val contrasted = enforceTextContrast(rawTextColor, backgroundColor)
            val finalTextColor = stabilizeTextColor(item.stableKey, contrasted, backgroundColor)
            item.copy(color = finalTextColor)
        }
    }

    private fun retainOnlyActiveColorKeys(activeKeys: Set<String>) {
        if (stableTextColors.isEmpty()) return
        val iterator = stableTextColors.keys.iterator()
        while (iterator.hasNext()) {
            val key = iterator.next()
            if (!activeKeys.contains(key)) {
                iterator.remove()
            }
        }
    }

    private fun stabilizeTextColor(
        key: String,
        candidate: Int,
        backgroundColor: Int
    ): Int {
        val previous = stableTextColors[key]
        val stabilized = if (previous == null) {
            candidate
        } else {
            val previousDistance = colorDistance(previous, backgroundColor)
            val candidateDistance = colorDistance(candidate, backgroundColor)
            val previousTone = toneBucket(previous)
            val candidateTone = toneBucket(candidate)

            if (previousTone != candidateTone &&
                candidateDistance < previousDistance + COLOR_SWITCH_HYSTERESIS
            ) {
                previous
            } else {
                candidate
            }
        }

        stableTextColors[key] = stabilized
        while (stableTextColors.size > COLOR_HISTORY_LIMIT) {
            val firstKey = stableTextColors.entries.firstOrNull()?.key ?: break
            stableTextColors.remove(firstKey)
        }
        return stabilized
    }

    private fun toneBucket(color: Int): Int {
        val luma = luminance(color)
        return when {
            luma >= 170.0 -> 1
            luma <= 85.0 -> -1
            else -> 0
        }
    }

    private fun enforceTextContrast(textColor: Int, backgroundColor: Int): Int {
        val distance = colorDistance(textColor, backgroundColor)
        if (distance >= MIN_TEXT_BACKGROUND_DISTANCE) return textColor

        val bgLuma = luminance(backgroundColor)
        return if (bgLuma >= 140.0) {
            Color.rgb(20, 20, 20)
        } else {
            Color.rgb(235, 235, 235)
        }
    }

    private fun luminance(color: Int): Double {
        return (0.299 * Color.red(color)) +
            (0.587 * Color.green(color)) +
            (0.114 * Color.blue(color))
    }

    private fun colorDistance(a: Int, b: Int): Double {
        val dr = (Color.red(a) - Color.red(b)).toDouble()
        val dg = (Color.green(a) - Color.green(b)).toDouble()
        val db = (Color.blue(a) - Color.blue(b)).toDouble()
        return kotlin.math.sqrt(dr * dr + dg * dg + db * db)
    }

    private fun estimateBackgroundColor(sourceBgr: Mat, box: android.graphics.Rect): Int {
        val x1 = box.left.coerceAtLeast(0)
        val y1 = box.top.coerceAtLeast(0)
        val x2 = box.right.coerceAtMost(sourceBgr.cols())
        val y2 = box.bottom.coerceAtMost(sourceBgr.rows())
        val width = x2 - x1
        val height = y2 - y1
        if (width <= 1 || height <= 1) return Color.WHITE

        // Use only pixels inside the bounding box:
        // majority cluster (dark/light) is treated as background color.
        val area = width * height
        val sampleStep = when {
            area > 40_000 -> 4
            area > 12_000 -> 3
            area > 4_000 -> 2
            else -> 1
        }

        var darkR = 0.0
        var darkG = 0.0
        var darkB = 0.0
        var darkCount = 0

        var lightR = 0.0
        var lightG = 0.0
        var lightB = 0.0
        var lightCount = 0

        for (y in y1 until y2 step sampleStep) {
            for (x in x1 until x2 step sampleStep) {
                val bgr = sourceBgr.get(y, x) ?: continue
                if (bgr.size < 3) continue

                val b = bgr[0]
                val g = bgr[1]
                val r = bgr[2]
                val lum = 0.114 * b + 0.587 * g + 0.299 * r

                if (lum < 128.0) {
                    darkR += r
                    darkG += g
                    darkB += b
                    darkCount++
                } else {
                    lightR += r
                    lightG += g
                    lightB += b
                    lightCount++
                }
            }
        }

        if (darkCount == 0 && lightCount == 0) {
            return averageColorInRect(sourceBgr, x1, y1, x2, y2)
        }

        if (darkCount == 0) {
            return Color.rgb(
                (lightR / lightCount).toInt().coerceIn(0, 255),
                (lightG / lightCount).toInt().coerceIn(0, 255),
                (lightB / lightCount).toInt().coerceIn(0, 255)
            )
        }

        if (lightCount == 0) {
            return Color.rgb(
                (darkR / darkCount).toInt().coerceIn(0, 255),
                (darkG / darkCount).toInt().coerceIn(0, 255),
                (darkB / darkCount).toInt().coerceIn(0, 255)
            )
        }

        val backgroundFromDark = darkCount >= lightCount
        val selectedR = if (backgroundFromDark) darkR / darkCount else lightR / lightCount
        val selectedG = if (backgroundFromDark) darkG / darkCount else lightG / lightCount
        val selectedB = if (backgroundFromDark) darkB / darkCount else lightB / lightCount

        return Color.rgb(
            selectedR.toInt().coerceIn(0, 255),
            selectedG.toInt().coerceIn(0, 255),
            selectedB.toInt().coerceIn(0, 255)
        )
    }

    private fun averageColorInRect(sourceBgr: Mat, x1: Int, y1: Int, x2: Int, y2: Int): Int {
        val width = x2 - x1
        val height = y2 - y1
        if (width <= 0 || height <= 0) return Color.WHITE

        val area = width * height
        val sampleStep = when {
            area > 40_000 -> 4
            area > 12_000 -> 3
            area > 4_000 -> 2
            else -> 1
        }

        var sumR = 0.0
        var sumG = 0.0
        var sumB = 0.0
        var count = 0

        for (y in y1 until y2 step sampleStep) {
            for (x in x1 until x2 step sampleStep) {
                val bgr = sourceBgr.get(y, x) ?: continue
                if (bgr.size < 3) continue
                sumB += bgr[0]
                sumG += bgr[1]
                sumR += bgr[2]
                count++
            }
        }

        if (count == 0) return Color.WHITE

        return Color.rgb(
            (sumR / count).toInt().coerceIn(0, 255),
            (sumG / count).toInt().coerceIn(0, 255),
            (sumB / count).toInt().coerceIn(0, 255)
        )
    }

    private fun toClampedPolygon(
        points: Array<android.graphics.Point>?,
        maxWidth: Int,
        maxHeight: Int
    ): MatOfPoint? {
        if (points == null || points.size < 3) return null

        val maxX = (maxWidth - 1).coerceAtLeast(0).toDouble()
        val maxY = (maxHeight - 1).coerceAtLeast(0).toDouble()
        val clamped = points.map {
            Point(
                it.x.toDouble().coerceIn(0.0, maxX),
                it.y.toDouble().coerceIn(0.0, maxY)
            )
        }

        val unique = clamped.distinctBy { "${it.x.toInt()}_${it.y.toInt()}" }
        if (unique.size < 3) return null

        return MatOfPoint(*unique.toTypedArray())
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

        var darkR = 0.0
        var darkG = 0.0
        var darkB = 0.0
        var darkCount = 0

        var lightR = 0.0
        var lightG = 0.0
        var lightB = 0.0
        var lightCount = 0

        for (y in y1 until y2 step sampleStep) {
            for (x in x1 until x2 step sampleStep) {
                val bgr = sourceBgr.get(y, x) ?: continue
                if (bgr.size < 3) continue

                val b = bgr[0]
                val g = bgr[1]
                val r = bgr[2]
                val lum = 0.114 * b + 0.587 * g + 0.299 * r

                if (lum < 128.0) {
                    darkR += r
                    darkG += g
                    darkB += b
                    darkCount++
                }
                else {
                    lightR += r
                    lightG += g
                    lightB += b
                    lightCount++
                }
            }
        }

        if (darkCount == 0 && lightCount == 0) return Color.BLACK
        if (darkCount == 0) return Color.rgb(235, 235, 235)
        if (lightCount == 0) return Color.rgb(20, 20, 20)

        val textFromDark = darkCount < lightCount
        val selected = if (textFromDark) {
            RgbColor(r = darkR / darkCount, g = darkG / darkCount, b = darkB / darkCount)
        } else {
            RgbColor(r = lightR / lightCount, g = lightG / lightCount, b = lightB / lightCount)
        }

        return Color.rgb(
            selected.r.toInt().coerceIn(0, 255),
            selected.g.toInt().coerceIn(0, 255),
            selected.b.toInt().coerceIn(0, 255)
        )
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
        val drawX = (x1 - textBounds.left).toFloat()
        val fontMetrics = textPaint.fontMetrics
        val drawY = (centerY - (fontMetrics.ascent + fontMetrics.descent) / 2f).roundToInt().toFloat()

        val angle = computeTextAngle(item.result).let { raw ->
            if (abs(raw) < ANGLE_SNAP_DEGREES) 0f else raw
        }
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
