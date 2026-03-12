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
        private const val COLOR_SWITCH_HYSTERESIS = 36.0
        private const val COLOR_HISTORY_LIMIT = 256
        private const val MAX_COLOR_DISTANCE_BUCKET = 442
    }

    private data class DrawItem(
        val stableKey: String,
        val result: OcrService.DetectionResult,
        val box: android.graphics.Rect,
        val text: String,
        val color: Int = Color.BLACK
    )

    private data class InpaintRegion(
        val box: android.graphics.Rect,
        val cornerPoints: Array<android.graphics.Point>? = null
    )

    private data class RgbColor(
        val r: Double,
        val g: Double,
        val b: Double
    )

    private class ColorAccumulator {
        var sumR = 0.0
        var sumG = 0.0
        var sumB = 0.0
        var count = 0

        fun add(bgr: DoubleArray) {
            if (bgr.size < 3) return
            sumB += bgr[0]
            sumG += bgr[1]
            sumR += bgr[2]
            count++
        }

        fun averageColor(): Int? {
            if (count == 0) return null
            return Color.rgb(
                (sumR / count).toInt().coerceIn(0, 255),
                (sumG / count).toInt().coerceIn(0, 255),
                (sumB / count).toInt().coerceIn(0, 255)
            )
        }
    }

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
    private var cachedInpaintedFrame: Mat? = null
    private var cachedInpaintSignature: String? = null
    private var cachedInpaintWidth = 0
    private var cachedInpaintHeight = 0

    fun replaceText(
        frame: Mat,
        detectionResults: List<OcrService.DetectionResult>,
        translatedTexts: List<String>
    ): Mat {
        if (frame.empty() || frame.cols() <= 0 || frame.rows() <= 0) {
            return Mat()
        }

        var working: Mat? = null
        var originalBgr: Mat? = null

        try {
            working = ensureBgr(frame)
            if (working.empty() || working.cols() <= 0 || working.rows() <= 0) {
                working.release()
                return Mat()
            }
            originalBgr = working.clone()

            val baseItems = collectDrawItems(detectionResults, translatedTexts)
            if (baseItems.isEmpty()) {
                stableTextColors.clear()
                clearInpaintCache()
                return working
            }
            retainOnlyActiveColorKeys(baseItems.mapTo(HashSet()) { it.stableKey })

            val inpaintRegions = collectInpaintRegions(baseItems)
            val inpaintMask = buildInpaintMask(working.cols(), working.rows(), inpaintRegions)
            val inpaintSignature = buildInpaintSignature(inpaintRegions)
            if (Core.countNonZero(inpaintMask) > 0) {
                val cachedFrame = cachedInpaintedFrame
                if (canReuseCachedInpaint(working.cols(), working.rows(), inpaintSignature) &&
                    cachedFrame != null &&
                    !cachedFrame.empty()
                ) {
                    cachedFrame.copyTo(working, inpaintMask)
                } else {
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
                    updateInpaintCache(inpainted, inpaintSignature)
                }
            } else {
                clearInpaintCache()
            }
            inpaintMask.release()

            val drawItems = applyTextColors(baseItems, originalBgr, working)

            Imgproc.cvtColor(working, working, Imgproc.COLOR_BGR2RGB)
            if (working.empty()) {
                return Mat()
            }
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
            originalBgr?.release()
        }

        return if (working != null && !working.empty()) {
            working
        } else {
            working?.release()
            Mat()
        }
    }

    private fun ensureBgr(frame: Mat): Mat {
        if (frame.empty() || frame.cols() <= 0 || frame.rows() <= 0) {
            return Mat()
        }

        val working = Mat()
        when (frame.type()) {
            CvType.CV_8UC3 -> frame.copyTo(working)
            CvType.CV_8UC4 -> Imgproc.cvtColor(frame, working, Imgproc.COLOR_RGBA2BGR)
            CvType.CV_8UC1 -> Imgproc.cvtColor(frame, working, Imgproc.COLOR_GRAY2BGR)
            else -> frame.convertTo(working, CvType.CV_8UC3)
        }
        return working
    }

    fun clearCaches() {
        stableTextColors.clear()
        clearInpaintCache()
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
            val stableRegionBox = result.blockBoundingBox ?: box
            val stableRegionCorners = result.blockCornerPoints ?: result.cornerPoints
            items.add(
                DrawItem(
                    stableKey = "${buildRegionKey(stableRegionBox, stableRegionCorners)}:${text.hashCode()}",
                    result = result,
                    box = box,
                    text = text
                )
            )
        }
        return items
    }

    private fun collectInpaintRegions(drawItems: List<DrawItem>): List<InpaintRegion> {
        val regions = ArrayList<InpaintRegion>(drawItems.size)
        val seen = HashSet<String>(drawItems.size)

        drawItems.forEach { item ->
            val regionBox = item.result.blockBoundingBox ?: item.box
            val regionCorners = item.result.blockCornerPoints ?: item.result.cornerPoints
            val key = buildRegionKey(regionBox, regionCorners)
            if (!seen.add(key)) return@forEach
            regions.add(
                InpaintRegion(
                    box = regionBox,
                    cornerPoints = regionCorners
                )
            )
        }

        return regions
    }

    private fun buildInpaintSignature(regions: List<InpaintRegion>): String {
        return regions.map { region ->
            buildRegionKey(region.box, region.cornerPoints)
        }
            .sorted()
            .joinToString(separator = "|")
    }

    private fun buildRegionKey(
        box: android.graphics.Rect,
        cornerPoints: Array<android.graphics.Point>?
    ): String {
        if (cornerPoints != null && cornerPoints.size >= 3) {
            val pointsKey = cornerPoints.joinToString(separator = ";") { point ->
                "${point.x}_${point.y}"
            }
            return "poly:$pointsKey"
        }
        return "rect:${box.left}_${box.top}_${box.right}_${box.bottom}"
    }

    private fun buildInpaintMask(
        width: Int,
        height: Int,
        regions: List<InpaintRegion>
    ): Mat {
        val mask = Mat.zeros(height, width, CvType.CV_8UC1)
        val white = Scalar(255.0)

        regions.forEach { region ->
            val polygon = toClampedPolygon(region.cornerPoints, width, height)
            if (polygon != null) {
                Imgproc.fillConvexPoly(mask, polygon, white)
                polygon.release()
            } else {
                val left = region.box.left.coerceAtLeast(0)
                val top = region.box.top.coerceAtLeast(0)
                val right = region.box.right.coerceAtMost(width)
                val bottom = region.box.bottom.coerceAtMost(height)
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
            stableTextColors[item.stableKey]?.let { lockedColor ->
                return@map item.copy(color = lockedColor)
            }
            val backgroundColor = estimateBackgroundColor(inpaintedBgr, item.box)
            val rawTextColor = estimateTextColor(originalBgr, item.result, backgroundColor)
            val contrasted = enforceTextContrast(rawTextColor, backgroundColor)
            val finalTextColor = stabilizeTextColor(item.stableKey, contrasted)
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
        candidate: Int
    ): Int {
        val stabilized = stableTextColors[key] ?: candidate
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

    private fun enforceTextContrast(textColor: Int, _backgroundColor: Int): Int {
        return textColor
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

    private fun colorDistance(bgr: DoubleArray, color: Int): Double {
        if (bgr.size < 3) return 0.0
        val dr = bgr[2] - Color.red(color)
        val dg = bgr[1] - Color.green(color)
        val db = bgr[0] - Color.blue(color)
        return kotlin.math.sqrt(dr * dr + dg * dg + db * db)
    }

    private fun sampleStepForArea(area: Int): Int {
        return when {
            area > 40_000 -> 4
            area > 12_000 -> 3
            area > 4_000 -> 2
            else -> 1
        }
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

    private fun toRelativePolygon(
        points: Array<android.graphics.Point>?,
        anchorBox: android.graphics.Rect,
        maxWidth: Int,
        maxHeight: Int
    ): MatOfPoint? {
        if (points == null || points.size < 3) return null

        val maxX = (maxWidth - 1).coerceAtLeast(0).toDouble()
        val maxY = (maxHeight - 1).coerceAtLeast(0).toDouble()
        val clamped = points.map {
            Point(
                (it.x - anchorBox.left).toDouble().coerceIn(0.0, maxX),
                (it.y - anchorBox.top).toDouble().coerceIn(0.0, maxY)
            )
        }

        val unique = clamped.distinctBy { "${it.x.toInt()}_${it.y.toInt()}" }
        if (unique.size < 3) return null

        return MatOfPoint(*unique.toTypedArray())
    }

    private fun buildTextDetectionMask(
        box: android.graphics.Rect,
        result: OcrService.DetectionResult
    ): Mat {
        val x1 = box.left.coerceAtLeast(0)
        val y1 = box.top.coerceAtLeast(0)
        val x2 = box.right.coerceAtLeast(x1 + 1)
        val y2 = box.bottom.coerceAtLeast(y1 + 1)
        val width = x2 - x1
        val height = y2 - y1
        val mask = Mat.zeros(height, width, CvType.CV_8UC1)
        val white = Scalar(255.0)
        var filled = false

        val regions = if (result.textRegions.isNotEmpty()) {
            result.textRegions
        } else {
            listOf(
                OcrService.TextRegion(
                    boundingBox = result.boundingBox,
                    cornerPoints = result.cornerPoints
                )
            )
        }

        regions.forEach { region ->
            val polygon = toRelativePolygon(region.cornerPoints, box, width, height)
            if (polygon != null) {
                Imgproc.fillConvexPoly(mask, polygon, white)
                polygon.release()
                filled = true
                return@forEach
            }

            val regionBox = region.boundingBox ?: return@forEach
            val left = (regionBox.left - box.left).coerceIn(0, width)
            val top = (regionBox.top - box.top).coerceIn(0, height)
            val right = (regionBox.right - box.left).coerceIn(0, width)
            val bottom = (regionBox.bottom - box.top).coerceIn(0, height)
            if (right <= left || bottom <= top) return@forEach

            Imgproc.rectangle(
                mask,
                Point(left.toDouble(), top.toDouble()),
                Point(right.toDouble(), bottom.toDouble()),
                white,
                -1
            )
            filled = true
        }

        if (!filled) {
            Imgproc.rectangle(
                mask,
                Point(0.0, 0.0),
                Point(width.toDouble(), height.toDouble()),
                white,
                -1
            )
        }

        return mask
    }

    private fun extractForegroundTextColor(
        sourceBgr: Mat,
        box: android.graphics.Rect,
        detectionMask: Mat,
        backgroundColor: Int
    ): Int? {
        val width = box.width()
        val height = box.height()
        val sampleStep = sampleStepForArea(width * height)
        val histogram = IntArray(MAX_COLOR_DISTANCE_BUCKET + 1)
        var maskedCount = 0

        for (relY in 0 until height step sampleStep) {
            for (relX in 0 until width step sampleStep) {
                val maskValue = detectionMask.get(relY, relX) ?: continue
                if (maskValue.isEmpty() || maskValue[0] <= 0.0) continue

                val bgr = sourceBgr.get(box.top + relY, box.left + relX) ?: continue
                if (bgr.size < 3) continue

                val bucket = colorDistance(bgr, backgroundColor)
                    .roundToInt()
                    .coerceIn(0, MAX_COLOR_DISTANCE_BUCKET)
                histogram[bucket]++
                maskedCount++
            }
        }

        if (maskedCount == 0) return null

        val threshold = otsuThreshold(histogram, maskedCount)
        val foreground = ColorAccumulator()
        val background = ColorAccumulator()

        for (relY in 0 until height step sampleStep) {
            for (relX in 0 until width step sampleStep) {
                val maskValue = detectionMask.get(relY, relX) ?: continue
                if (maskValue.isEmpty() || maskValue[0] <= 0.0) continue

                val bgr = sourceBgr.get(box.top + relY, box.left + relX) ?: continue
                if (bgr.size < 3) continue

                val bucket = colorDistance(bgr, backgroundColor)
                    .roundToInt()
                    .coerceIn(0, MAX_COLOR_DISTANCE_BUCKET)

                if (bucket > threshold) {
                    foreground.add(bgr)
                } else {
                    background.add(bgr)
                }
            }
        }

        if (foreground.count == 0 || background.count == 0) return null

        val foregroundColor = foreground.averageColor() ?: return null
        val backgroundSampleColor = background.averageColor() ?: return null
        if (colorDistance(foregroundColor, backgroundColor) <=
            colorDistance(backgroundSampleColor, backgroundColor)
        ) {
            return null
        }

        return foregroundColor
    }

    private fun estimateMaskedMinorityTextColor(
        sourceBgr: Mat,
        box: android.graphics.Rect,
        detectionMask: Mat
    ): Int? {
        val width = box.width()
        val height = box.height()
        val sampleStep = sampleStepForArea(width * height)
        var darkR = 0.0
        var darkG = 0.0
        var darkB = 0.0
        var darkCount = 0

        var lightR = 0.0
        var lightG = 0.0
        var lightB = 0.0
        var lightCount = 0

        for (relY in 0 until height step sampleStep) {
            for (relX in 0 until width step sampleStep) {
                val maskValue = detectionMask.get(relY, relX) ?: continue
                if (maskValue.isEmpty() || maskValue[0] <= 0.0) continue

                val bgr = sourceBgr.get(box.top + relY, box.left + relX) ?: continue
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

        if (darkCount == 0 && lightCount == 0) return null
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

    private fun canReuseCachedInpaint(
        width: Int,
        height: Int,
        signature: String
    ): Boolean {
        return cachedInpaintSignature == signature &&
            cachedInpaintWidth == width &&
            cachedInpaintHeight == height
    }

    private fun updateInpaintCache(
        inpainted: Mat,
        signature: String
    ) {
        clearInpaintCache()
        cachedInpaintedFrame = inpainted.clone()
        cachedInpaintSignature = signature
        cachedInpaintWidth = inpainted.cols()
        cachedInpaintHeight = inpainted.rows()
    }

    private fun clearInpaintCache() {
        cachedInpaintedFrame?.release()
        cachedInpaintedFrame = null
        cachedInpaintSignature = null
        cachedInpaintWidth = 0
        cachedInpaintHeight = 0
    }

    private fun otsuThreshold(histogram: IntArray, totalCount: Int): Int {
        if (totalCount <= 0) return 0

        var sum = 0.0
        histogram.forEachIndexed { index, count ->
            sum += index * count.toDouble()
        }

        var sumBackground = 0.0
        var weightBackground = 0
        var maxVariance = Double.NEGATIVE_INFINITY
        var threshold = 0

        histogram.forEachIndexed { index, count ->
            weightBackground += count
            if (weightBackground == 0) return@forEachIndexed

            val weightForeground = totalCount - weightBackground
            if (weightForeground == 0) return@forEachIndexed

            sumBackground += index * count.toDouble()
            val meanBackground = sumBackground / weightBackground
            val meanForeground = (sum - sumBackground) / weightForeground
            val betweenClassVariance =
                weightBackground.toDouble() * weightForeground.toDouble() *
                    (meanBackground - meanForeground) * (meanBackground - meanForeground)

            if (betweenClassVariance > maxVariance) {
                maxVariance = betweenClassVariance
                threshold = index
            }
        }

        return threshold
    }

    private fun estimateTextColor(
        sourceBgr: Mat,
        result: OcrService.DetectionResult,
        backgroundColor: Int
    ): Int {
        val rawBox = result.boundingBox ?: return Color.BLACK
        val box = android.graphics.Rect(
            rawBox.left.coerceAtLeast(0),
            rawBox.top.coerceAtLeast(0),
            rawBox.right.coerceAtMost(sourceBgr.cols()),
            rawBox.bottom.coerceAtMost(sourceBgr.rows())
        )
        val width = box.width()
        val height = box.height()
        if (width <= 1 || height <= 1) return Color.BLACK

        val detectionMask = buildTextDetectionMask(box, result)
        try {
            extractForegroundTextColor(sourceBgr, box, detectionMask, backgroundColor)?.let {
                return it
            }

            estimateMaskedMinorityTextColor(sourceBgr, box, detectionMask)?.let {
                return it
            }
        } finally {
            detectionMask.release()
        }

        return estimateTextColorInBox(sourceBgr, box)
    }

    private fun estimateTextColorInBox(sourceBgr: Mat, box: android.graphics.Rect): Int {
        val x1 = box.left.coerceAtLeast(0)
        val y1 = box.top.coerceAtLeast(0)
        val x2 = box.right.coerceAtMost(sourceBgr.cols())
        val y2 = box.bottom.coerceAtMost(sourceBgr.rows())
        val width = x2 - x1
        val height = y2 - y1
        if (width <= 1 || height <= 1) return Color.BLACK

        val sampleStep = sampleStepForArea(width * height)
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
        val layoutBox = computeLayoutBox(item.result, item.box)
        val visibleLeft = layoutBox.left.coerceAtLeast(0)
        val visibleTop = layoutBox.top.coerceAtLeast(0)
        val visibleRight = layoutBox.right.coerceAtMost(maxWidth)
        val visibleBottom = layoutBox.bottom.coerceAtMost(maxHeight)
        if (visibleRight <= visibleLeft || visibleBottom <= visibleTop) return

        val x1 = layoutBox.left
        val y1 = layoutBox.top
        val x2 = layoutBox.right
        val y2 = layoutBox.bottom
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

    private fun computeLayoutBox(
        result: OcrService.DetectionResult,
        fallbackBox: android.graphics.Rect
    ): android.graphics.Rect {
        val points = result.cornerPoints
        if (points == null || points.isEmpty()) return fallbackBox

        val left = points.minOf { it.x }
        val top = points.minOf { it.y }
        val right = points.maxOf { it.x }
        val bottom = points.maxOf { it.y }
        if (right <= left || bottom <= top) return fallbackBox

        return android.graphics.Rect(left, top, right, bottom)
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
