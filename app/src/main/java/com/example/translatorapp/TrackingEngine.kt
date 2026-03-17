package com.example.translatorapp

import com.example.translatorapp.ocr.OcrService
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.roundToInt
import kotlin.math.sin

internal data class TrackingTransform(
    val m00: Double = 1.0,
    val m01: Double = 0.0,
    val m02: Double = 0.0,
    val m10: Double = 0.0,
    val m11: Double = 1.0,
    val m12: Double = 0.0
) {
    val dx: Double
        get() = m02

    val dy: Double
        get() = m12

    val scale: Double
        get() = hypot(m00, m10)
}

internal data class TrackingSnapshot(
    val referenceResults: List<OcrService.DetectionResult>,
    val baseResults: List<OcrService.DetectionResult>,
    val translations: List<String>,
    val anchorGray: Mat,
    val accumulatedTransform: TrackingTransform
)

internal data class TrackingMotionStats(
    val prevCentroid: Point,
    val nextCentroid: Point,
    val scaleHint: Double?
)

internal data class TrackingConfig(
    val minTrackShiftPx: Double,
    val maxTrackStepPx: Double,
    val minScaleDelta: Double,
    val minTrackScale: Double,
    val maxTrackScale: Double,
    val trackingFeatureCount: Int,
    val trackingFeatureQualityLevel: Double,
    val trackingFeatureMinDistance: Double,
    val minTrackingRoiSizePx: Int,
    val minTrackingTargetTextSizePx: Double,
    val maxTrackingUpscale: Double,
    val trackingFlowWindowSize: Size,
    val trackingFlowMaxLevel: Int,
    val minTrackingValidPoints: Int,
    val minTrackingTranslationPoints: Int,
    val minTrackingAffineInliers: Int,
    val minTrackingRegionSizePx: Int,
    val minTrackingScaleRadiusPx: Double,
    val maxTrackRotationDegrees: Double,
    val trackingFlowCriteria: TermCriteria
)

internal class TrackingEngine(
    private val config: TrackingConfig
) {
    fun toGray(src: Mat): Mat {
        if (src.empty() || src.cols() <= 0 || src.rows() <= 0) {
            return Mat()
        }
        val gray = Mat()
        return try {
            when (src.channels()) {
                4 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
                3 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
                1 -> src.copyTo(gray)
                else -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            }
            gray
        } catch (e: Exception) {
            gray.release()
            Mat()
        }
    }

    fun estimateTrackingTransform(
        prev: Mat?,
        current: Mat,
        referenceResults: List<OcrService.DetectionResult>,
        referencePivot: Point
    ): TrackingTransform? {
        if (prev == null || prev.empty() || current.empty()) return null

        val trackingRegion = computeTrackingRegion(prev.cols(), prev.rows(), referenceResults)
        val trackingScale = computeTrackingUpscale(referenceResults)
        val prevTracking = createTrackingInput(prev, trackingRegion, trackingScale)
        val currentTracking = createTrackingInput(current, trackingRegion, trackingScale)
        val points = MatOfPoint()
        return try {
            if (prevTracking.empty() || currentTracking.empty()) {
                return null
            }

            val trackingMask = buildTrackingMask(
                prevTracking.cols(),
                prevTracking.rows(),
                referenceResults,
                trackingRegion,
                trackingScale
            )
            try {
                if (trackingMask != null) {
                    Imgproc.goodFeaturesToTrack(
                        prevTracking,
                        points,
                        config.trackingFeatureCount,
                        config.trackingFeatureQualityLevel,
                        config.trackingFeatureMinDistance,
                        trackingMask
                    )
                } else {
                    Imgproc.goodFeaturesToTrack(
                        prevTracking,
                        points,
                        config.trackingFeatureCount,
                        config.trackingFeatureQualityLevel,
                        config.trackingFeatureMinDistance
                    )
                }
                if (points.empty() && trackingMask != null) {
                    Imgproc.goodFeaturesToTrack(
                        prevTracking,
                        points,
                        config.trackingFeatureCount,
                        config.trackingFeatureQualityLevel,
                        config.trackingFeatureMinDistance
                    )
                }
            } finally {
                trackingMask?.release()
            }

            if (points.empty()) {
                return estimateGlobalTranslationFallback(prev, current, referencePivot)
            }

            val prevPts = MatOfPoint2f(*points.toArray())
            val nextPts = MatOfPoint2f()
            val status = MatOfByte()
            val err = MatOfFloat()
            try {
                Video.calcOpticalFlowPyrLK(
                    prevTracking,
                    currentTracking,
                    prevPts,
                    nextPts,
                    status,
                    err,
                    config.trackingFlowWindowSize,
                    config.trackingFlowMaxLevel,
                    config.trackingFlowCriteria
                )

                val prevArr = prevPts.toArray()
                val nextArr = nextPts.toArray()
                val statusArr = status.toArray()
                val prevValid = ArrayList<Point>(statusArr.size)
                val nextValid = ArrayList<Point>(statusArr.size)

                for (i in statusArr.indices) {
                    if (statusArr[i].toInt() != 1) continue
                    val from = trackingPointToFrame(prevArr[i], trackingRegion, trackingScale)
                    val to = trackingPointToFrame(nextArr[i], trackingRegion, trackingScale)
                    if (!from.x.isFinite() || !from.y.isFinite() || !to.x.isFinite() || !to.y.isFinite()) continue
                    prevValid.add(from)
                    nextValid.add(to)
                }

                if (prevValid.size < config.minTrackingTranslationPoints) {
                    return estimateGlobalTranslationFallback(prev, current, referencePivot)
                }

                val motionStats = computeTrackingMotionStats(prevValid, nextValid)
                if (prevValid.size < config.minTrackingValidPoints) {
                    return estimateTranslationOnlyTransform(
                        prevPoints = prevValid,
                        nextPoints = nextValid,
                        referencePivot = referencePivot,
                        motionStats = motionStats
                    )
                }

                val prevInliers = MatOfPoint2f(*prevValid.toTypedArray())
                val nextInliers = MatOfPoint2f(*nextValid.toTypedArray())
                val inlierMask = Mat()
                try {
                    val affine = Calib3d.estimateAffinePartial2D(
                        prevInliers,
                        nextInliers,
                        inlierMask,
                        Calib3d.RANSAC,
                        3.0,
                        2000L,
                        0.99,
                        10L
                    )

                    val inlierCount = if (!inlierMask.empty()) {
                        Core.countNonZero(inlierMask)
                    } else {
                        prevValid.size
                    }

                    if (affine.empty() || affine.rows() < 2 || affine.cols() < 3) {
                        affine.release()
                        return estimateTranslationOnlyTransform(
                            prevPoints = prevValid,
                            nextPoints = nextValid,
                            referencePivot = referencePivot,
                            motionStats = motionStats
                        ) ?: estimateGlobalTranslationFallback(prev, current, referencePivot)
                    }
                    if (inlierCount < config.minTrackingAffineInliers) {
                        affine.release()
                        return estimateTranslationOnlyTransform(
                            prevPoints = prevValid,
                            nextPoints = nextValid,
                            referencePivot = referencePivot,
                            motionStats = motionStats
                        ) ?: estimateGlobalTranslationFallback(prev, current, referencePivot)
                    }

                    val rawTransform = TrackingTransform(
                        m00 = affine.readDouble(0, 0) ?: 1.0,
                        m01 = affine.readDouble(0, 1) ?: 0.0,
                        m02 = affine.readDouble(0, 2) ?: 0.0,
                        m10 = affine.readDouble(1, 0) ?: 0.0,
                        m11 = affine.readDouble(1, 1) ?: 1.0,
                        m12 = affine.readDouble(1, 2) ?: 0.0
                    )
                    affine.release()

                    sanitizeTrackingTransform(
                        raw = rawTransform,
                        referencePivot = referencePivot,
                        motionStats = motionStats
                    ) ?: estimateGlobalTranslationFallback(prev, current, referencePivot)
                } finally {
                    prevInliers.release()
                    nextInliers.release()
                    inlierMask.release()
                }
            } finally {
                prevPts.release()
                nextPts.release()
                status.release()
                err.release()
            }
        } catch (_: Exception) {
            null
        } finally {
            prevTracking.release()
            currentTracking.release()
            points.release()
        }
    }

    fun computeTrackingPivot(
        referenceResults: List<OcrService.DetectionResult>,
        width: Int,
        height: Int
    ): Point {
        val boxes = referenceResults.mapNotNull { it.blockBoundingBox ?: it.boundingBox }
        if (boxes.isEmpty()) {
            return Point(width / 2.0, height / 2.0)
        }

        val left = boxes.minOf { it.left }.coerceIn(0, width - 1)
        val top = boxes.minOf { it.top }.coerceIn(0, height - 1)
        val right = boxes.maxOf { it.right }.coerceIn(1, width)
        val bottom = boxes.maxOf { it.bottom }.coerceIn(1, height)
        return Point((left + right) / 2.0, (top + bottom) / 2.0)
    }

    fun measureTransformMotion(transform: TrackingTransform, pivot: Point): Double {
        val mappedPivot = transformPoint(pivot.x, pivot.y, transform)
        if (!mappedPivot.x.isFinite() || !mappedPivot.y.isFinite()) {
            return Double.POSITIVE_INFINITY
        }
        return hypot(mappedPivot.x - pivot.x, mappedPivot.y - pivot.y)
    }

    fun composeTransforms(
        step: TrackingTransform,
        accumulated: TrackingTransform
    ): TrackingTransform {
        return TrackingTransform(
            m00 = step.m00 * accumulated.m00 + step.m01 * accumulated.m10,
            m01 = step.m00 * accumulated.m01 + step.m01 * accumulated.m11,
            m02 = step.m00 * accumulated.m02 + step.m01 * accumulated.m12 + step.m02,
            m10 = step.m10 * accumulated.m00 + step.m11 * accumulated.m10,
            m11 = step.m10 * accumulated.m01 + step.m11 * accumulated.m11,
            m12 = step.m10 * accumulated.m02 + step.m11 * accumulated.m12 + step.m12
        )
    }

    fun applyTrackingTransform(
        results: List<OcrService.DetectionResult>,
        transform: TrackingTransform,
        maxW: Int,
        maxH: Int
    ): List<OcrService.DetectionResult> {
        return results.map { result ->
            val shiftedCorners = shiftPoints(result.cornerPoints, transform, maxW, maxH)
            val shiftedBox = shiftedCorners?.let { boundsFromPoints(it, maxW, maxH) }
                ?: result.boundingBox?.let { shiftRect(it, transform, maxW, maxH) }

            val shiftedBlockBox = (result.blockBoundingBox ?: result.boundingBox)?.let {
                shiftRectKeepingSize(it, transform, maxW, maxH)
            }
            val shiftedBlockCorners = shiftedBlockBox?.let { rectToPoints(it) }

            val shiftedTextRegions = result.textRegions.mapNotNull { region ->
                val shiftedRegionCorners = shiftPoints(region.cornerPoints, transform, maxW, maxH)
                val shiftedRegionBox = shiftedRegionCorners?.let { boundsFromPoints(it, maxW, maxH) }
                    ?: region.boundingBox?.let { shiftRect(it, transform, maxW, maxH) }

                if (shiftedRegionBox == null && shiftedRegionCorners == null) {
                    null
                } else {
                    region.copy(
                        boundingBox = shiftedRegionBox,
                        cornerPoints = shiftedRegionCorners
                    )
                }
            }

            result.copy(
                boundingBox = shiftedBox,
                cornerPoints = shiftedCorners,
                blockBoundingBox = shiftedBlockBox,
                blockCornerPoints = shiftedBlockCorners,
                textRegions = shiftedTextRegions
            )
        }
    }

    private fun estimateGlobalTranslationFallback(
        prev: Mat,
        current: Mat,
        referencePivot: Point
    ): TrackingTransform? {
        if (prev.empty() || current.empty()) return null
        val points = MatOfPoint()
        try {
            try {
                Imgproc.goodFeaturesToTrack(
                    prev,
                    points,
                    maxOf(config.trackingFeatureCount / 2, 120),
                    config.trackingFeatureQualityLevel,
                    config.trackingFeatureMinDistance
                )
                if (points.empty()) {
                    return null
                }

                val prevPts = MatOfPoint2f(*points.toArray())
                val nextPts = MatOfPoint2f()
                val status = MatOfByte()
                val err = MatOfFloat()

                try {
                    Video.calcOpticalFlowPyrLK(
                        prev,
                        current,
                        prevPts,
                        nextPts,
                        status,
                        err,
                        Size(51.0, 51.0),
                        config.trackingFlowMaxLevel + 1,
                        config.trackingFlowCriteria
                    )

                    val prevArr = prevPts.toArray()
                    val nextArr = nextPts.toArray()
                    val statusArr = status.toArray()

                    val prevValid = ArrayList<Point>(statusArr.size)
                    val nextValid = ArrayList<Point>(statusArr.size)
                    for (i in statusArr.indices) {
                        if (statusArr[i].toInt() != 1) continue
                        val from = prevArr[i]
                        val to = nextArr[i]
                        if (!from.x.isFinite() || !from.y.isFinite() || !to.x.isFinite() || !to.y.isFinite()) continue
                        prevValid.add(from)
                        nextValid.add(to)
                    }

                    if (prevValid.size < config.minTrackingTranslationPoints) {
                        return null
                    }

                    val motionStats = computeTrackingMotionStats(prevValid, nextValid)
                    return estimateTranslationOnlyTransform(
                        prevPoints = prevValid,
                        nextPoints = nextValid,
                        referencePivot = referencePivot,
                        motionStats = motionStats
                    )
                } finally {
                    prevPts.release()
                    nextPts.release()
                    status.release()
                    err.release()
                }
            } catch (_: Exception) {
                return null
            }
        } finally {
            points.release()
        }
    }

    private fun buildTrackingMask(
        width: Int,
        height: Int,
        referenceResults: List<OcrService.DetectionResult>,
        trackingRegion: android.graphics.Rect,
        trackingScale: Double
    ): Mat? {
        if (width <= 0 || height <= 0 || referenceResults.isEmpty()) return null

        val mask = Mat.zeros(height, width, CvType.CV_8UC1)
        val fill = Scalar(255.0)
        var regions = 0

        referenceResults.forEach { result ->
            val box = result.blockBoundingBox ?: result.boundingBox ?: return@forEach
            val centerX = ((box.left + box.right) / 2.0 - trackingRegion.left) * trackingScale
            val centerY = ((box.top + box.bottom) / 2.0 - trackingRegion.top) * trackingScale
            val halfWidth = maxOf(
                (box.width() * 0.5 + box.width() * 0.5) * trackingScale,
                config.minTrackingRegionSizePx * trackingScale / 2.0
            )
            val halfHeight = maxOf(
                (box.height() * 0.5 + box.height() * 0.7) * trackingScale,
                config.minTrackingRegionSizePx * trackingScale / 2.0
            )
            val left = floor(centerX - halfWidth).toInt().coerceIn(0, width - 1)
            val top = floor(centerY - halfHeight).toInt().coerceIn(0, height - 1)
            val right = ceil(centerX + halfWidth).toInt().coerceIn(1, width)
            val bottom = ceil(centerY + halfHeight).toInt().coerceIn(1, height)
            if (right <= left || bottom <= top) return@forEach

            Imgproc.rectangle(
                mask,
                Point(left.toDouble(), top.toDouble()),
                Point(right.toDouble(), bottom.toDouble()),
                fill,
                -1
            )
            regions++
        }

        if (regions == 0) {
            mask.release()
            return null
        }

        return mask
    }

    private fun computeTrackingRegion(
        width: Int,
        height: Int,
        referenceResults: List<OcrService.DetectionResult>
    ): android.graphics.Rect {
        val boxes = referenceResults.mapNotNull { it.blockBoundingBox ?: it.boundingBox }
        if (boxes.isEmpty()) {
            return android.graphics.Rect(0, 0, width, height)
        }

        val contentLeft = boxes.minOf { it.left }.coerceIn(0, width - 1)
        val contentTop = boxes.minOf { it.top }.coerceIn(0, height - 1)
        val contentRight = boxes.maxOf { it.right }.coerceIn(1, width)
        val contentBottom = boxes.maxOf { it.bottom }.coerceIn(1, height)
        val contentWidth = (contentRight - contentLeft).coerceAtLeast(1)
        val contentHeight = (contentBottom - contentTop).coerceAtLeast(1)

        val padX = maxOf(contentWidth * 1.8, 84.0)
        val padY = maxOf(contentHeight * 1.9, 84.0)
        val targetWidth = maxOf(contentWidth + padX * 2.0, config.minTrackingRoiSizePx.toDouble())
        val targetHeight = maxOf(contentHeight + padY * 2.0, config.minTrackingRoiSizePx.toDouble())
        val centerX = (contentLeft + contentRight) / 2.0
        val centerY = (contentTop + contentBottom) / 2.0

        var left = floor(centerX - targetWidth / 2.0).toInt()
        var top = floor(centerY - targetHeight / 2.0).toInt()
        var right = ceil(centerX + targetWidth / 2.0).toInt()
        var bottom = ceil(centerY + targetHeight / 2.0).toInt()

        if (left < 0) {
            right -= left
            left = 0
        }
        if (top < 0) {
            bottom -= top
            top = 0
        }
        if (right > width) {
            val overflow = right - width
            left -= overflow
            right = width
        }
        if (bottom > height) {
            val overflow = bottom - height
            top -= overflow
            bottom = height
        }

        left = left.coerceIn(0, width - 1)
        top = top.coerceIn(0, height - 1)
        right = right.coerceIn(left + 1, width)
        bottom = bottom.coerceIn(top + 1, height)
        return android.graphics.Rect(left, top, right, bottom)
    }

    private fun computeTrackingUpscale(
        referenceResults: List<OcrService.DetectionResult>
    ): Double {
        val boxes = referenceResults.mapNotNull { it.blockBoundingBox ?: it.boundingBox }
        if (boxes.isEmpty()) return 1.0

        val contentLeft = boxes.minOf { it.left }
        val contentTop = boxes.minOf { it.top }
        val contentRight = boxes.maxOf { it.right }
        val contentBottom = boxes.maxOf { it.bottom }
        val contentWidth = (contentRight - contentLeft).coerceAtLeast(1)
        val contentHeight = (contentBottom - contentTop).coerceAtLeast(1)
        val contentMinDimension = minOf(contentWidth, contentHeight).toDouble()
        if (contentMinDimension <= 0.0) return 1.0

        return (config.minTrackingTargetTextSizePx / contentMinDimension)
            .coerceIn(1.0, config.maxTrackingUpscale)
    }

    private fun createTrackingInput(
        source: Mat,
        trackingRegion: android.graphics.Rect,
        trackingScale: Double
    ): Mat {
        val roi = Mat(
            source,
            org.opencv.core.Rect(
                trackingRegion.left,
                trackingRegion.top,
                trackingRegion.width(),
                trackingRegion.height()
            )
        )
        return try {
            if (trackingScale > 1.01) {
                val resized = Mat()
                Imgproc.resize(
                    roi,
                    resized,
                    Size(),
                    trackingScale,
                    trackingScale,
                    Imgproc.INTER_CUBIC
                )
                resized
            } else {
                roi.clone()
            }
        } finally {
            roi.release()
        }
    }

    private fun trackingPointToFrame(
        point: Point,
        trackingRegion: android.graphics.Rect,
        trackingScale: Double
    ): Point {
        return Point(
            trackingRegion.left + point.x / trackingScale,
            trackingRegion.top + point.y / trackingScale
        )
    }

    private fun estimateTranslationOnlyTransform(
        prevPoints: List<Point>,
        nextPoints: List<Point>,
        referencePivot: Point,
        motionStats: TrackingMotionStats
    ): TrackingTransform? {
        if (prevPoints.size != nextPoints.size || prevPoints.size < config.minTrackingTranslationPoints) {
            return null
        }

        val dxSamples = ArrayList<Double>(prevPoints.size)
        val dySamples = ArrayList<Double>(prevPoints.size)
        for (i in prevPoints.indices) {
            dxSamples.add(nextPoints[i].x - prevPoints[i].x)
            dySamples.add(nextPoints[i].y - prevPoints[i].y)
        }

        val medianDx = median(dxSamples) ?: return null
        val medianDy = median(dySamples) ?: return null
        val rawTransform = TrackingTransform(
            m02 = medianDx,
            m12 = medianDy
        )
        return sanitizeTrackingTransform(
            raw = rawTransform,
            referencePivot = referencePivot,
            motionStats = motionStats.copy(scaleHint = 1.0)
        )
    }

    private fun sanitizeTrackingTransform(
        raw: TrackingTransform,
        referencePivot: Point,
        motionStats: TrackingMotionStats
    ): TrackingTransform? {
        val rawScale = raw.scale
        val scale = sanitizeScale(
            motionStats.scaleHint
                ?.takeIf { it.isFinite() && it > 0.0 }
                ?: rawScale
        )
        val rotation = sanitizeRotation(atan2(raw.m10, raw.m00))
        val cosValue = cos(rotation)
        val sinValue = sin(rotation)
        val m00 = scale * cosValue
        val m01 = -scale * sinValue
        val m10 = scale * sinValue
        val m11 = scale * cosValue

        val prevCentroid = motionStats.prevCentroid
        val nextCentroid = motionStats.nextCentroid
        val rawTargetPivotX = nextCentroid.x -
            (m00 * (prevCentroid.x - referencePivot.x) + m01 * (prevCentroid.y - referencePivot.y))
        val rawTargetPivotY = nextCentroid.y -
            (m10 * (prevCentroid.x - referencePivot.x) + m11 * (prevCentroid.y - referencePivot.y))
        if (!rawTargetPivotX.isFinite() || !rawTargetPivotY.isFinite()) return null

        val pivotDx = sanitizeShift(rawTargetPivotX - referencePivot.x)
        val pivotDy = sanitizeShift(rawTargetPivotY - referencePivot.y)
        val targetPivotX = referencePivot.x + pivotDx
        val targetPivotY = referencePivot.y + pivotDy
        val m02 = targetPivotX - (m00 * referencePivot.x + m01 * referencePivot.y)
        val m12 = targetPivotY - (m10 * referencePivot.x + m11 * referencePivot.y)

        return TrackingTransform(
            m00 = m00,
            m01 = m01,
            m02 = m02,
            m10 = m10,
            m11 = m11,
            m12 = m12
        )
    }

    private fun sanitizeShift(delta: Double): Double {
        if (!delta.isFinite()) return 0.0
        if (abs(delta) < config.minTrackShiftPx) return 0.0
        return delta.coerceIn(-config.maxTrackStepPx, config.maxTrackStepPx)
    }

    private fun sanitizeScale(scale: Double): Double {
        if (!scale.isFinite() || scale <= 0.0) return 1.0
        val clamped = scale.coerceIn(config.minTrackScale, config.maxTrackScale)
        if (abs(clamped - 1.0) < config.minScaleDelta) return 1.0
        return clamped
    }

    private fun sanitizeRotation(rotation: Double): Double {
        if (!rotation.isFinite()) return 0.0
        val maxRotation = Math.toRadians(config.maxTrackRotationDegrees)
        return rotation.coerceIn(-maxRotation, maxRotation)
    }

    private fun computeTrackingMotionStats(
        prevPoints: List<Point>,
        nextPoints: List<Point>
    ): TrackingMotionStats {
        val prevCentroid = Point(
            prevPoints.sumOf { it.x } / prevPoints.size,
            prevPoints.sumOf { it.y } / prevPoints.size
        )
        val nextCentroid = Point(
            nextPoints.sumOf { it.x } / nextPoints.size,
            nextPoints.sumOf { it.y } / nextPoints.size
        )

        val scaleSamples = ArrayList<Double>(prevPoints.size)
        for (i in prevPoints.indices) {
            val prevRadius = hypot(
                prevPoints[i].x - prevCentroid.x,
                prevPoints[i].y - prevCentroid.y
            )
            val nextRadius = hypot(
                nextPoints[i].x - nextCentroid.x,
                nextPoints[i].y - nextCentroid.y
            )
            if (prevRadius < config.minTrackingScaleRadiusPx) continue
            if (!nextRadius.isFinite() || nextRadius <= 0.0) continue
            scaleSamples.add(nextRadius / prevRadius)
        }

        return TrackingMotionStats(
            prevCentroid = prevCentroid,
            nextCentroid = nextCentroid,
            scaleHint = median(scaleSamples)
        )
    }

    private fun median(values: List<Double>): Double? {
        if (values.isEmpty()) return null
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle]
        }
    }

    private fun shiftRectKeepingSize(
        rect: android.graphics.Rect,
        transform: TrackingTransform,
        maxW: Int,
        maxH: Int
    ): android.graphics.Rect? {
        val cx = (rect.left + rect.right) * 0.5
        val cy = (rect.top + rect.bottom) * 0.5
        val transformedCenter = transformPoint(cx, cy, transform)
        val halfWidth = rect.width() / 2.0
        val halfHeight = rect.height() / 2.0

        val shiftedLeft = floor(transformedCenter.x - halfWidth).toInt()
        val shiftedTop = floor(transformedCenter.y - halfHeight).toInt()
        val shiftedRight = ceil(transformedCenter.x + halfWidth).toInt()
        val shiftedBottom = ceil(transformedCenter.y + halfHeight).toInt()

        if (!TrackingFrameBounds.containsRect(
                left = shiftedLeft,
                top = shiftedTop,
                right = shiftedRight,
                bottom = shiftedBottom,
                maxWidth = maxW,
                maxHeight = maxH
            )
        ) {
            return null
        }

        return android.graphics.Rect(shiftedLeft, shiftedTop, shiftedRight, shiftedBottom)
    }

    private fun shiftRect(
        rect: android.graphics.Rect,
        transform: TrackingTransform,
        maxW: Int,
        maxH: Int
    ): android.graphics.Rect? {
        val corners = arrayOf(
            transformPoint(rect.left.toDouble(), rect.top.toDouble(), transform),
            transformPoint(rect.right.toDouble(), rect.top.toDouble(), transform),
            transformPoint(rect.left.toDouble(), rect.bottom.toDouble(), transform),
            transformPoint(rect.right.toDouble(), rect.bottom.toDouble(), transform)
        )
        val shiftedLeft = floor(corners.minOf { it.x }).toInt()
        val shiftedTop = floor(corners.minOf { it.y }).toInt()
        val shiftedRight = ceil(corners.maxOf { it.x }).toInt()
        val shiftedBottom = ceil(corners.maxOf { it.y }).toInt()

        if (!TrackingFrameBounds.containsRect(
                left = shiftedLeft,
                top = shiftedTop,
                right = shiftedRight,
                bottom = shiftedBottom,
                maxWidth = maxW,
                maxHeight = maxH
            )
        ) {
            return null
        }

        return android.graphics.Rect(shiftedLeft, shiftedTop, shiftedRight, shiftedBottom)
    }

    private fun shiftPoints(
        points: Array<android.graphics.Point>?,
        transform: TrackingTransform,
        maxW: Int,
        maxH: Int
    ): Array<android.graphics.Point>? {
        if (points == null) return null
        return points.map { p ->
            val transformed = transformPoint(p.x.toDouble(), p.y.toDouble(), transform)
            if (!TrackingFrameBounds.containsPoint(
                    x = transformed.x,
                    y = transformed.y,
                    maxWidth = maxW,
                    maxHeight = maxH
                )
            ) {
                return null
            }
            val x = transformed.x.roundToInt()
            val y = transformed.y.roundToInt()
            android.graphics.Point(x, y)
        }.toTypedArray()
    }

    private fun boundsFromPoints(
        points: Array<android.graphics.Point>?,
        maxW: Int,
        maxH: Int
    ): android.graphics.Rect? {
        if (points == null || points.isEmpty()) return null

        val rawLeft = points.minOf { it.x }
        val rawTop = points.minOf { it.y }
        val rawRight = points.maxOf { it.x }
        val rawBottom = points.maxOf { it.y }

        if (!TrackingFrameBounds.containsRect(
                left = rawLeft,
                top = rawTop,
                right = rawRight,
                bottom = rawBottom,
                maxWidth = maxW,
                maxHeight = maxH
            )
        ) {
            return null
        }

        return android.graphics.Rect(rawLeft, rawTop, rawRight, rawBottom)
    }

    private fun rectToPoints(rect: android.graphics.Rect): Array<android.graphics.Point> {
        return arrayOf(
            android.graphics.Point(rect.left, rect.top),
            android.graphics.Point(rect.right, rect.top),
            android.graphics.Point(rect.right, rect.bottom),
            android.graphics.Point(rect.left, rect.bottom)
        )
    }

    private fun transformPoint(x: Double, y: Double, transform: TrackingTransform): Point {
        return Point(
            transform.m00 * x + transform.m01 * y + transform.m02,
            transform.m10 * x + transform.m11 * y + transform.m12
        )
    }

    private fun Mat.readDouble(row: Int, col: Int): Double? {
        return get(row, col)?.getOrNull(0)
    }
}
