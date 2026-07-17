package com.vinayak.sift

import android.app.ActivityManager
import android.content.ContentUris
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Process
import android.provider.MediaStore
import android.util.Log
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.roundToInt
import kotlin.math.sqrt
import org.tensorflow.lite.Interpreter

data class IndexResult(val added: Int, val removed: Int, val failed: Int, val totalIndexed: Int)

data class VideoIndexResult(
    val videosIndexed: Int,
    val videosRemoved: Int,
    val videosFailed: Int,
    val totalVideos: Int,
    val totalKeyframes: Int,
)

/**
 * Owns the on-device MobileCLIP2-S0 image encoder, the embedding store, and the
 * incremental photo/video indexing passes. Extracted out of the RN module so a
 * foreground service and a WorkManager worker (background/killed-app indexing)
 * can drive the same indexing core the RN module drives in-app, sharing one
 * [EmbeddingStore] and one TFLite interpreter (guarded by [interpreterLock] —
 * it isn't thread-safe for concurrent inference).
 *
 * [onPhotoProgress]/[onVideoProgress] fire every ~10 items (and on completion)
 * so a caller can surface a progress bar / notification without polling.
 */
class SiftIndexer(
    private val context: Context,
    private val onPhotoProgress: (done: Int, total: Int) -> Unit = { _, _ -> },
    private val onVideoProgress: (done: Int, total: Int) -> Unit = { _, _ -> },
) {
    companion object {
        private const val TAG = "SiftIndexer"
    }

    val store by lazy { EmbeddingStore(context) }

    private val imageSize = 256
    val embedDim = 512

    private val interpreterLock = Any()

    private val prefs by lazy {
        context.getSharedPreferences("sift_settings", Context.MODE_PRIVATE)
    }

    @Volatile
    private var throttleMs: Long = -1

    fun deviceDefaultThrottleMs(): Long {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        val gb = mi.totalMem / (1024.0 * 1024 * 1024)
        return when {
            gb < 4.0 -> 250L // low-end (e.g. Galaxy F22): coolest
            gb < 6.0 -> 120L // mid
            else -> 50L // high-end: fast
        }
    }

    fun currentThrottleMs(): Long {
        if (throttleMs < 0) {
            throttleMs = prefs.getLong("throttle_ms", deviceDefaultThrottleMs())
        }
        return throttleMs
    }

    fun setThrottleMs(ms: Long) {
        throttleMs = ms
        prefs.edit().putLong("throttle_ms", ms).apply()
    }

    // --- Index scope (how far back to index, and whether to index videos) ----

    fun indexSinceMs(): Long = prefs.getLong("index_since_ms", 0L) // 0 = all time
    fun indexMaxFiles(): Int = prefs.getInt("index_max_files", 0) // 0 = no limit
    fun indexVideosEnabled(): Boolean = prefs.getBoolean("index_videos", true)

    // Bumped whenever the scope changes. An indexGallery/indexVideos pass already
    // running under the old scope checks this and aborts early instead of running
    // to completion with a stale (often larger) cap — otherwise a scope change
    // while indexing is in progress gets silently ignored until the stale pass
    // finishes on its own.
    @Volatile
    private var scopeGeneration: Long = 0L

    fun bumpScopeGeneration() {
        scopeGeneration++
    }

    /** Cutoff for DATE_MODIFIED (seconds since epoch); 0 means no lower bound. */
    private fun cutoffSeconds(): Long {
        val since = indexSinceMs()
        return if (since > 0) since / 1000 else 0
    }

    /** MediaStore selection + args for the current cutoff, or (null,null) for all. */
    fun dateSelection(dateCol: String): Pair<String?, Array<String>?> {
        val cutoff = cutoffSeconds()
        return if (cutoff > 0) "$dateCol >= ?" to arrayOf(cutoff.toString())
        else null to null
    }

    private fun interpreterOptions() = Interpreter.Options().apply {
        // Multi-threaded XNNPACK CPU inference — the Helio G80 has 8 cores and no
        // NPU, so threads are the main acceleration lever.
        numThreads = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
    }

    private val imageInterpreter: Interpreter by lazy {
        Interpreter(loadAsset("image_encoder.tflite"), interpreterOptions())
    }

    private fun loadAsset(name: String): MappedByteBuffer {
        val fd = context.assets.openFd(name)
        FileInputStream(fd.fileDescriptor).use { fis ->
            return fis.channel.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength,
            )
        }
    }

    /** Decode the image at [uriString], resize shorter side to 256, center-crop 256x256. */
    private fun decodeAndCrop(uriString: String): Bitmap {
        val uri = Uri.parse(uriString)

        // Pass 1: read dimensions only.
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri).use { input ->
            BitmapFactory.decodeStream(input, null, bounds)
        }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw IllegalArgumentException("Could not decode image: $uriString")
        }

        // Pass 2: decode with inSampleSize so the shorter side stays >= 2*256.
        // This box-filter downsample during decode approximates the antialiasing
        // that torchvision's resize applies, keeping embeddings faithful to the
        // Python reference (plain bilinear from a huge photo aliases badly).
        val shortSide = minOf(bounds.outWidth, bounds.outHeight)
        var sample = 1
        while (shortSide / (sample * 2) >= imageSize * 2) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        val bitmap =
            context.contentResolver.openInputStream(uri).use { input ->
                BitmapFactory.decodeStream(input, null, opts)
                    ?: throw IllegalArgumentException("Could not decode image: $uriString")
            }

        return resizeCrop(bitmap)
    }

    /** Resize shorter side to 256, center-crop 256x256. */
    private fun resizeCrop(bitmap: Bitmap): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val scale = imageSize.toFloat() / minOf(w, h)
        val scaledW = Math.round(w * scale)
        val scaledH = Math.round(h * scale)
        val scaled = Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, true)
        val x = (scaledW - imageSize) / 2
        val y = (scaledH - imageSize) / 2
        return Bitmap.createBitmap(scaled, x, y, imageSize, imageSize)
    }

    /** Fill a direct float32 buffer in NCHW order with pixels scaled to [0,1]. */
    private fun toInputBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer =
            ByteBuffer.allocateDirect(3 * imageSize * imageSize * 4)
                .order(ByteOrder.nativeOrder())
        val pixels = IntArray(imageSize * imageSize)
        bitmap.getPixels(pixels, 0, imageSize, 0, 0, imageSize, imageSize)
        // NCHW: write the full R plane, then G, then B.
        for (channel in 0 until 3) {
            val shift = when (channel) {
                0 -> 16 // R
                1 -> 8 // G
                else -> 0 // B
            }
            for (p in pixels) {
                buffer.putFloat(((p shr shift) and 0xFF) / 255f)
            }
        }
        buffer.rewind()
        return buffer
    }

    /** Preprocess a bitmap (already-decoded frame) and run the image encoder. */
    private fun embedBitmap(bitmap: Bitmap): FloatArray {
        val input = toInputBuffer(resizeCrop(bitmap))
        val output = Array(1) { FloatArray(embedDim) }
        synchronized(interpreterLock) { imageInterpreter.run(input, output) }
        return output[0]
    }

    /** Decode -> preprocess -> run the image encoder, returning the 512-d embedding. */
    fun embedUri(uriString: String): FloatArray {
        val input = toInputBuffer(decodeAndCrop(uriString))
        val output = Array(1) { FloatArray(embedDim) }
        synchronized(interpreterLock) { imageInterpreter.run(input, output) }
        return output[0]
    }

    /** L2-normalize then quantize to int8 (512 bytes) for compact storage. */
    private fun quantizeForStorage(embedding: FloatArray): ByteArray {
        var norm = 0f
        for (v in embedding) norm += v * v
        norm = sqrt(norm).coerceAtLeast(1e-9f)
        val bytes = ByteArray(embedding.size)
        for (i in embedding.indices) {
            val q = (embedding[i] / norm * 127f).roundToInt().coerceIn(-127, 127)
            bytes[i] = q.toByte()
        }
        return bytes
    }

    /**
     * Incrementally index the gallery: enumerate MediaStore images, embed only
     * the ones that are new or whose date_modified changed, prune deleted ones,
     * and report progress via [onPhotoProgress] along the way.
     *
     * @param maxCount cap for testing; 0 = all images.
     */
    fun indexGalleryOnce(maxCount: Int = 0, onBacklogKnown: (Int) -> Unit = {}): IndexResult {
        val myGeneration = scopeGeneration
        // Background priority so foreground search preempts indexing on the CPU.
        Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)

        val indexed = store.indexedState()
        val liveIds = HashSet<Long>()

        data class Item(val id: Long, val dateModified: Long, val uri: String)
        val toEmbed = ArrayList<Item>()

        val projection = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DATE_MODIFIED,
        )
        val (sel, selArgs) = dateSelection(MediaStore.Images.Media.DATE_MODIFIED)
        context.contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            projection,
            sel, selArgs,
            "${MediaStore.Images.Media.DATE_MODIFIED} DESC",
        )?.use { c ->
            val idCol = c.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            val dmCol = c.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_MODIFIED)
            val cap = indexMaxFiles() // 0 = no limit; rows are DESC by date
            while (c.moveToNext()) {
                if (myGeneration != scopeGeneration) break // scope changed mid-scan
                if (cap > 0 && liveIds.size >= cap) break
                val id = c.getLong(idCol)
                val dm = c.getLong(dmCol)
                liveIds.add(id)
                if (indexed[id] != dm) {
                    val uri = ContentUris.withAppendedId(
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id
                    )
                    toEmbed.add(Item(id, dm, uri.toString()))
                    if (maxCount > 0 && toEmbed.size >= maxCount) break
                }
            }
        }

        // A newer scope has since superseded this pass — bail out without
        // pruning/embedding against what's now a stale (often larger) scope.
        // The follow-up call the scope change triggers picks up cleanly.
        if (myGeneration != scopeGeneration) {
            return IndexResult(0, 0, 0, store.count())
        }

        val removed = store.pruneNotIn(liveIds)
        val total = toEmbed.size
        onBacklogKnown(total)
        var done = 0
        var failed = 0
        for (item in toEmbed) {
            if (myGeneration != scopeGeneration) break // scope changed mid-embed
            try {
                val emb = embedUri(item.uri)
                store.upsert(item.id, item.dateModified, quantizeForStorage(emb))
            } catch (e: Exception) {
                // Skip unreadable images rather than aborting the whole index.
                failed++
                Log.w(TAG, "Failed to embed photo ${item.id}: ${e.message}")
            }
            Thread.sleep(currentThrottleMs()) // thermal throttle
            done++
            if (done % 10 == 0 || done == total) onPhotoProgress(done, total)
        }

        return IndexResult(done, removed, failed, store.count())
    }

    // --- Video keyframing ------------------------------------------------------

    private val descSide = 16 // cheap descriptor is a 16x16 grayscale
    private val preFilterThreshold = 6f // mean abs grayscale diff below this = "no change"
    private val keyframeMaxCosine = 0.90f // store only if < this similar to last keyframe
    private val candidateIntervalMs = 2000L
    private val maxCandidatesPerVideo = 40

    /** Tiny grayscale descriptor for the cheap pre-filter (avoids embedding static frames). */
    private fun cheapDescriptor(bitmap: Bitmap): FloatArray {
        val small = Bitmap.createScaledBitmap(bitmap, descSide, descSide, true)
        val px = IntArray(descSide * descSide)
        small.getPixels(px, 0, descSide, 0, 0, descSide, descSide)
        return FloatArray(px.size) { i ->
            val p = px[i]
            0.299f * ((p shr 16) and 0xFF) +
                0.587f * ((p shr 8) and 0xFF) +
                0.114f * (p and 0xFF)
        }
    }

    private fun descriptorDiff(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) sum += kotlin.math.abs(a[i] - b[i])
        return sum / a.size
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        var na = 0f
        var nb = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        return dot / (sqrt(na) * sqrt(nb) + 1e-9f)
    }

    /**
     * Adaptive keyframing with a cheap pre-filter. Walks candidate frames; skips
     * embedding ones visually near-identical to the last embedded frame (cheap
     * grayscale diff); of the rest, stores a keyframe only when its embedding is
     * sufficiently different from the last stored keyframe (real scene change).
     */
    private fun keyframeVideo(uri: String, durationMs: Long): List<Pair<Long, ByteArray>> {
        val retriever = MediaMetadataRetriever()
        val keyframes = ArrayList<Pair<Long, ByteArray>>()
        try {
            retriever.setDataSource(context, Uri.parse(uri))
            val interval = maxOf(candidateIntervalMs, durationMs / maxCandidatesPerVideo)
            var lastEmbeddedDesc: FloatArray? = null
            var lastStoredEmb: FloatArray? = null

            var t = 0L
            while (t <= durationMs) {
                val frame = retriever.getScaledFrameAtTime(
                    t * 1000, // us
                    MediaMetadataRetriever.OPTION_CLOSEST_SYNC,
                    512, 512,
                )
                if (frame != null) {
                    val desc = cheapDescriptor(frame)
                    val stale = lastEmbeddedDesc?.let {
                        descriptorDiff(desc, it) < preFilterThreshold
                    } ?: false
                    if (!stale) {
                        lastEmbeddedDesc = desc
                        val emb = embedBitmap(frame)
                        val prev = lastStoredEmb
                        if (prev == null || cosine(emb, prev) < keyframeMaxCosine) {
                            keyframes.add(t to quantizeForStorage(emb))
                            lastStoredEmb = emb
                        }
                        Thread.sleep(currentThrottleMs()) // thermal throttle
                    }
                }
                t += interval
            }
        } finally {
            retriever.release()
        }
        return keyframes
    }

    /** Incrementally keyframe gallery videos (new/changed only), prune deleted. */
    fun indexVideosOnce(maxCount: Int = 0, onBacklogKnown: (Int) -> Unit = {}): VideoIndexResult {
        val myGeneration = scopeGeneration
        Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)

        // Videos turned off in Settings: drop any existing keyframes and stop.
        if (!indexVideosEnabled()) {
            val removed = store.pruneVideosNotIn(emptySet())
            return VideoIndexResult(0, removed, 0, 0, 0)
        }

        val indexed = store.indexedVideoState()
        val liveIds = HashSet<Long>()

        data class Vid(val id: Long, val dm: Long, val dur: Long, val uri: String)
        val toIndex = ArrayList<Vid>()

        val projection = arrayOf(
            MediaStore.Video.Media._ID,
            MediaStore.Video.Media.DATE_MODIFIED,
            MediaStore.Video.Media.DURATION,
        )
        val (vsel, vselArgs) = dateSelection(MediaStore.Video.Media.DATE_MODIFIED)
        context.contentResolver.query(
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
            projection, vsel, vselArgs,
            "${MediaStore.Video.Media.DATE_MODIFIED} DESC",
        )?.use { c ->
            val idCol = c.getColumnIndexOrThrow(MediaStore.Video.Media._ID)
            val dmCol = c.getColumnIndexOrThrow(MediaStore.Video.Media.DATE_MODIFIED)
            val durCol = c.getColumnIndexOrThrow(MediaStore.Video.Media.DURATION)
            val cap = indexMaxFiles() // 0 = no limit; rows are DESC by date
            while (c.moveToNext()) {
                if (myGeneration != scopeGeneration) break // scope changed mid-scan
                if (cap > 0 && liveIds.size >= cap) break
                val id = c.getLong(idCol)
                val dm = c.getLong(dmCol)
                liveIds.add(id)
                if (indexed[id] != dm) {
                    val uri = ContentUris.withAppendedId(
                        MediaStore.Video.Media.EXTERNAL_CONTENT_URI, id
                    )
                    toIndex.add(Vid(id, dm, c.getLong(durCol), uri.toString()))
                    if (maxCount > 0 && toIndex.size >= maxCount) break
                }
            }
        }

        if (myGeneration != scopeGeneration) {
            return VideoIndexResult(0, 0, 0, store.videoCount(), store.keyframeCount())
        }

        val removed = store.pruneVideosNotIn(liveIds)
        val total = toIndex.size
        onBacklogKnown(total)
        var done = 0
        var failed = 0
        for (v in toIndex) {
            if (myGeneration != scopeGeneration) break // scope changed mid-embed
            try {
                val kfs = keyframeVideo(v.uri, v.dur)
                store.replaceVideoKeyframes(v.id, v.dm, kfs)
            } catch (e: Exception) {
                // Skip unreadable/corrupt videos.
                failed++
                Log.w(TAG, "Failed to keyframe video ${v.id}: ${e.message}")
            }
            done++
            onVideoProgress(done, total)
        }

        return VideoIndexResult(done, removed, failed, store.videoCount(), store.keyframeCount())
    }
}
