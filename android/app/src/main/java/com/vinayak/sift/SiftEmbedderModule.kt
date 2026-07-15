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
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap
import com.facebook.react.modules.core.DeviceEventManagerModule
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Executors
import kotlin.math.roundToInt
import kotlin.math.sqrt
import org.tensorflow.lite.Interpreter

/**
 * On-device MobileCLIP2-S0 image embedding via LiteRT.
 *
 * First increment: prove we can load the bundled .tflite image encoder and turn
 * a real gallery photo into a 512-d embedding on the device. Preprocessing for
 * MobileCLIP2-S0 is just: resize shorter side to 256, center-crop 256x256, RGB
 * scaled to [0,1] (the model uses NO mean/std normalization), laid out NCHW.
 */
class SiftEmbedderModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    override fun getName() = "SiftEmbedder"

    private val imageSize = 256
    private val embedDim = 512

    private val store by lazy { EmbeddingStore(reactContext) }
    private val indexExecutor = Executors.newSingleThreadExecutor()
    // Runs video indexing concurrently with photo indexing rather than after it —
    // both still funnel through imageInterpreter, which isn't thread-safe for
    // concurrent inference, so actual encoder calls are serialized via
    // interpreterLock. This overlaps decode/IO work but not inference itself.
    private val videoExecutor = Executors.newSingleThreadExecutor()
    private val interpreterLock = Any()
    // Separate from indexing so search stays responsive while a long index runs.
    private val searchExecutor = Executors.newSingleThreadExecutor()

    // Idle gap after each embed so the SoC can shed heat. Device-dependent default
    // (lower-RAM phones throttle more), persisted, and overridable from Settings.
    private val prefs by lazy {
        reactContext.getSharedPreferences("sift_settings", Context.MODE_PRIVATE)
    }

    @Volatile
    private var throttleMs: Long = -1

    private fun deviceDefaultThrottleMs(): Long {
        val am = reactContext.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        val gb = mi.totalMem / (1024.0 * 1024 * 1024)
        return when {
            gb < 4.0 -> 250L // low-end (e.g. Galaxy F22): coolest
            gb < 6.0 -> 120L // mid
            else -> 50L // high-end: fast
        }
    }

    private fun currentThrottleMs(): Long {
        if (throttleMs < 0) {
            throttleMs = prefs.getLong("throttle_ms", deviceDefaultThrottleMs())
        }
        return throttleMs
    }

    // --- Index scope (how far back to index, and whether to index videos) ----

    private fun indexSinceMs(): Long = prefs.getLong("index_since_ms", 0L) // 0 = all time
    private fun indexMaxFiles(): Int = prefs.getInt("index_max_files", 0) // 0 = no limit
    private fun indexVideosEnabled(): Boolean = prefs.getBoolean("index_videos", true)

    /** Cutoff for DATE_MODIFIED (seconds since epoch); 0 means no lower bound. */
    private fun cutoffSeconds(): Long {
        val since = indexSinceMs()
        return if (since > 0) since / 1000 else 0
    }

    /** MediaStore selection + args for the current cutoff, or (null,null) for all. */
    private fun dateSelection(dateCol: String): Pair<String?, Array<String>?> {
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

    private val textInterpreter: Interpreter by lazy {
        Interpreter(loadAsset("text_encoder.tflite"), interpreterOptions())
    }

    private fun loadAsset(name: String): MappedByteBuffer {
        val fd = reactContext.assets.openFd(name)
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
        reactContext.contentResolver.openInputStream(uri).use { input ->
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
            reactContext.contentResolver.openInputStream(uri).use { input ->
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
    private fun embedUri(uriString: String): FloatArray {
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

    private fun emit(event: String, params: WritableMap) {
        reactContext
            .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit(event, params)
    }

    @ReactMethod
    fun embedImage(uriString: String, promise: Promise) {
        try {
            val result: WritableArray = Arguments.createArray()
            for (v in embedUri(uriString)) result.pushDouble(v.toDouble())
            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("EMBED_IMAGE_ERROR", e.message, e)
        }
    }

    /**
     * Incrementally index the gallery: enumerate MediaStore images, embed only
     * the ones that are new or whose date_modified changed, prune deleted ones,
     * and emit "SiftIndexProgress" events along the way.
     *
     * @param maxCount cap for testing; 0 = all images.
     */
    @ReactMethod
    fun indexGallery(maxCount: Int, promise: Promise) {
        indexExecutor.execute {
            // Background priority so foreground search preempts indexing on the CPU.
            Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
            try {
                val indexed = store.indexedState()
                val liveIds = HashSet<Long>()

                data class Item(val id: Long, val dateModified: Long, val uri: String)
                val toEmbed = ArrayList<Item>()

                val projection = arrayOf(
                    MediaStore.Images.Media._ID,
                    MediaStore.Images.Media.DATE_MODIFIED,
                )
                val (sel, selArgs) = dateSelection(MediaStore.Images.Media.DATE_MODIFIED)
                reactContext.contentResolver.query(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                    projection,
                    sel, selArgs,
                    "${MediaStore.Images.Media.DATE_MODIFIED} DESC",
                )?.use { c ->
                    val idCol = c.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
                    val dmCol = c.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_MODIFIED)
                    val cap = indexMaxFiles() // 0 = no limit; rows are DESC by date
                    while (c.moveToNext()) {
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

                val removed = store.pruneNotIn(liveIds)
                val total = toEmbed.size
                var done = 0
                for (item in toEmbed) {
                    try {
                        val emb = embedUri(item.uri)
                        store.upsert(item.id, item.dateModified, quantizeForStorage(emb))
                    } catch (e: Exception) {
                        // Skip unreadable images rather than aborting the whole index.
                    }
                    Thread.sleep(currentThrottleMs()) // thermal throttle
                    done++
                    if (done % 10 == 0 || done == total) {
                        emit("SiftIndexProgress", Arguments.createMap().apply {
                            putInt("done", done)
                            putInt("total", total)
                        })
                    }
                }

                promise.resolve(Arguments.createMap().apply {
                    putInt("added", total)
                    putInt("removed", removed)
                    putInt("totalIndexed", store.count())
                })
            } catch (e: Exception) {
                promise.reject("INDEX_ERROR", e.message, e)
            }
        }
    }

    /** Read a bundled asset as UTF-8 text (used to load the BPE merges for the JS tokenizer). */
    @ReactMethod
    fun readAsset(name: String, promise: Promise) {
        try {
            reactContext.assets.open(name).bufferedReader().use {
                promise.resolve(it.readText())
            }
        } catch (e: Exception) {
            promise.reject("READ_ASSET_ERROR", e.message, e)
        }
    }

    private val ctxLen = 77

    /** Run the text encoder on CLIP token ids (already sot/eot-wrapped, padded to 77). */
    @ReactMethod
    fun embedText(tokenIds: ReadableArray, promise: Promise) {
        try {
            val input = Array(1) { IntArray(ctxLen) }
            for (i in 0 until ctxLen) {
                input[0][i] = if (i < tokenIds.size()) tokenIds.getInt(i) else 0
            }
            val output = Array(1) { FloatArray(embedDim) }
            textInterpreter.run(input, output)

            val result: WritableArray = Arguments.createArray()
            for (v in output[0]) result.pushDouble(v.toDouble())
            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("EMBED_TEXT_ERROR", e.message, e)
        }
    }

    private fun dotInt8(bytes: ByteArray, q: FloatArray): Float {
        var dot = 0f
        for (i in 0 until embedDim) dot += (bytes[i].toInt() / 127f) * q[i]
        return dot
    }

    /**
     * Brute-force cosine search over photos AND video keyframes. Photos and
     * keyframes share one embedding space, so a single query ranks both. Video
     * hits are deduped to the best-scoring moment per video and carry its
     * timestamp. Returns top-K as {assetId, uri, score, isVideo, timestampMs}.
     */
    @ReactMethod
    fun searchImages(queryEmb: ReadableArray, topK: Int, promise: Promise) {
        searchExecutor.execute {
            try {
                val q = FloatArray(embedDim) { queryEmb.getDouble(it).toFloat() }
                var qn = 0f
                for (v in q) qn += v * v
                qn = sqrt(qn).coerceAtLeast(1e-9f)
                for (i in q.indices) q[i] /= qn

                data class Hit(
                    val id: Long,
                    val score: Float,
                    val isVideo: Boolean,
                    val timestampMs: Long,
                )

                // Photos.
                val hits = ArrayList<Hit>()
                for ((id, bytes) in store.all()) {
                    hits.add(Hit(id, dotInt8(bytes, q), false, 0L))
                }

                // Video keyframes -> best moment per video.
                val bestPerVideo = HashMap<Long, Hit>()
                for (kf in store.allKeyframes()) {
                    val s = dotInt8(kf.embedding, q)
                    val cur = bestPerVideo[kf.videoId]
                    if (cur == null || s > cur.score) {
                        bestPerVideo[kf.videoId] = Hit(kf.videoId, s, true, kf.timestampMs)
                    }
                }
                hits.addAll(bestPerVideo.values)

                // Over-fetch, then drop hits whose file was deleted since the last full
                // index (the index only prunes deletions on app open, not per-search) —
                // check existence for just this candidate pool, not the whole library.
                val sorted = hits.sortedByDescending { it.score }
                val pool = sorted.take((topK * 3).coerceAtLeast(topK))
                val livePhotoIds = existingIds(
                    pool.filter { !it.isVideo }.map { it.id },
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                )
                val liveVideoIds = existingIds(
                    pool.filter { it.isVideo }.map { it.id },
                    MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                )
                val top = pool.filter { h ->
                    if (h.isVideo) h.id in liveVideoIds else h.id in livePhotoIds
                }.take(topK)

                val result: WritableArray = Arguments.createArray()
                for (h in top) {
                    val base = if (h.isVideo)
                        MediaStore.Video.Media.EXTERNAL_CONTENT_URI
                    else
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI
                    result.pushMap(Arguments.createMap().apply {
                        putString("assetId", h.id.toString())
                        putString("uri", ContentUris.withAppendedId(base, h.id).toString())
                        putDouble("score", h.score.toDouble())
                        putBoolean("isVideo", h.isVideo)
                        putDouble("timestampMs", h.timestampMs.toDouble())
                    })
                }
                promise.resolve(result)
            } catch (e: Exception) {
                promise.reject("SEARCH_ERROR", e.message, e)
            }
        }
    }

    /** Which of [ids] still resolve to a row under [baseUri] (i.e. weren't deleted). */
    private fun existingIds(ids: List<Long>, baseUri: Uri): Set<Long> {
        if (ids.isEmpty()) return emptySet()
        val live = HashSet<Long>()
        val placeholders = ids.joinToString(",") { "?" }
        reactContext.contentResolver.query(
            baseUri,
            arrayOf(MediaStore.MediaColumns._ID),
            "${MediaStore.MediaColumns._ID} IN ($placeholders)",
            ids.map { it.toString() }.toTypedArray(),
            null,
        )?.use { c ->
            val idCol = c.getColumnIndexOrThrow(MediaStore.MediaColumns._ID)
            while (c.moveToNext()) live.add(c.getLong(idCol))
        }
        return live
    }

    /** Open the item in the system gallery/viewer so the user sees it in context. */
    @ReactMethod
    fun openExternally(uri: String, isVideo: Boolean, promise: Promise) {
        try {
            val intent = android.content.Intent(android.content.Intent.ACTION_VIEW).apply {
                setDataAndType(Uri.parse(uri), if (isVideo) "video/*" else "image/*")
                addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
                addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            reactContext.startActivity(intent)
            promise.resolve(null)
        } catch (e: Exception) {
            promise.reject("OPEN_EXTERNAL_ERROR", e.message, e)
        }
    }

    /** Open the item in the device's Gallery/Photos app (shows it in its album/place). */
    @ReactMethod
    fun openInGallery(uri: String, isVideo: Boolean, promise: Promise) {
        try {
            val intent = android.content.Intent(android.content.Intent.ACTION_VIEW).apply {
                setDataAndType(Uri.parse(uri), if (isVideo) "video/*" else "image/*")
                addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
                addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            reactContext.startActivity(intent)
            promise.resolve(null)
        } catch (e: Exception) {
            promise.reject("OPEN_GALLERY_ERROR", e.message, e)
        }
    }

    /** Open a video in the full-screen player, seeked to [timestampMs]. */
    @ReactMethod
    fun openVideoAt(uri: String, timestampMs: Double, promise: Promise) {
        try {
            val intent = android.content.Intent(reactContext, PlayerActivity::class.java).apply {
                addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
                putExtra("uri", uri)
                putExtra("timestampMs", timestampMs.toLong())
            }
            reactContext.startActivity(intent)
            promise.resolve(null)
        } catch (e: Exception) {
            promise.reject("OPEN_VIDEO_ERROR", e.message, e)
        }
    }

    /** Asset ids currently indexed, as strings — for marking indexed thumbnails. */
    @ReactMethod
    fun indexedIds(promise: Promise) {
        try {
            val result: WritableArray = Arguments.createArray()
            for (id in store.indexedState().keys) result.pushString(id.toString())
            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("IDS_ERROR", e.message, e)
        }
    }

    // Required by RN's NativeEventEmitter contract (silences the addListener warning).
    @ReactMethod fun addListener(eventName: String) {}

    @ReactMethod fun removeListeners(count: Int) {}

    @ReactMethod
    fun indexedCount(promise: Promise) {
        try {
            promise.resolve(store.count())
        } catch (e: Exception) {
            promise.reject("COUNT_ERROR", e.message, e)
        }
    }

    /** In-scope count for [uri]: date-range filtered, then capped at max files. */
    private fun mediaStoreCount(uri: Uri, dateCol: String): Int {
        val (sel, selArgs) = dateSelection(dateCol)
        reactContext.contentResolver.query(
            uri, arrayOf(MediaStore.MediaColumns._ID), sel, selArgs, null,
        )?.use {
            val cap = indexMaxFiles()
            return if (cap > 0) minOf(it.count, cap) else it.count
        }
        return 0
    }

    /** Indexed-vs-total counts for photos and videos, for the library status UI. */
    @ReactMethod
    fun libraryStats(promise: Promise) {
        searchExecutor.execute {
            try {
                val videosOn = indexVideosEnabled()
                promise.resolve(Arguments.createMap().apply {
                    putInt("photosIndexed", store.count())
                    putInt(
                        "photosTotal",
                        mediaStoreCount(
                            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                            MediaStore.Images.Media.DATE_MODIFIED,
                        ),
                    )
                    putInt("videosIndexed", store.videoCount())
                    putInt(
                        "videosTotal",
                        if (videosOn) mediaStoreCount(
                            MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                            MediaStore.Video.Media.DATE_MODIFIED,
                        ) else 0,
                    )
                    putInt("keyframes", store.keyframeCount())
                })
            } catch (e: Exception) {
                promise.reject("STATS_ERROR", e.message, e)
            }
        }
    }

    /** All indexing settings, for the Settings screen. */
    @ReactMethod
    fun getSettings(promise: Promise) {
        try {
            promise.resolve(Arguments.createMap().apply {
                putInt("throttleMs", currentThrottleMs().toInt())
                putInt("deviceDefaultMs", deviceDefaultThrottleMs().toInt())
                putDouble("indexSinceMs", indexSinceMs().toDouble())
                putInt("indexMaxFiles", indexMaxFiles())
                putBoolean("indexVideos", indexVideosEnabled())
            })
        } catch (e: Exception) {
            promise.reject("SETTINGS_ERROR", e.message, e)
        }
    }

    /** Set the indexing throttle (ms per embed); applies live to the running index. */
    @ReactMethod
    fun setIndexThrottle(ms: Double, promise: Promise) {
        try {
            throttleMs = ms.toLong().coerceIn(0L, 2000L)
            prefs.edit().putLong("throttle_ms", throttleMs).apply()
            promise.resolve(null)
        } catch (e: Exception) {
            promise.reject("SETTINGS_ERROR", e.message, e)
        }
    }

    /** Wipe all indexed data (photos + video keyframes). */
    @ReactMethod
    fun clearIndex(promise: Promise) {
        indexExecutor.execute {
            try {
                // Photo and video indexing now run on separate executors; wait for
                // any in-flight video pass to drain before wiping so it can't write
                // through a clear.
                videoExecutor.submit {}.get()
                store.clearAll()
                promise.resolve(null)
            } catch (e: Exception) {
                promise.reject("CLEAR_ERROR", e.message, e)
            }
        }
    }

    /**
     * Set index scope: index items newer than [sinceMs] (0 = all time), at most
     * [maxFiles] most-recent items (0 = no limit), and whether to index videos.
     */
    @ReactMethod
    fun setIndexScope(
        sinceMs: Double,
        maxFiles: Double,
        indexVideos: Boolean,
        promise: Promise,
    ) {
        try {
            prefs.edit()
                .putLong("index_since_ms", sinceMs.toLong())
                .putInt("index_max_files", maxFiles.toInt())
                .putBoolean("index_videos", indexVideos)
                .apply()
            promise.resolve(null)
        } catch (e: Exception) {
            promise.reject("SETTINGS_ERROR", e.message, e)
        }
    }

    // --- Video keyframing ----------------------------------------------------

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
            retriever.setDataSource(reactContext, Uri.parse(uri))
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
    @ReactMethod
    fun indexVideos(maxCount: Int, promise: Promise) {
        videoExecutor.execute {
            Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
            try {
                // Videos turned off in Settings: drop any existing keyframes and stop.
                if (!indexVideosEnabled()) {
                    val removed = store.pruneVideosNotIn(emptySet())
                    promise.resolve(Arguments.createMap().apply {
                        putInt("videosIndexed", 0)
                        putInt("videosRemoved", removed)
                        putInt("totalVideos", 0)
                        putInt("totalKeyframes", 0)
                    })
                    return@execute
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
                reactContext.contentResolver.query(
                    MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                    projection, vsel, vselArgs,
                    "${MediaStore.Video.Media.DATE_MODIFIED} DESC",
                )?.use { c ->
                    val idCol = c.getColumnIndexOrThrow(MediaStore.Video.Media._ID)
                    val dmCol = c.getColumnIndexOrThrow(MediaStore.Video.Media.DATE_MODIFIED)
                    val durCol = c.getColumnIndexOrThrow(MediaStore.Video.Media.DURATION)
                    val cap = indexMaxFiles() // 0 = no limit; rows are DESC by date
                    while (c.moveToNext()) {
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

                val removed = store.pruneVideosNotIn(liveIds)
                val total = toIndex.size
                var done = 0
                for (v in toIndex) {
                    try {
                        val kfs = keyframeVideo(v.uri, v.dur)
                        store.replaceVideoKeyframes(v.id, v.dm, kfs)
                    } catch (e: Exception) {
                        // Skip unreadable/corrupt videos.
                    }
                    done++
                    emit("SiftVideoProgress", Arguments.createMap().apply {
                        putInt("done", done)
                        putInt("total", total)
                    })
                }

                promise.resolve(Arguments.createMap().apply {
                    putInt("videosIndexed", total)
                    putInt("videosRemoved", removed)
                    putInt("totalVideos", store.videoCount())
                    putInt("totalKeyframes", store.keyframeCount())
                })
            } catch (e: Exception) {
                promise.reject("VIDEO_INDEX_ERROR", e.message, e)
            }
        }
    }
}
