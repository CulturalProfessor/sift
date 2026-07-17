package com.vinayak.sift

import android.app.Activity
import android.app.RecoverableSecurityException
import android.content.ContentUris
import android.content.Context
import android.content.IntentSender
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import com.facebook.react.bridge.ActivityEventListener
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
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Executors
import kotlin.math.sqrt
import org.json.JSONArray
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

    // --- Delete flow: MediaStore requires user confirmation via a system dialog
    // on API 29+, delivered back as an Activity result rather than synchronously. --
    private val deleteRequestCode = 17583
    private var pendingDeletePromise: Promise? = null
    private var pendingDeleteId: Long = -1
    private var pendingDeleteIsVideo: Boolean = false

    private val activityEventListener = object : ActivityEventListener {
        override fun onActivityResult(
            activity: Activity,
            requestCode: Int,
            resultCode: Int,
            data: android.content.Intent?,
        ) {
            if (requestCode != deleteRequestCode) return
            val promise = pendingDeletePromise ?: return
            val id = pendingDeleteId
            val isVideo = pendingDeleteIsVideo
            pendingDeletePromise = null
            pendingDeleteId = -1
            if (resultCode == Activity.RESULT_OK) {
                if (id >= 0) {
                    if (isVideo) store.removeVideo(id) else store.removePhoto(id)
                }
                promise.resolve(true)
            } else {
                promise.resolve(false) // user declined the system confirmation
            }
        }

        override fun onNewIntent(intent: android.content.Intent) {}
    }

    init {
        reactContext.addActivityEventListener(activityEventListener)
    }

    private val indexExecutor = Executors.newSingleThreadExecutor()
    // Runs video indexing concurrently with photo indexing rather than after it —
    // both still funnel through the image encoder, which isn't thread-safe for
    // concurrent inference, so actual encoder calls are serialized inside
    // SiftIndexer. This overlaps decode/IO work but not inference itself.
    private val videoExecutor = Executors.newSingleThreadExecutor()
    // Separate from indexing so search stays responsive while a long index runs.
    private val searchExecutor = Executors.newSingleThreadExecutor()

    // Owns the image encoder, the embedding store, and the indexing passes —
    // shared with the WorkManager worker so in-app and background indexing
    // drive the same core.
    private val indexer by lazy {
        SiftIndexer(
            reactContext,
            onPhotoProgress = { done, total ->
                photoDone = done; photoTotal = total
                emit("SiftIndexProgress", Arguments.createMap().apply {
                    putInt("done", done)
                    putInt("total", total)
                })
                IndexingService.updateProgress(photoDone, photoTotal, videoDone, videoTotal)
            },
            onVideoProgress = { done, total ->
                videoDone = done; videoTotal = total
                emit("SiftVideoProgress", Arguments.createMap().apply {
                    putInt("done", done)
                    putInt("total", total)
                })
                IndexingService.updateProgress(photoDone, photoTotal, videoDone, videoTotal)
            },
        )
    }
    private val store get() = indexer.store
    private val embedDim get() = indexer.embedDim

    // --- Foreground service (keeps a large pass running through screen-off) ---
    // Backlogs at or under this size finish quickly enough that the in-process
    // executor survives without foreground priority; only a real bulk index
    // pays the notification's attention cost.
    private val foregroundServiceThreshold = 50
    @Volatile private var photoDone = 0
    @Volatile private var photoTotal = 0
    @Volatile private var videoDone = 0
    @Volatile private var videoTotal = 0
    @Volatile private var activeForegroundPasses = 0
    private val foregroundServiceLock = Any()

    /** Returns true if this pass registered itself with the foreground service. */
    private fun maybeStartForegroundService(backlog: Int): Boolean {
        if (backlog <= foregroundServiceThreshold) return false
        synchronized(foregroundServiceLock) {
            if (activeForegroundPasses == 0) {
                IndexingService.onCancelRequested = { indexer.bumpScopeGeneration() }
                IndexingService.start(reactContext)
            }
            activeForegroundPasses++
        }
        return true
    }

    private fun endForegroundPass(startedForeground: Boolean) {
        if (!startedForeground) return
        synchronized(foregroundServiceLock) {
            activeForegroundPasses--
            if (activeForegroundPasses <= 0) {
                activeForegroundPasses = 0
                IndexingService.stop(reactContext)
            }
        }
    }

    // Search-only prefs (match threshold, result count, recent queries) — indexing
    // prefs (throttle/scope) live on SiftIndexer, which owns that SharedPreferences
    // file jointly with this (same "sift_settings" file, safe to read/write from both).
    private val prefs by lazy {
        reactContext.getSharedPreferences("sift_settings", Context.MODE_PRIVATE)
    }

    private fun matchMinPercent(): Int = prefs.getInt("match_min_percent", 60)
    private fun topK(): Int = prefs.getInt("top_k", 5)

    private fun interpreterOptions() = Interpreter.Options().apply {
        // Multi-threaded XNNPACK CPU inference — the Helio G80 has 8 cores and no
        // NPU, so threads are the main acceleration lever.
        numThreads = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
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

    private fun emit(event: String, params: WritableMap) {
        reactContext
            .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit(event, params)
    }

    @ReactMethod
    fun embedImage(uriString: String, promise: Promise) {
        try {
            val result: WritableArray = Arguments.createArray()
            for (v in indexer.embedUri(uriString)) result.pushDouble(v.toDouble())
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
            var startedForeground = false
            try {
                val result = indexer.indexGalleryOnce(maxCount) { backlog ->
                    startedForeground = maybeStartForegroundService(backlog)
                }
                promise.resolve(Arguments.createMap().apply {
                    putInt("added", result.added)
                    putInt("removed", result.removed)
                    putInt("failed", result.failed)
                    putInt("totalIndexed", result.totalIndexed)
                })
            } catch (e: Exception) {
                promise.reject("INDEX_ERROR", e.message, e)
            } finally {
                endForegroundPass(startedForeground)
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

    // Bumped synchronously on every searchImages call (before it's even queued), so
    // a stale search still churning through the executor can tell it's been
    // superseded and bail out instead of running to completion for a discarded result.
    @Volatile
    private var latestSearchToken: Long = 0L

    /**
     * Brute-force cosine search over photos AND video keyframes. Photos and
     * keyframes share one embedding space, so a single query ranks both. Video
     * hits are deduped to the best-scoring moment per video and carry its
     * timestamp. Returns top-K as {assetId, uri, score, isVideo, timestampMs}.
     */
    @ReactMethod
    fun searchImages(queryEmb: ReadableArray, topK: Int, token: Double, promise: Promise) {
        val myToken = token.toLong()
        latestSearchToken = myToken
        searchExecutor.execute {
            try {
                if (myToken != latestSearchToken) {
                    promise.resolve(Arguments.createArray())
                    return@execute
                }

                val q = FloatArray(embedDim) { queryEmb.getDouble(it).toFloat() }
                var qn = 0f
                for (v in q) qn += v * v
                qn = sqrt(qn).coerceAtLeast(1e-9f)
                for (i in q.indices) q[i] /= qn

                promise.resolve(rankAndFilter(q, topK, excludeId = null, excludeIsVideo = false) {
                    myToken == latestSearchToken
                })
            } catch (e: Exception) {
                promise.reject("SEARCH_ERROR", e.message, e)
            }
        }
    }

    /**
     * "Find similar": reuse a stored asset's own embedding as the query — no
     * inference needed, so this is effectively instant. Excludes the source
     * asset itself from the results.
     */
    @ReactMethod
    fun searchByAsset(assetId: Double, isVideo: Boolean, topK: Int, promise: Promise) {
        searchExecutor.execute {
            try {
                val id = assetId.toLong()
                val source: FloatArray = if (isVideo) {
                    val keyframes = store.getVideoKeyframes(id)
                    if (keyframes.isEmpty()) {
                        promise.reject("NOT_INDEXED", "Video $id has no stored embedding")
                        return@execute
                    }
                    val avg = FloatArray(embedDim)
                    for (kf in keyframes) {
                        for (i in 0 until embedDim) avg[i] += kf.embedding[i] / 127f
                    }
                    for (i in avg.indices) avg[i] /= keyframes.size
                    avg
                } else {
                    val bytes = store.getPhotoEmbedding(id)
                        ?: run {
                            promise.reject("NOT_INDEXED", "Photo $id has no stored embedding")
                            return@execute
                        }
                    FloatArray(embedDim) { bytes[it] / 127f }
                }

                var norm = 0f
                for (v in source) norm += v * v
                norm = sqrt(norm).coerceAtLeast(1e-9f)
                for (i in source.indices) source[i] /= norm

                promise.resolve(rankAndFilter(source, topK, excludeId = id, excludeIsVideo = isVideo) { true })
            } catch (e: Exception) {
                promise.reject("SEARCH_ERROR", e.message, e)
            }
        }
    }

    private data class Hit(
        val id: Long,
        val score: Float,
        val isVideo: Boolean,
        val timestampMs: Long,
    )

    /**
     * Brute-force cosine search over photos AND video keyframes given an
     * already-unit-normalized query [q]. Photos and keyframes share one
     * embedding space, so a single query ranks both. Video hits are deduped
     * to the best-scoring moment per video and carry its timestamp.
     * [stillLive] is polled between passes so a superseded text search can
     * bail out early; asset-similarity searches just pass `{ true }`.
     */
    private fun rankAndFilter(
        q: FloatArray,
        topK: Int,
        excludeId: Long?,
        excludeIsVideo: Boolean,
        stillLive: () -> Boolean,
    ): WritableArray {
        if (!stillLive()) return Arguments.createArray()

        // Photos.
        val hits = ArrayList<Hit>()
        for ((id, bytes) in store.all()) {
            if (!excludeIsVideo && id == excludeId) continue
            if (!stillLive()) return Arguments.createArray()
            hits.add(Hit(id, dotInt8(bytes, q), false, 0L))
        }

        // Video keyframes -> best moment per video.
        val bestPerVideo = HashMap<Long, Hit>()
        for (kf in store.allKeyframes()) {
            if (excludeIsVideo && kf.videoId == excludeId) continue
            if (!stillLive()) return Arguments.createArray()
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
        // Deleted-elsewhere hits are stale rows — drop them from the store now
        // rather than waiting for the next full index pass to prune them.
        for (h in pool) {
            val stillLive = if (h.isVideo) h.id in liveVideoIds else h.id in livePhotoIds
            if (!stillLive) {
                if (h.isVideo) store.removeVideo(h.id) else store.removePhoto(h.id)
            }
        }
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
        return result
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

    /**
     * Delete a photo/video from the device. On API 30+ this shows a system
     * confirmation dialog (MediaStore.createDeleteRequest); the result comes back
     * async via onActivityResult. Resolves true if deleted, false if the user
     * declined the confirmation.
     */
    @ReactMethod
    fun deleteAsset(uriString: String, isVideo: Boolean, promise: Promise) {
        val uri = Uri.parse(uriString)
        val id = ContentUris.parseId(uri)
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                val activity = reactContext.currentActivity
                if (activity == null) {
                    promise.reject("DELETE_ERROR", "No foreground activity")
                    return
                }
                val pending = MediaStore.createDeleteRequest(
                    reactContext.contentResolver, listOf(uri),
                )
                pendingDeletePromise = promise
                pendingDeleteId = id
                pendingDeleteIsVideo = isVideo
                activity.startIntentSenderForResult(
                    pending.intentSender, deleteRequestCode, null, 0, 0, 0,
                )
                // Resolved from activityEventListener once the system dialog returns.
                return
            }

            try {
                reactContext.contentResolver.delete(uri, null, null)
                if (isVideo) store.removeVideo(id) else store.removePhoto(id)
                promise.resolve(true)
            } catch (e: RecoverableSecurityException) {
                // API 29: no direct delete permission — replay via the recovery intent.
                val activity = reactContext.currentActivity
                if (activity == null) {
                    promise.reject("DELETE_ERROR", "No foreground activity")
                    return
                }
                pendingDeletePromise = promise
                pendingDeleteId = id
                pendingDeleteIsVideo = isVideo
                val sender: IntentSender = e.userAction.actionIntent.intentSender
                activity.startIntentSenderForResult(
                    sender, deleteRequestCode, null, 0, 0, 0,
                )
            }
        } catch (e: Exception) {
            promise.reject("DELETE_ERROR", e.message, e)
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
        val (sel, selArgs) = indexer.dateSelection(dateCol)
        reactContext.contentResolver.query(
            uri, arrayOf(MediaStore.MediaColumns._ID), sel, selArgs, null,
        )?.use {
            val cap = indexer.indexMaxFiles()
            return if (cap > 0) minOf(it.count, cap) else it.count
        }
        return 0
    }

    /** Indexed-vs-total counts for photos and videos, for the library status UI. */
    @ReactMethod
    fun libraryStats(promise: Promise) {
        searchExecutor.execute {
            try {
                val videosOn = indexer.indexVideosEnabled()
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
                putInt("throttleMs", indexer.currentThrottleMs().toInt())
                putInt("deviceDefaultMs", indexer.deviceDefaultThrottleMs().toInt())
                putDouble("indexSinceMs", indexer.indexSinceMs().toDouble())
                putInt("indexMaxFiles", indexer.indexMaxFiles())
                putBoolean("indexVideos", indexer.indexVideosEnabled())
                putInt("matchMinPercent", matchMinPercent())
                putInt("topK", topK())
            })
        } catch (e: Exception) {
            promise.reject("SETTINGS_ERROR", e.message, e)
        }
    }

    /** Set the indexing throttle (ms per embed); applies live to the running index. */
    @ReactMethod
    fun setIndexThrottle(ms: Double, promise: Promise) {
        try {
            indexer.setThrottleMs(ms.toLong().coerceIn(0L, 2000L))
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
            indexer.bumpScopeGeneration()
            promise.resolve(null)
        } catch (e: Exception) {
            promise.reject("SETTINGS_ERROR", e.message, e)
        }
    }

    /** Minimum match % (of cosine similarity) for a result to be shown. */
    @ReactMethod
    fun setMatchMin(percent: Double, promise: Promise) {
        try {
            prefs.edit().putInt("match_min_percent", percent.toInt().coerceIn(0, 100)).apply()
            promise.resolve(null)
        } catch (e: Exception) {
            promise.reject("SETTINGS_ERROR", e.message, e)
        }
    }

    /** Max number of results fetched per search. */
    @ReactMethod
    fun setTopK(k: Double, promise: Promise) {
        try {
            prefs.edit().putInt("top_k", k.toInt().coerceIn(1, 100)).apply()
            promise.resolve(null)
        } catch (e: Exception) {
            promise.reject("SETTINGS_ERROR", e.message, e)
        }
    }

    private val recentQueriesKey = "recent_queries"
    private val maxRecentQueries = 10

    /** Last successful search queries, most recent first. */
    @ReactMethod
    fun getRecentQueries(promise: Promise) {
        try {
            val result: WritableArray = Arguments.createArray()
            val raw = prefs.getString(recentQueriesKey, null)
            if (raw != null) {
                val arr = JSONArray(raw)
                for (i in 0 until arr.length()) result.pushString(arr.getString(i))
            }
            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("SETTINGS_ERROR", e.message, e)
        }
    }

    /** Record a successful search query, deduped and capped at [maxRecentQueries]. */
    @ReactMethod
    fun addRecentQuery(query: String, promise: Promise) {
        try {
            val trimmed = query.trim()
            if (trimmed.isEmpty()) {
                promise.resolve(null)
                return
            }
            val existing = ArrayList<String>()
            val raw = prefs.getString(recentQueriesKey, null)
            if (raw != null) {
                val arr = JSONArray(raw)
                for (i in 0 until arr.length()) existing.add(arr.getString(i))
            }
            existing.removeAll { it.equals(trimmed, ignoreCase = true) }
            existing.add(0, trimmed)
            while (existing.size > maxRecentQueries) existing.removeAt(existing.size - 1)

            val out = JSONArray()
            for (q in existing) out.put(q)
            prefs.edit().putString(recentQueriesKey, out.toString()).apply()
            promise.resolve(null)
        } catch (e: Exception) {
            promise.reject("SETTINGS_ERROR", e.message, e)
        }
    }

    /** Incrementally keyframe gallery videos (new/changed only), prune deleted. */
    @ReactMethod
    fun indexVideos(maxCount: Int, promise: Promise) {
        videoExecutor.execute {
            var startedForeground = false
            try {
                val result = indexer.indexVideosOnce(maxCount) { backlog ->
                    startedForeground = maybeStartForegroundService(backlog)
                }
                promise.resolve(Arguments.createMap().apply {
                    putInt("videosIndexed", result.videosIndexed)
                    putInt("videosRemoved", result.videosRemoved)
                    putInt("videosFailed", result.videosFailed)
                    putInt("totalVideos", result.totalVideos)
                    putInt("totalKeyframes", result.totalKeyframes)
                })
            } catch (e: Exception) {
                promise.reject("VIDEO_INDEX_ERROR", e.message, e)
            } finally {
                endForegroundPass(startedForeground)
            }
        }
    }
}
