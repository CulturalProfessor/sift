package com.vinayak.sift

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

/**
 * SQLite store for per-photo image embeddings.
 *
 * Each embedding is L2-normalized then quantized to int8 (512 bytes/photo), so
 * even a huge library stays in the tens of MB. Keyed by the MediaStore asset id
 * with date_modified, which is what makes indexing incremental: a later run only
 * embeds ids that are new or whose date_modified changed.
 */
class EmbeddingStore(context: Context) :
    SQLiteOpenHelper(context, "sift.db", null, 2) {

    init {
        // WAL lets search read concurrently while indexing writes (no lock stall).
        setWriteAheadLoggingEnabled(true)
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            "CREATE TABLE embeddings (" +
                "asset_id INTEGER PRIMARY KEY, " +
                "date_modified INTEGER NOT NULL, " +
                "embedding BLOB NOT NULL)"
        )
        createVideoTables(db)
    }

    private fun createVideoTables(db: SQLiteDatabase) {
        // One row per stored keyframe (a video has several).
        db.execSQL(
            "CREATE TABLE IF NOT EXISTS video_keyframes (" +
                "kf_id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                "video_id INTEGER NOT NULL, " +
                "timestamp_ms INTEGER NOT NULL, " +
                "embedding BLOB NOT NULL)"
        )
        db.execSQL(
            "CREATE INDEX IF NOT EXISTS idx_kf_video ON video_keyframes(video_id)"
        )
        // Tracks which videos are indexed (for incremental re-indexing).
        db.execSQL(
            "CREATE TABLE IF NOT EXISTS videos (" +
                "video_id INTEGER PRIMARY KEY, " +
                "date_modified INTEGER NOT NULL)"
        )
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        // v1 -> v2 adds video tables; keep the (expensive to rebuild) photo index.
        if (oldVersion < 2) createVideoTables(db)
    }

    /** asset_id -> date_modified for everything already indexed. */
    fun indexedState(): HashMap<Long, Long> {
        val map = HashMap<Long, Long>()
        readableDatabase.query(
            "embeddings",
            arrayOf("asset_id", "date_modified"),
            null, null, null, null, null,
        ).use { c ->
            while (c.moveToNext()) map[c.getLong(0)] = c.getLong(1)
        }
        return map
    }

    fun upsert(assetId: Long, dateModified: Long, embedding: ByteArray) {
        val values = ContentValues().apply {
            put("asset_id", assetId)
            put("date_modified", dateModified)
            put("embedding", embedding)
        }
        writableDatabase.insertWithOnConflict(
            "embeddings", null, values, SQLiteDatabase.CONFLICT_REPLACE
        )
    }

    /** Remove embeddings whose asset_id is no longer present in the gallery. */
    fun pruneNotIn(liveIds: Set<Long>): Int {
        if (liveIds.isEmpty()) return 0
        var removed = 0
        val existing = indexedState().keys
        val db = writableDatabase
        db.beginTransaction()
        try {
            for (id in existing) {
                if (id !in liveIds) {
                    db.delete("embeddings", "asset_id = ?", arrayOf(id.toString()))
                    removed++
                }
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        return removed
    }

    /** All stored embeddings as (asset_id, int8 bytes) for brute-force search. */
    fun all(): List<Pair<Long, ByteArray>> {
        val out = ArrayList<Pair<Long, ByteArray>>()
        readableDatabase.query(
            "embeddings",
            arrayOf("asset_id", "embedding"),
            null, null, null, null, null,
        ).use { c ->
            while (c.moveToNext()) out.add(c.getLong(0) to c.getBlob(1))
        }
        return out
    }

    fun count(): Int {
        readableDatabase.rawQuery("SELECT COUNT(*) FROM embeddings", null).use { c ->
            return if (c.moveToFirst()) c.getInt(0) else 0
        }
    }

    /** Wipe everything (photos + video keyframes) — for the Settings reset. */
    fun clearAll() {
        writableDatabase.apply {
            delete("embeddings", null, null)
            delete("video_keyframes", null, null)
            delete("videos", null, null)
        }
    }

    // --- Video keyframes -----------------------------------------------------

    /** video_id -> date_modified for videos already keyframed. */
    fun indexedVideoState(): HashMap<Long, Long> {
        val map = HashMap<Long, Long>()
        readableDatabase.query(
            "videos", arrayOf("video_id", "date_modified"), null, null, null, null, null,
        ).use { c ->
            while (c.moveToNext()) map[c.getLong(0)] = c.getLong(1)
        }
        return map
    }

    /** Replace all keyframes for one video (adaptive keyframing produces the list). */
    fun replaceVideoKeyframes(
        videoId: Long,
        dateModified: Long,
        keyframes: List<Pair<Long, ByteArray>>, // (timestampMs, int8 embedding)
    ) {
        val db = writableDatabase
        db.beginTransaction()
        try {
            db.delete("video_keyframes", "video_id = ?", arrayOf(videoId.toString()))
            for ((ts, emb) in keyframes) {
                db.insert("video_keyframes", null, ContentValues().apply {
                    put("video_id", videoId)
                    put("timestamp_ms", ts)
                    put("embedding", emb)
                })
            }
            db.insertWithOnConflict("videos", null, ContentValues().apply {
                put("video_id", videoId)
                put("date_modified", dateModified)
            }, SQLiteDatabase.CONFLICT_REPLACE)
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    fun pruneVideosNotIn(liveIds: Set<Long>): Int {
        var removed = 0
        val db = writableDatabase
        db.beginTransaction()
        try {
            for (id in indexedVideoState().keys) {
                if (id !in liveIds) {
                    db.delete("video_keyframes", "video_id = ?", arrayOf(id.toString()))
                    db.delete("videos", "video_id = ?", arrayOf(id.toString()))
                    removed++
                }
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        return removed
    }

    data class Keyframe(val videoId: Long, val timestampMs: Long, val embedding: ByteArray)

    fun allKeyframes(): List<Keyframe> {
        val out = ArrayList<Keyframe>()
        readableDatabase.query(
            "video_keyframes",
            arrayOf("video_id", "timestamp_ms", "embedding"),
            null, null, null, null, null,
        ).use { c ->
            while (c.moveToNext()) {
                out.add(Keyframe(c.getLong(0), c.getLong(1), c.getBlob(2)))
            }
        }
        return out
    }

    fun videoCount(): Int {
        readableDatabase.rawQuery("SELECT COUNT(*) FROM videos", null).use { c ->
            return if (c.moveToFirst()) c.getInt(0) else 0
        }
    }

    fun keyframeCount(): Int {
        readableDatabase.rawQuery("SELECT COUNT(*) FROM video_keyframes", null).use { c ->
            return if (c.moveToFirst()) c.getInt(0) else 0
        }
    }
}
