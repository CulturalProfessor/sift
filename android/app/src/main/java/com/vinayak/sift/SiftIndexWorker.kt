package com.vinayak.sift

import android.content.Context
import android.util.Log
import androidx.work.Constraints
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.Worker
import androidx.work.WorkerParameters
import java.util.concurrent.TimeUnit

/**
 * Periodic catch-up indexing so photos/videos taken while the app is closed
 * are already searchable at next open, not just picked up on the next
 * foreground index pass. WorkManager (rather than a plain background thread)
 * is what actually gets opportunistic execution while the app isn't running —
 * Android freezes ordinary background threads within about a minute of the
 * process leaving the foreground.
 *
 * Reuses the same [SiftIndexer] core the RN module and [IndexingService]
 * drive, so this is the exact same incremental pass — new/changed items only,
 * same scope settings (since/max-files/videos-on) the user configured in
 * Settings.
 */
class SiftIndexWorker(context: Context, params: WorkerParameters) : Worker(context, params) {
    companion object {
        private const val TAG = "SiftIndexWorker"
        private const val UNIQUE_NAME = "sift_periodic_index"
        private val INTERVAL_HOURS = 6L

        /** Idempotent: call on every app start, WorkManager dedupes by unique name. */
        fun schedule(context: Context) {
            val constraints = Constraints.Builder()
                .setRequiresBatteryNotLow(true)
                .build()
            val request = PeriodicWorkRequestBuilder<SiftIndexWorker>(
                INTERVAL_HOURS, TimeUnit.HOURS,
            ).setConstraints(constraints).build()
            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                UNIQUE_NAME, ExistingPeriodicWorkPolicy.KEEP, request,
            )
        }
    }

    override fun doWork(): Result {
        // A large in-app pass already holds foreground priority — running a
        // second pass concurrently would just contend for the same
        // single-threaded TFLite interpreter lock for no benefit.
        if (IndexingService.isActive()) return Result.success()

        return try {
            val indexer = SiftIndexer(applicationContext)
            val photos = indexer.indexGalleryOnce()
            val videos = indexer.indexVideosOnce()
            Log.i(
                TAG,
                "Periodic catch-up: +${photos.added} photos, +${videos.videosIndexed} videos",
            )
            Result.success()
        } catch (e: Exception) {
            Log.w(TAG, "Periodic catch-up failed: ${e.message}")
            Result.retry()
        }
    }
}
