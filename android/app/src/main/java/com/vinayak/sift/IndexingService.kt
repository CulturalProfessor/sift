package com.vinayak.sift

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat

/**
 * Foreground service that keeps a large indexing pass running through
 * screen-off / app-switch (Android otherwise throttles background threads
 * within ~a minute of the app leaving the foreground). Started by
 * [SiftEmbedderModule] when a pass's backlog crosses a size threshold — the
 * actual indexing still runs on the module's own executors via [SiftIndexer];
 * this service only holds foreground priority and shows progress.
 */
class IndexingService : Service() {
    companion object {
        private const val CHANNEL_ID = "sift_indexing"
        private const val NOTIF_ID = 4201
        private const val ACTION_CANCEL = "com.vinayak.sift.action.CANCEL_INDEXING"

        @Volatile private var instance: IndexingService? = null

        /** Set by the module before [start]; invoked if the user taps Cancel. */
        @Volatile var onCancelRequested: (() -> Unit)? = null

        fun start(context: Context) {
            ContextCompat.startForegroundService(context, Intent(context, IndexingService::class.java))
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, IndexingService::class.java))
        }

        fun updateProgress(photoDone: Int, photoTotal: Int, videoDone: Int, videoTotal: Int) {
            instance?.postProgress(photoDone, photoTotal, videoDone, videoTotal)
        }

        /** True while a large in-app pass already holds foreground priority. */
        fun isActive(): Boolean = instance != null
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        instance = this
        createChannel()
        startForeground(NOTIF_ID, buildNotification(0, 0, 0, 0))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_CANCEL) {
            onCancelRequested?.invoke()
            stopSelf()
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        instance = null
        super.onDestroy()
    }

    private fun postProgress(photoDone: Int, photoTotal: Int, videoDone: Int, videoTotal: Int) {
        val nm = getSystemService(NotificationManager::class.java)
        nm.notify(NOTIF_ID, buildNotification(photoDone, photoTotal, videoDone, videoTotal))
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID, "Indexing", NotificationManager.IMPORTANCE_LOW,
            ).apply { setShowBadge(false) }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    private fun buildNotification(
        photoDone: Int,
        photoTotal: Int,
        videoDone: Int,
        videoTotal: Int,
    ): Notification {
        val cancelIntent = PendingIntent.getService(
            this, 0,
            Intent(this, IndexingService::class.java).setAction(ACTION_CANCEL),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val done = photoDone + videoDone
        val total = photoTotal + videoTotal
        val text = if (total > 0) "$done / $total items" else "Starting…"
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Sift is indexing your library")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setProgress(total.coerceAtLeast(1), done, total == 0)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .addAction(0, "Cancel", cancelIntent)
            .build()
    }
}
