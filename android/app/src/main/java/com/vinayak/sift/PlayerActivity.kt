package com.vinayak.sift

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView

/**
 * Full-screen video player that opens a clip seeked to a given timestamp — the
 * payoff of video moment search (tap "▶ 0:44" and land at 0:44). Launched via an
 * intent with "uri" and "timestampMs" extras from the native module.
 */
class PlayerActivity : ComponentActivity() {

    private var player: ExoPlayer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val uri = intent.getStringExtra("uri") ?: run { finish(); return }
        val timestampMs = intent.getLongExtra("timestampMs", 0L)

        val playerView = PlayerView(this)
        setContentView(playerView)

        player = ExoPlayer.Builder(this).build().also { exo ->
            playerView.player = exo
            exo.setMediaItem(MediaItem.fromUri(Uri.parse(uri)))
            exo.prepare()
            exo.seekTo(timestampMs)
            exo.playWhenReady = true
        }
    }

    override fun onStop() {
        super.onStop()
        player?.release()
        player = null
        finish()
    }
}
