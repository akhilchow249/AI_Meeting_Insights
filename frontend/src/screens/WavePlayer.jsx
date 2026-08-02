/* src/screens/WavePlayer.jsx — WaveSurfer.js waveform player */
import React, { useEffect, useRef, useState } from 'react'
import { audioUrl } from '../api'

function fmtTime(s) {
  if (!isFinite(s)) return '0:00'
  const m = Math.floor(s / 60), sec = Math.floor(s % 60)
  return `${m}:${String(sec).padStart(2,'0')}`
}

export default function WavePlayer({ sessionId, currentTime, onSeek, onTimeUpdate }) {
  const containerRef = useRef(null)
  const wsRef        = useRef(null)
  const [duration,  setDuration]  = useState(0)
  const [playTime,  setPlayTime]  = useState(0)
  const [playing,   setPlaying]   = useState(false)
  const [loading,   setLoading]   = useState(true)

  useEffect(() => {
    if (!containerRef.current) return

    let ws
    import('wavesurfer.js').then(({ default: WaveSurfer }) => {
      ws = WaveSurfer.create({
        container:        containerRef.current,
        waveColor:        '#1a2540',
        progressColor:    '#f0a000',
        cursorColor:      '#f0a000',
        cursorWidth:      2,
        barWidth:         2,
        barGap:           1,
        barRadius:        2,
        height:           72,
        normalize:        true,
        backend:          'WebAudio',
        interact:         true,
      })
      wsRef.current = ws

      ws.on('ready',       () => { setLoading(false); setDuration(ws.getDuration()) })
      ws.on('timeupdate',  (t) => { setPlayTime(t); onTimeUpdate?.(t) })
      ws.on('play',        () => setPlaying(true))
      ws.on('pause',       () => setPlaying(false))
      ws.on('finish',      () => setPlaying(false))
      ws.on('interaction', (t) => onSeek?.(t))

      ws.load(audioUrl(sessionId))
    }).catch(err => {
      console.warn('WaveSurfer not loaded:', err)
      setLoading(false)
    })

    return () => { ws?.destroy(); wsRef.current = null }
  }, [sessionId])

  // Seek from outside (transcript click)
  useEffect(() => {
    if (wsRef.current && currentTime != null && isFinite(currentTime)) {
      const dur = wsRef.current.getDuration()
      if (dur > 0) wsRef.current.seekTo(currentTime / dur)
    }
  }, [currentTime])

  const togglePlay = () => wsRef.current?.playPause()

  return (
    <div className="bg-bg-surface border border-bg-border rounded-xl overflow-hidden">
      {/* Waveform */}
      <div className="px-4 pt-4 pb-2 relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-bg-surface/80 z-10">
            <div className="w-5 h-5 border-2 border-amber-glow border-t-transparent rounded-full animate-spin" />
          </div>
        )}
        <div ref={containerRef} />
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3 px-4 pb-4">
        <button
          onClick={togglePlay}
          className="w-9 h-9 rounded-full bg-amber-glow/15 border border-amber-glow/30 flex items-center justify-center hover:bg-amber-glow/25 transition-colors text-amber-glow"
        >
          {playing ? (
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <path d="M6 19h4V5H6zm8-14v14h4V5z" />
            </svg>
          ) : (
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>

        <span className="text-xs font-mono text-ink-muted min-w-[80px]">
          {fmtTime(playTime)} / {fmtTime(duration)}
        </span>

        {/* Volume */}
        <div className="flex-1" />
        <input
          type="range" min={0} max={1} step={0.05} defaultValue={1}
          className="w-20 accent-amber-glow"
          onChange={e => wsRef.current?.setVolume(parseFloat(e.target.value))}
        />
      </div>
    </div>
  )
}
