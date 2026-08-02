/* src/screens/Upload.jsx */
import React, { useState, useRef, useCallback } from 'react'
import { uploadVideo } from '../api'
import { Spinner } from '../components/UI'

const ACCEPTED = ['video/mp4','video/quicktime','video/x-msvideo','video/webm']
const ACCEPTED_EXT = '.mp4, .mov, .avi, .webm'

function fmtBytes(b) {
  if (b < 1024) return `${b} B`
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1024 / 1024).toFixed(1)} MB`
}
function fmtDuration(s) {
  const m = Math.floor(s / 60), sec = Math.floor(s % 60)
  return `${m}:${String(sec).padStart(2, '0')}`
}

export default function UploadScreen({ onProcessStart }) {
  const [file, setFile]         = useState(null)
  const [meta, setMeta]         = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading]= useState(false)
  const [uploadPct, setUploadPct]= useState(0)
  const [error, setError]       = useState(null)
  const inputRef                = useRef()
  const videoRef                = useRef()

  const handleFile = useCallback((f) => {
    if (!f) return
    if (f.size > 2 * 1024 * 1024 * 1024) {
      setError('File exceeds the 2 GB limit.')
      return
    }
    setError(null)
    setFile(f)

    // Extract duration via <video> element
    const url = URL.createObjectURL(f)
    const v = document.createElement('video')
    v.preload = 'metadata'
    v.onloadedmetadata = () => {
      setMeta({
        duration:   v.duration,
        width:      v.videoWidth,
        height:     v.videoHeight,
        size:       f.size,
        type:       f.type,
        name:       f.name,
        preview:    url,
      })
    }
    v.src = url
  }, [])

  const onDrop = (e) => {
    e.preventDefault(); setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  const handleProcess = async () => {
    if (!file) return
    setUploading(true)
    setError(null)
    try {
      const result = await uploadVideo(file, setUploadPct)
      onProcessStart(result.session_id, meta)
    } catch (err) {
      setError(err.message)
      setUploading(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-24 animate-fade-in">
      {/* Heading */}
      <div className="text-center mb-10">
        <h1 className="font-display text-4xl font-bold text-ink mb-2">
          Upload Your <span className="text-amber-glow">Meeting</span>
        </h1>
        <p className="text-ink-muted text-base max-w-md">
          Drop a recording and MeetingIQ will extract a full transcript, speaker diarisation, and AI-generated intelligence report.
        </p>
      </div>

      {/* Drop zone */}
      <div
        onDrop={onDrop}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onClick={() => !file && inputRef.current?.click()}
        className={`relative w-full max-w-2xl rounded-2xl border-2 border-dashed transition-all duration-200 cursor-pointer
          ${dragOver ? 'drop-active' : 'border-bg-border hover:border-amber-dim'}
          ${file ? 'cursor-default' : ''}
          bg-bg-surface`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(',')}
          className="hidden"
          onChange={e => handleFile(e.target.files[0])}
        />

        {!file ? (
          /* Empty state */
          <div className="flex flex-col items-center justify-center py-20 px-8 text-center">
            <div className="w-16 h-16 rounded-2xl bg-amber-glow/10 border border-amber-glow/20 flex items-center justify-center mb-5">
              <svg className="w-8 h-8 text-amber-glow" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
              </svg>
            </div>
            <p className="text-ink font-semibold text-lg mb-1">Drag & drop or click to browse</p>
            <p className="text-ink-muted text-sm mb-6">Max 2 GB · Max 2 hours</p>
            <div className="flex items-center gap-2">
              {['MP4','MOV','AVI','WebM'].map(f => (
                <span key={f} className="px-2.5 py-1 bg-bg-elevated border border-bg-border rounded-lg text-xs font-mono text-ink-muted">{f}</span>
              ))}
            </div>
          </div>
        ) : (
          /* File preview */
          <div className="p-6 flex gap-6">
            {/* Thumbnail */}
            <div className="relative flex-shrink-0">
              {meta?.preview ? (
                <video
                  ref={videoRef}
                  src={meta.preview}
                  className="w-48 h-28 object-cover rounded-xl border border-bg-border"
                  muted
                />
              ) : (
                <div className="w-48 h-28 rounded-xl bg-bg-elevated border border-bg-border flex items-center justify-center">
                  <span className="text-3xl">🎬</span>
                </div>
              )}
              <div className="absolute bottom-2 right-2 px-1.5 py-0.5 bg-black/70 rounded text-xs font-mono text-white">
                {meta ? fmtDuration(meta.duration) : '—:——'}
              </div>
            </div>

            {/* Meta grid */}
            <div className="flex-1 min-w-0">
              <p className="font-display font-semibold text-ink text-base truncate mb-4">{file.name}</p>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: 'File size',   value: fmtBytes(file.size) },
                  { label: 'Duration',    value: meta ? fmtDuration(meta.duration) : 'Loading…' },
                  { label: 'Resolution',  value: meta ? `${meta.width}×${meta.height}` : 'Loading…' },
                  { label: 'Format',      value: file.type.split('/')[1]?.toUpperCase() || '—' },
                ].map(({ label, value }) => (
                  <div key={label} className="bg-bg-elevated rounded-lg p-2.5">
                    <p className="text-xs text-ink-muted font-mono uppercase tracking-wider mb-0.5">{label}</p>
                    <p className="text-sm text-ink font-medium">{value}</p>
                  </div>
                ))}
              </div>

              <button
                onClick={() => { setFile(null); setMeta(null) }}
                className="mt-3 text-xs text-ink-muted hover:text-fail transition-colors"
              >
                ✕ Remove file
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mt-4 px-4 py-3 bg-fail/10 border border-fail/25 rounded-xl text-fail text-sm max-w-2xl w-full">
          {error}
        </div>
      )}

      {/* Upload progress */}
      {uploading && (
        <div className="mt-4 w-full max-w-2xl">
          <div className="flex items-center justify-between text-xs text-ink-muted font-mono mb-1.5">
            <span>Uploading…</span>
            <span>{uploadPct}%</span>
          </div>
          <div className="h-1.5 bg-bg-elevated rounded-full overflow-hidden">
            <div
              className="h-full bg-amber-glow rounded-full transition-all duration-300"
              style={{ width: `${uploadPct}%` }}
            />
          </div>
        </div>
      )}

      {/* CTA */}
      <button
        onClick={handleProcess}
        disabled={!file || uploading}
        className={`mt-6 flex items-center gap-2.5 px-8 py-3.5 rounded-xl font-display font-semibold text-base transition-all duration-200
          ${file && !uploading
            ? 'bg-amber-glow text-black hover:bg-amber-DEFAULT hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-amber-glow/20'
            : 'bg-bg-elevated text-ink-faint cursor-not-allowed'
          }`}
      >
        {uploading ? (
          <><Spinner size="sm" color="white" /> Processing…</>
        ) : (
          <>
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
            </svg>
            Process Meeting
          </>
        )}
      </button>

      <p className="mt-3 text-xs text-ink-faint">
        Processing typically takes 2–4× real-time. A 1-hour meeting takes ~2–4 minutes.
      </p>
    </div>
  )
}
