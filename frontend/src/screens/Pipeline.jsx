/* src/screens/Pipeline.jsx
 *
 * Wired to real backend SSE via openProgressStream (api.js).
 * Falls back to demo simulation when sessionId is null/undefined.
 *
 * SSE event handling (from main.py _progress_sse_generator):
 *
 *  state_catchup              → restore current stage state on page refresh
 *  transcript_preview_catchup → replay buffered word batches on connect
 *  transcript_preview         → live word batch during ASR (Stage 3)
 *  heartbeat                  → ignore (connection keepalive)
 *  pipeline_complete          → call onComplete, close SSE
 *  plain event                → { stage, status, percent, detail } stage transition
 *
 * Stage key → array index mapping (must match MOCK_PIPELINE_STAGES order):
 *   video_ingestion   → 0
 *   audio_extraction  → 1
 *   transcription     → 2  (ASR — also fires transcript_preview events)
 *   diarisation       → 3
 *   nlp_analysis      → 4
 *   genai_report      → 5
 *   indexing          → 6
 */

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { openProgressStream, retryStage } from '../api'
import { StatusBadge, ProgressBar } from '../components/UI'
import { MOCK_PIPELINE_STAGES } from '../mockData'

/* ── Stage key → index map ────────────────────────────────────────────────── */
const STAGE_IDX = {
  video_ingestion:  0,
  ingestion:        0, // backend alias for video_ingestion
  audio_extraction: 1,
  transcription:    2,
  diarisation:      3,
  nlp_analysis:     4,
  genai_report:     5,
  indexing:         6,
}

const STAGE_ORDER = [
  'video_ingestion',
  'audio_extraction',
  'transcription',
  'diarisation',
  'nlp_analysis',
  'genai_report',
  'indexing',
]

function normalizeStageKey(key) {
  return key === 'ingestion' ? 'video_ingestion' : key
}

function fmt(ms) {
  if (!ms) return null
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function cleanSpeaker(value) {
  return String(value ?? '').trim()
}

function normaliseSpeakerLabel(value, fallback = 'Speaker 0') {
  const text = cleanSpeaker(value)
  if (!text) return fallback

  const match = text.match(/^SPEAKER[_\s-]?0*(\d+)$/i)
  if (match) return `Speaker ${Number(match[1])}`
  if (/^\d+$/.test(text)) return `Speaker ${Number(text)}`
  return text
}

function parseSpeakerNumber(value, fallback = 0) {
  if (typeof value === 'number' && Number.isFinite(value)) return value

  const text = cleanSpeaker(value)
  if (!text) return fallback

  const match = text.match(/^SPEAKER[_\s-]?0*(\d+)$/i)
    || text.match(/^speaker\s+0*(\d+)$/i)
    || text.match(/^0*(\d+)$/)

  return match ? Number(match[1]) : fallback
}

/* ── Word batch → transcript segments helper ─────────────────────────────── */
// Groups a flat words array (from transcript_preview) into speaker segments
// by splitting on pauses > 1.5s or speaker changes.
let _segBuffer  = []
let _segCounter = 0

function wordsToSegment(words) {
  if (!words?.length) return null
  // Use the first word's speaker label if available, else 0
  const speaker = parseSpeakerNumber(words[0]?.speaker, _segCounter % 4)
  const text    = words.map(w => w.word ?? w.text ?? '').join(' ').trim()
  if (!text) return null
  _segCounter++
  return {
    id: _segCounter,
    speaker,
    speakerLabel: normaliseSpeakerLabel(words[0]?.speaker ?? speaker),
    text,
    ts: words[0]?.start ?? 0,
  }
}

export default function PipelineScreen({ sessionId, onComplete }) {
  const [stages, setStages] = useState(() =>
    MOCK_PIPELINE_STAGES.map((s, i) => ({
      ...s,
      status:   'queued',
      progress: 0,
      elapsed:  null,
      detail:   null,
      error:    null,
    }))
  )
  const [previewSegs, setPreviewSegs] = useState([])
  const [eta,         setEta]         = useState(null)
  const previewRef  = useRef(null)
  const stageStart  = useRef({})   // stage → Date.now() when status=running
  const closeSSE    = useRef(null)

  /* ── Update a single stage by key ──────────────────────────────────────── */
  const updateStage = useCallback((key, patch) => {
    const idx = STAGE_IDX[key]
    if (idx == null) return
    setStages(prev => prev.map((s, i) => i === idx ? { ...s, ...patch } : s))
  }, [])

  const markPreviousStagesComplete = useCallback((key) => {
    const normalized = normalizeStageKey(key)
    const currentIdx = STAGE_ORDER.indexOf(normalized)
    if (currentIdx <= 0) return

    setStages(prev =>
      prev.map((s, i) => {
        if (i >= currentIdx) return s
        // If we have progressed to a later stage, prior stages must be done.
        if (s.status === 'queued' || s.status === 'running') {
          return { ...s, status: 'complete', progress: 100, error: null }
        }
        return s
      })
    )
  }, [])

  /* ── Append preview segment, auto-scroll ───────────────────────────────── */
  const addPreviewWords = useCallback((words) => {
    const seg = wordsToSegment(words)
    if (!seg) return
    setPreviewSegs(prev => [...prev, seg])
    setTimeout(() => {
      previewRef.current?.scrollTo({ top: 99999, behavior: 'smooth' })
    }, 50)
  }, [])

  /* ── SSE handler ────────────────────────────────────────────────────────── */
  const handleEvent = useCallback((event) => {
    const type = event.type

    /* heartbeat — ignore */
    if (type === 'heartbeat') return

    /* pipeline_complete — all 7 stages done */
    if (type === 'pipeline_complete') {
      onComplete?.(sessionId)
      return
    }

    /* transcript_preview_catchup — replayed on reconnect */
    if (type === 'transcript_preview_catchup') {
      addPreviewWords(event.words)
      return
    }

    /* transcript_preview — live during ASR */
    if (type === 'transcript_preview') {
      addPreviewWords(event.words)
      return
    }

    /* state_catchup — restore state on page refresh */
    if (type === 'state_catchup') {
      const key = event.stage
      if (!key) return
      if (event.status === 'running' || event.status === 'complete') {
        markPreviousStagesComplete(key)
      }
      updateStage(key, {
        status:   event.status  ?? 'queued',
        progress: parseInt(event.percent ?? 0, 10),
        detail:   event.detail ?? null,
      })
      return
    }

    /* plain stage progress event: { stage, status, percent, detail } */
    const { stage, status, percent, detail } = event
    if (!stage || !status) return

    const now = Date.now()

    if (status === 'ready') {
      markPreviousStagesComplete(stage)
      updateStage(stage, {
        status:   'running',
        progress: Math.max(parseInt(percent ?? 0, 10), 5),
        detail:   detail ?? 'Ready to start',
        error:    null,
      })
      // The report screen owns opening /report/stream. If we wait for a later
      // pipeline_complete event, GenAI generation never starts and progress
      // appears stuck around the gateway's midpoint update.
      if (stage === 'genai_report') {
        onComplete?.(sessionId)
      }
      return
    }

    if (status === 'running') {
      markPreviousStagesComplete(stage)
      if (!stageStart.current[stage]) stageStart.current[stage] = now
      updateStage(stage, {
        status:   'running',
        progress: parseInt(percent ?? 0, 10),
        detail:   detail ?? null,
        error:    null,
      })
      // ETA hint only for transcription (longest stage)
      if (stage === 'transcription' && detail) {
        const match = detail.match(/(\d+)\s*s(?:ec)?/)
        if (match) setEta(parseInt(match[1], 10))
      }
    }

    else if (status === 'complete') {
      const elapsed = stageStart.current[stage]
        ? Math.round(now - stageStart.current[stage])
        : null
      updateStage(stage, { status: 'complete', progress: 100, elapsed })
      if (detail) updateStage(stage, { detail })
      if (stage === 'transcription') setEta(null)
    }

    else if (status === 'failed') {
      updateStage(stage, {
        status: 'failed',
        detail: detail ?? null,
        error:  detail ?? 'Stage failed unexpectedly',
      })
    }
  }, [sessionId, onComplete, updateStage, addPreviewWords, markPreviousStagesComplete])

  /* ── Open real SSE or fall back to demo simulation ──────────────────────── */
  useEffect(() => {
    _segBuffer  = []
    _segCounter = 0

    if (!sessionId) {
      // ── Demo simulation (no real backend) ────────────────────────────────
      const TIMELINE = [
        [400,   { stage:'video_ingestion',  status:'running',  percent:0   }],
        [900,   { stage:'video_ingestion',  status:'running',  percent:70  }],
        [1600,  { stage:'video_ingestion',  status:'complete', percent:100 }],
        [1800,  { stage:'audio_extraction', status:'running',  percent:0   }],
        [3000,  { stage:'audio_extraction', status:'running',  percent:55  }],
        [4200,  { stage:'audio_extraction', status:'complete', percent:100 }],
        [4400,  { stage:'transcription',    status:'running',  percent:0,   detail:'ETA 180s' }],
        [5800,  { stage:'transcription',    status:'running',  percent:20,  detail:'ETA 144s' }],
        [7200,  { stage:'transcription',    status:'running',  percent:42,  detail:'ETA 104s' }],
        [8600,  { stage:'transcription',    status:'running',  percent:64,  detail:'ETA 65s'  }],
        [10000, { stage:'transcription',    status:'running',  percent:83,  detail:'ETA 30s'  }],
        [11400, { stage:'transcription',    status:'complete', percent:100 }],
        [11600, { stage:'diarisation',      status:'running',  percent:0   }],
        [13200, { stage:'diarisation',      status:'running',  percent:48  }],
        [14800, { stage:'diarisation',      status:'complete', percent:100 }],
        [15000, { stage:'nlp_analysis',     status:'running',  percent:0   }],
        [16400, { stage:'nlp_analysis',     status:'running',  percent:60  }],
        [17600, { stage:'nlp_analysis',     status:'complete', percent:100 }],
        [17800, { stage:'genai_report',     status:'running',  percent:0   }],
        [19200, { stage:'genai_report',     status:'running',  percent:55  }],
        [20800, { stage:'genai_report',     status:'complete', percent:100 }],
        [21000, { stage:'indexing',         status:'running',  percent:0   }],
        [21800, { stage:'indexing',         status:'running',  percent:70  }],
        [22600, { stage:'indexing',         status:'complete', percent:100 }],
      ]

      const PREVIEW_TIMELINE = [
        [5000,  ['Alright everyone, let\'s kick off the Q3 roadmap review.']],
        [6500,  ['I want to make sure we align on priorities before the board presentation.']],
        [8000,  ['The mobile redesign has slipped by two sprints — we need to decide today.']],
        [9500,  ['From engineering, the blocker is the new payment SDK.']],
        [11000, ['We only got sandbox access last Thursday, production creds still pending.']],
      ]

      const timers = TIMELINE.map(([delay, event]) =>
        setTimeout(() => handleEvent(event), delay)
      )
      const previewTimers = PREVIEW_TIMELINE.map(([delay, texts], idx) =>
        setTimeout(() => {
          const words = texts[0].split(' ').map((w, i) => ({
            word: w, start: idx * 5 + i * 0.3, speaker: idx % 4
          }))
          handleEvent({ type: 'transcript_preview', words, session_id: 'demo' })
        }, delay)
      )
      const completeTimer = setTimeout(() => {
        handleEvent({ type: 'pipeline_complete', session_id: 'demo' })
      }, 23000)

      return () => {
        timers.forEach(clearTimeout)
        previewTimers.forEach(clearTimeout)
        clearTimeout(completeTimer)
      }
    }

    // ── Real backend SSE ──────────────────────────────────────────────────
    closeSSE.current = openProgressStream(
      sessionId,
      handleEvent,
      (err) => console.warn('[SSE] progress stream error:', err)
    )
    return () => { closeSSE.current?.() }
  }, [sessionId, handleEvent])

  const handleRetry = async (stageKey) => {
    updateStage(stageKey, { status: 'running', progress: 0, error: null })
    // Map frontend stage key → backend retry key (only asr/nlp are retryable)
    const backendKey = stageKey === 'transcription' ? 'asr'
                     : stageKey === 'nlp_analysis'  ? 'nlp'
                     : null
    if (backendKey && sessionId) {
      try { await retryStage(sessionId, backendKey) }
      catch (err) {
        updateStage(stageKey, { status: 'failed', error: err.message })
      }
    }
  }

  const completedN  = stages.filter(s => s.status === 'complete').length
  const runningIdx  = stages.findIndex(s => s.status === 'running')

  return (
    <div className="min-h-screen flex flex-col lg:flex-row pt-14 animate-fade-in">

      {/* ── Left panel: pipeline tracker ─────────────────────────────────── */}
      <div className="lg:w-96 flex-shrink-0 border-r border-bg-border p-6 flex flex-col">
        <div className="mb-6">
          <p className="text-xs text-ink-muted font-mono uppercase tracking-widest mb-1">Processing Pipeline</p>
          <h2 className="font-display text-xl font-bold text-ink">
            {completedN === 7 ? '✓ All stages complete' : `Stage ${Math.min(runningIdx + 1, 7)} of 7`}
          </h2>
          {sessionId && (
            <p className="text-xs text-ink-faint font-mono mt-1 truncate">{sessionId}</p>
          )}
        </div>

        {/* Overall progress bar */}
        <div className="mb-6">
          <div className="flex justify-between text-xs text-ink-muted font-mono mb-1.5">
            <span>{completedN}/7 stages complete</span>
            <span>{Math.round((completedN / 7) * 100)}%</span>
          </div>
          <div className="h-1.5 bg-bg-elevated rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-amber-glow to-teal-glow rounded-full transition-all duration-700"
              style={{ width: `${(completedN / 7) * 100}%` }}
            />
          </div>
        </div>

        {/* Stage list */}
        <div className="relative flex flex-col gap-0">
          <div className="absolute left-5 top-5 bottom-5 w-px bg-bg-border" />

          {stages.map((stage, i) => {
            const isRunning  = stage.status === 'running'
            const isComplete = stage.status === 'complete'
            const isFailed   = stage.status === 'failed'
            const isQueued   = stage.status === 'queued'

            return (
              <div key={stage.key} className="relative flex gap-4 pb-6 last:pb-0">
                {/* Status circle */}
                <div className={`relative z-10 w-10 h-10 flex-shrink-0 rounded-full border-2 flex items-center justify-center transition-all duration-500
                  ${isComplete ? 'bg-good/15 border-good/40'
                  : isRunning  ? 'bg-amber-glow/15 border-amber-glow ring-pulse'
                  : isFailed   ? 'bg-fail/15 border-fail/40'
                  :               'bg-bg-elevated border-bg-border'}`}
                >
                  {isComplete ? (
                    <svg className="w-4 h-4 text-good" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                    </svg>
                  ) : isFailed ? (
                    <svg className="w-4 h-4 text-fail" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  ) : (
                    <span className="text-base leading-none">{stage.icon}</span>
                  )}
                </div>

                <div className="flex-1 min-w-0 pt-1.5">
                  <div className="flex items-center justify-between gap-2 mb-0.5">
                    <p className={`text-sm font-medium ${isRunning ? 'text-amber-glow' : isComplete ? 'text-ink' : isQueued ? 'text-ink-faint' : 'text-ink'}`}>
                      {stage.label}
                    </p>
                    <StatusBadge status={stage.status} />
                  </div>
                  <p className="text-xs text-ink-muted mb-2">{stage.desc}</p>
                  {stage.detail && !isFailed && (
                    <p className="text-xs text-amber-dim mb-2 leading-relaxed">{stage.detail}</p>
                  )}

                  {isRunning && (
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs font-mono text-ink-muted">
                        <span>{stage.progress}%</span>
                        {stage.key === 'transcription' && eta != null && (
                          <span className="text-amber-dim">ETA {eta}s remaining</span>
                        )}
                      </div>
                      <ProgressBar pct={stage.progress} animated />
                    </div>
                  )}

                  {isComplete && stage.elapsed && (
                    <p className="text-xs font-mono text-good/70">↳ {fmt(stage.elapsed)}</p>
                  )}

                  {isFailed && (
                    <div className="mt-1 flex items-center gap-3">
                      <p className="text-xs text-fail truncate flex-1">{stage.error}</p>
                      <button
                        onClick={() => handleRetry(stage.key)}
                        className="text-xs px-2 py-0.5 border border-fail/30 text-fail rounded hover:bg-fail/10 transition-colors whitespace-nowrap"
                      >
                        ↻ Retry
                      </button>
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Right panel: live transcript preview ─────────────────────────── */}
      <div className="flex-1 flex flex-col p-6 overflow-hidden">
        <div className="mb-4">
          <p className="text-xs text-ink-muted font-mono uppercase tracking-widest mb-1">Live Transcript Preview</p>
          <h3 className="font-display text-lg font-semibold text-ink">
            {runningIdx >= 2 && runningIdx <= 3 ? 'Transcription in progress…' : 'Waiting for Speech-to-Text stage…'}
          </h3>
          <p className="text-xs text-ink-faint mt-0.5">Segments appear as Whisper processes the audio</p>
        </div>

        <div
          ref={previewRef}
          className="flex-1 overflow-y-auto space-y-3 pr-2"
          style={{ maxHeight: 'calc(100vh - 220px)' }}
        >
          {previewSegs.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 text-center">
              <div className="w-12 h-12 rounded-xl bg-bg-elevated border border-bg-border flex items-center justify-center mb-3">
                <span className="text-2xl opacity-40">🗣️</span>
              </div>
              <p className="text-ink-muted text-sm max-w-xs">
                Transcript segments will appear here within 60 seconds of the Speech-to-Text stage starting.
              </p>
            </div>
          ) : (
            previewSegs.map((seg, i) => (
              <div
                key={seg.id}
                className="flex gap-3 bg-bg-surface border border-bg-border rounded-xl p-3 animate-slide-in"
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className={`text-xs font-mono speaker-${seg.speaker % 6}`}>
                      {seg.speakerLabel || `Speaker ${seg.speaker}`}
                    </span>
                    {seg.ts > 0 && (
                      <span className="text-xs text-ink-faint font-mono">
                        {Math.floor(seg.ts / 60)}:{String(Math.floor(seg.ts % 60)).padStart(2,'0')}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-ink leading-relaxed">
                    {seg.text}
                    {i === previewSegs.length - 1 && <span className="streaming-cursor" />}
                  </p>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
