/* src/api.js — typed wrappers for every gateway endpoint (main.py)
 *
 * Endpoint map vs main.py
 * POST /api/upload                              → ingest_video()
 * GET  /api/sessions/{id}/progress (SSE)        → session_progress()
 * GET  /api/sessions/{id}/report/stream (SSE)   → stream_report()  ← /stream suffix!
 * GET  /api/sessions/{id}/report                → get_report()
 * GET  /api/sessions/{id}/transcript            → get_transcript()
 * GET  /api/sessions/{id}/transcript/diarised   → get_diarised_transcript()
 * GET  /api/sessions/{id}/transcript/preview    → get_transcript_preview()
 * GET  /api/sessions/{id}/nlp                   → get_nlp_results()
 * GET  /api/sessions/{id}/status                → get_session_status()
 * GET  /api/sessions/{id}                       → get_session_detail()
 * GET  /api/sessions                            → list_sessions()
 * POST /api/sessions/{id}/retry/{stage}         → retry_stage()
 * GET  /api/sessions/{id}/video                 → stream_video() (Range-aware)
 * GET  /api/sessions/{id}/audio                 → stream_audio() (WAV for WaveSurfer)
 * GET  /health                                  → health()
 */

const BASE = import.meta.env.VITE_API_BASE ?? ''

/* ── Upload ──────────────────────────────────────────────────────────────── */
export async function uploadVideo(file, onProgress) {
  const formData = new FormData()
  formData.append('file', file)
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${BASE}/api/upload`)
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.(Math.round((e.loaded / e.total) * 100))
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText)) }
        catch (_) { reject(new Error('Invalid JSON in upload response')) }
      } else {
        reject(new Error(`Upload failed: ${xhr.status} ${xhr.statusText}`))
      }
    }
    xhr.onerror = () => reject(new Error('Network error during upload'))
    xhr.send(formData)
  })
}

/* ── SSE: pipeline progress ──────────────────────────────────────────────── */
// Event shapes from _progress_sse_generator:
//   { type:"state_catchup", stage, status, percent, detail }
//   { type:"transcript_preview_catchup", words:[...] }
//   { type:"transcript_preview", words:[...], session_id }
//   { type:"heartbeat", ts }
//   { type:"pipeline_complete", session_id }
//   plain: { session_id, stage, status, percent, detail }
export function openProgressStream(sessionId, onEvent, onError) {
  const es = new EventSource(`${BASE}/api/sessions/${sessionId}/progress`)
  es.onmessage = (e) => { try { onEvent(JSON.parse(e.data)) } catch (_) {} }
  es.onerror = (err) => { onError?.(err) }
  return () => es.close()
}

/* ── SSE: GenAI report token stream ─────────────────────────────────────── */
// CRITICAL: /report/stream — NOT /report
// Token payload from backend uses { type:"token", content:"..." }.
export function openReportStream(sessionId, onChunk, onDone, onError, onEvent) {
  const es = new EventSource(`${BASE}/api/sessions/${sessionId}/report/stream`)
  es.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data)
      onEvent?.(d)
      if (d.type === 'error') { onError?.(new Error(d.message ?? 'Stream error')); es.close(); return }
      if (d.type === 'done' || d.done === true) { onDone?.(); es.close(); return }
      const chunk = d.content ?? d.token ?? d.text ?? ''
      if (chunk) onChunk(chunk)
    } catch (_) {}
  }
  es.onerror = (err) => { onError?.(err); es.close() }
  return () => es.close()
}

/* ── Completed report JSON ───────────────────────────────────────────────── */
export async function fetchReport(sessionId) {
  const res = await fetch(`${BASE}/api/sessions/${sessionId}/report`)
  if (!res.ok) throw new Error(`Report not ready (${res.status})`)
  return res.json()
}

/* ── Word-level transcript (Stage 3) ────────────────────────────────────── */
export async function fetchTranscript(sessionId) {
  const res = await fetch(`${BASE}/api/sessions/${sessionId}/transcript`)
  if (!res.ok) throw new Error(`Transcript not available (${res.status})`)
  return res.json()
}

/* ── Speaker-diarised transcript (Stage 4) ──────────────────────────────── */
// Returns { segments: [{id, speaker, start, end, text}] }
export async function fetchDiarisedTranscript(sessionId) {
  const res = await fetch(`${BASE}/api/sessions/${sessionId}/transcript/diarised`)
  if (!res.ok) throw new Error(`Diarised transcript not available (${res.status})`)
  return res.json()
}

/* ── Live transcript preview (Stage 3 in-progress) ──────────────────────── */
// Returns { words:[{word,start,end,confidence}], batch_count:N }
export async function fetchTranscriptPreview(sessionId) {
  const res = await fetch(`${BASE}/api/sessions/${sessionId}/transcript/preview`)
  if (!res.ok) return { words: [], batch_count: 0 }
  return res.json()
}

/* ── NLP results (Stage 5) ───────────────────────────────────────────────── */
export async function fetchNlpResults(sessionId) {
  const res = await fetch(`${BASE}/api/sessions/${sessionId}/nlp`)
  if (!res.ok) throw new Error(`NLP results not available (${res.status})`)
  return res.json()
}

/* ── Session status (lightweight poll) ──────────────────────────────────── */
export async function fetchSessionStatus(sessionId) {
  const res = await fetch(`${BASE}/api/sessions/${sessionId}/status`)
  if (!res.ok) throw new Error(`Session not found (${res.status})`)
  return res.json()
}

/* ── Session full detail ─────────────────────────────────────────────────── */
export async function fetchSession(sessionId) {
  const res = await fetch(`${BASE}/api/sessions/${sessionId}`)
  if (!res.ok) throw new Error(`Session not found (${res.status})`)
  return res.json()
}

/* ── Meeting library list ─────────────────────────────────────────────────── */
// Gateway has no server-side ?q= filter — client filters by filename
export async function fetchSessions({ limit = 20, offset = 0 } = {}) {
  const params = new URLSearchParams({ limit, offset })
  const res = await fetch(`${BASE}/api/sessions?${params}`)
  if (!res.ok) throw new Error(`Failed to load sessions (${res.status})`)
  return res.json()   // { sessions:[...], total, limit, offset }
}

/* ── Clear all sessions ──────────────────────────────────────────────────── */
export async function clearAllSessions() {
  const res = await fetch(`${BASE}/api/sessions/clear`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`Failed to clear sessions (${res.status})`)
  return res.json()
}

/* ── Retry a failed stage ────────────────────────────────────────────────── */
// Retryable: asr | nlp  (as defined in main.py RETRYABLE dict)
export async function retryStage(sessionId, stage) {
  const res = await fetch(`${BASE}/api/sessions/${sessionId}/retry/${stage}`, { method: 'POST' })
  if (!res.ok) throw new Error(`Retry failed (${res.status})`)
  return res.json()
}

/* ── Gateway health ──────────────────────────────────────────────────────── */
export async function fetchHealth() {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error('Gateway unhealthy')
  return res.json()   // { status, redis, ingestion, genai }
}

/* ── Media URLs ──────────────────────────────────────────────────────────── */
export const videoUrl = (sessionId) => `${BASE}/api/sessions/${sessionId}/video`
export const audioUrl = (sessionId) => `${BASE}/api/sessions/${sessionId}/audio`
