import React, { useState, useEffect, useRef, useCallback } from 'react'
import {
  fetchDiarisedTranscript,
  fetchNlpResults,
  fetchReport,
  openReportStream,
} from '../api'
import { PriorityBadge } from '../components/UI'
import WavePlayer from './WavePlayer'

const REPORT_TITLES = [
  'Executive Summary',
  'Key Decisions Made',
  'Pain Points & Blockers',
  'Action Items',
  'Meeting Sentiment Arc',
  'Key Topics Discussed',
  'Recommended Follow-ups',
]

const SECTION_ALIASES = [
  ['executive summary', 'meeting summary'],
  ['key decisions made', 'key decisions', 'decisions made'],
  ['pain points and blockers', 'pain points blockers', 'pain points', 'blockers'],
  ['action items', 'next actions', 'actions'],
  ['meeting sentiment arc', 'sentiment arc', 'meeting sentiment'],
  ['key topics discussed', 'key topics', 'topics discussed'],
  ['recommended follow ups', 'follow up recommendations', 'followup recommendations', 'recommended next steps'],
]

function cleanText(value) {
  return String(value ?? '')
    .replace(/\r\n/g, '\n')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/__(.*?)__/g, '$1')
    .replace(/\bSPEAKER[_\s-]?0*(\d+)\b/gi, (_, n) => `Speaker ${Number(n)}`)
    .replace(/\bspeaker[_\s-]?0*(\d+)\b/g, (_, n) => `Speaker ${Number(n)}`)
    .trim()
}

function normaliseSpeaker(value) {
  const text = cleanText(value)
  const match = text.match(/^SPEAKER[_\s-]?0*(\d+)$/i)
    || text.match(/^speaker\s+0*(\d+)$/i)
    || text.match(/^0*(\d+)$/)
  return match ? `Speaker ${Number(match[1])}` : text
}

function parseSpeakerNumber(value, fallback = 0) {
  if (typeof value === 'number' && Number.isFinite(value)) return value

  const text = cleanText(value)
  if (!text) return fallback

  const match = text.match(/^SPEAKER[_\s-]?0*(\d+)$/i)
    || text.match(/^speaker\s+0*(\d+)$/i)
    || text.match(/^0*(\d+)$/)

  return match ? Number(match[1]) : fallback
}

function normaliseLevel(value, fallback = 'medium') {
  const text = cleanText(value).toLowerCase()
  if (text.includes('critical')) return 'critical'
  if (text.includes('high')) return 'high'
  if (text.includes('medium')) return 'medium'
  if (text.includes('low')) return 'low'
  return fallback
}

function inferPriority(due, priority, confidence) {
  const text = cleanText(priority).toLowerCase()
  if (text) return normaliseLevel(text, 'low')

  const dueText = cleanText(due).toLowerCase()
  if (/today|eod|end of day|urgent|asap|immediately/.test(dueText)) return 'high'
  if (/tomorrow|this week|next week|next meeting|before/.test(dueText)) return 'medium'
  return Number(confidence) >= 0.75 ? 'medium' : 'low'
}

function buildSectionScaffold() {
  return REPORT_TITLES.map((title, index) => ({
    index,
    title,
    header: `## ${index + 1}. ${title}`,
    content: '',
  }))
}

function trimEmbeddedSections(content) {
  const text = String(content ?? '').replace(/\r\n/g, '\n').trim()
  if (!text) return ''
  const splitAt = text.search(/\n[ \t]{0,3}#{2,6}[ \t]+/)
  return splitAt === -1 ? text : text.slice(0, splitAt).trim()
}

function extractListItems(content) {
  return trimEmbeddedSections(content)
    .split('\n')
    .map((line) => cleanText(line))
    .filter((line) => /^[-*+]\s+/.test(line) || /^\d+\.\s+/.test(line))
    .map((line) => cleanText(line.replace(/^[-*+]\s+/, '').replace(/^\d+\.\s+/, '')))
    .filter(Boolean)
}

function extractMarkdownTableRows(content) {
  const lines = trimEmbeddedSections(content)
    .split('\n')
    .map((line) => cleanText(line))
    .filter(Boolean)

  const headerIndex = lines.findIndex((line, index) => (
    /^\|.+\|$/.test(line) && /^\|[\s:|-]+\|$/.test(lines[index + 1] ?? '')
  ))

  if (headerIndex === -1) return []

  const rows = []
  for (let i = headerIndex + 2; i < lines.length; i += 1) {
    const line = lines[i]
    if (!/^\|.+\|$/.test(line)) break
    const cells = line.split('|').slice(1, -1).map((cell) => cleanText(cell))
    if (cells.some(Boolean)) rows.push(cells)
  }
  return rows
}

function splitLooseTableLine(line) {
  const raw = String(line ?? '').replace(/\r/g, '').trim()
  if (!raw) return []
  if (/^\|[\s:|-]+\|$/.test(raw)) return []
  if (/^\|.+\|$/.test(raw)) return raw.split('|').slice(1, -1).map((cell) => cleanText(cell))
  if (raw.includes('\t')) return raw.split('\t').map((cell) => cleanText(cell))

  const cells = raw.split(/\s{2,}/).map((cell) => cleanText(cell)).filter(Boolean)
  return cells.length >= 2 ? cells : []
}

function looksLikeActionHeader(cells) {
  if (cells.length < 4) return false
  const normalized = cells.slice(0, 4).map((cell) => normaliseHeading(cell))
  return normalized[0] === 'action'
    && normalized[1] === 'owner'
    && ['due', 'due date'].includes(normalized[2])
    && normalized[3] === 'priority'
}

function buildActionItem(action, owner, due, priority, confidence = 0.8) {
  const dueText = cleanText(due) || 'Not specified'
  return {
    action: cleanText(action),
    owner: cleanText(owner) || 'Not specified',
    due: dueText,
    priority: inferPriority(dueText, cleanText(priority), confidence),
  }
}

function extractActionItemsFromContent(content) {
  const markdownRows = extractMarkdownTableRows(content)
  if (markdownRows.length) {
    return markdownRows
      .map((cells) => buildActionItem(cells[0], cells[1], cells[2], cells[3]))
      .filter((item) => item.action)
  }

  const parsedLines = trimEmbeddedSections(content)
    .split('\n')
    .map((line) => splitLooseTableLine(line))
    .filter((cells) => cells.length > 0)

  const headerIndex = parsedLines.findIndex((cells) => looksLikeActionHeader(cells))
  if (headerIndex === -1) return []

  return parsedLines
    .slice(headerIndex + 1)
    .map((cells) => buildActionItem(cells[0], cells[1], cells[2], cells[3]))
    .filter((item) => item.action && normaliseHeading(item.action) !== 'action items')
}

function fmtTime(seconds) {
  if (!isFinite(seconds)) return '0:00'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${String(secs).padStart(2, '0')}`
}

function fmtElapsed(ms) {
  const totalSeconds = Math.max(0, Math.floor((ms || 0) / 1000))
  const mins = Math.floor(totalSeconds / 60)
  const secs = totalSeconds % 60
  return `${mins}:${String(secs).padStart(2, '0')}`
}

function fmtSpeakerTag(value) {
  const text = String(value ?? '').trim()
  const match = text.match(/\b(?:SPEAKER|speaker)[_\s-]?0*(\d+)\b/)
  const speaker = match ? Number(match[1]) : parseSpeakerNumber(value, 0)
  return `speaker_${String(speaker).padStart(2, '0')}`
}

function normaliseReportSections(raw) {
  const base = buildSectionScaffold()
  const sections = Array.isArray(raw?.sections) ? raw.sections : []

  sections.forEach((section, i) => {
    const index = Number.isFinite(Number(section?.index)) ? Number(section.index) : i
    if (index < 0 || index >= base.length) return
    base[index] = {
      index,
      title: cleanText(section?.title) || base[index].title,
      header: cleanText(section?.header) || base[index].header,
      content: cleanText(section?.content),
    }
  })

  return base
}

function sectionLabel(section) {
  const header = cleanText(section?.header).replace(/^#+\s*/, '')
  return header || cleanText(section?.title)
}

function normaliseHeading(value) {
  return cleanText(value)
    .replace(/^#+\s*/, '')
    .replace(/^[0-9]+[\.)]\s*/, '')
    .replace(/&/g, ' and ')
    .replace(/[-_]+/g, ' ')
    .replace(/[^a-zA-Z0-9 ]+/g, '')
    .replace(/\s+/g, ' ')
    .toLowerCase()
    .trim()
}

function resolveSectionIndex(heading) {
  const normalized = normaliseHeading(heading)
  return SECTION_ALIASES.findIndex((aliases) => aliases.includes(normalized))
}

function parseStreamedReportSections(rawText) {
  const text = String(rawText ?? '').replace(/\r\n/g, '\n')
  const sections = buildSectionScaffold()

  if (!text.trim()) return sections

  const headingRe = /(^|\n)[ \t]{0,3}#{2,6}[ \t]+(.+?)\s*(?=\n|$)/gm
  const matches = []
  let match

  while ((match = headingRe.exec(text))) {
    const index = resolveSectionIndex(match[2])
    if (index === -1 || matches.some((item) => item.index === index)) continue

    matches.push({
      index,
      start: match.index + match[1].length,
      contentStart: headingRe.lastIndex,
    })
  }

  if (!matches.length) {
    sections[0].content = text.trim()
    return sections
  }

  matches.forEach((item, i) => {
    const next = matches[i + 1]
    sections[item.index] = {
      ...sections[item.index],
      content: text.slice(item.contentStart, next?.start ?? text.length).trim(),
    }
  })

  return sections
}

function ReportContent({ content, streaming = false }) {
  const blocks = cleanText(content)
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean)

  if (!blocks.length) {
    return (
      <p className={`text-sm text-ink/90 leading-relaxed whitespace-pre-wrap ${streaming ? 'streaming-cursor' : ''}`}>
        {content || <span className="text-ink-faint italic">Waiting for report stream...</span>}
      </p>
    )
  }

  return (
    <div className={streaming ? 'streaming-cursor' : ''}>
      <div className="space-y-3 text-sm text-ink/90 leading-relaxed">
        {blocks.map((block, blockIndex) => {
          const lines = block.split('\n').map((line) => line.trim()).filter(Boolean)
          const isBulletList = lines.length > 0 && lines.every((line) => /^[-*]\s+/.test(line))
          const isOrderedList = lines.length > 0 && lines.every((line) => /^\d+\.\s+/.test(line))
          const isTable = lines.length >= 2
            && /^\|.+\|$/.test(lines[0])
            && /^\|[\s:|-]+\|$/.test(lines[1])

          if (isBulletList) {
            return (
              <ul key={blockIndex} className="space-y-2">
                {lines.map((line, lineIndex) => (
                  <li key={lineIndex} className="flex items-start gap-2">
                    <span className="mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-amber-glow" />
                    <span>{line.replace(/^[-*]\s+/, '')}</span>
                  </li>
                ))}
              </ul>
            )
          }

          if (isOrderedList) {
            return (
              <ol key={blockIndex} className="space-y-2">
                {lines.map((line, lineIndex) => {
                  const ordered = line.match(/^(\d+)\.\s+(.*)$/)
                  return (
                    <li key={lineIndex} className="flex items-start gap-2">
                      <span className="mt-0.5 flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-teal-glow/15 text-xs font-bold text-teal-glow">
                        {ordered?.[1] ?? lineIndex + 1}
                      </span>
                      <span>{ordered?.[2] ?? line}</span>
                    </li>
                  )
                })}
              </ol>
            )
          }

          if (isTable) {
            const cells = lines.map((line) => line.split('|').slice(1, -1).map((cell) => cell.trim()))
            const headers = cells[0] ?? []
            const rows = cells.slice(2).filter((row) => row.some(Boolean))

            return (
              <div key={blockIndex} className="overflow-x-auto rounded-lg border border-bg-border bg-bg-elevated/60 p-3">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-bg-border text-ink-muted font-mono uppercase tracking-wider">
                      {headers.map((header, i) => (
                        <th key={i} className="pb-2 pr-3 text-left font-medium">{header}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-bg-border">
                    {rows.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {row.map((cell, cellIndex) => (
                          <td key={cellIndex} className="py-2 pr-3 text-sm text-ink/90 align-top">{cell}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          }

          return (
            <p key={blockIndex} className="whitespace-pre-wrap">
              {block}
            </p>
          )
        })}
      </div>
    </div>
  )
}

function SectionPlaceholder({ children }) {
  return <p className="text-sm italic text-ink-faint">{children}</p>
}

function ActionItemsTable({ items }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-bg-border bg-bg-elevated/60 p-3">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-bg-border text-left text-xs font-mono uppercase tracking-wider text-ink-muted">
            <th className="pb-2 pr-3 font-medium">Action</th>
            <th className="pb-2 pr-3 font-medium">Owner</th>
            <th className="pb-2 pr-3 font-medium">Due Date</th>
            <th className="pb-2 font-medium">Priority</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-bg-border">
          {items.map((item, index) => (
            <tr key={index}>
              <td className="py-3 pr-3 align-top text-ink/90">{item.action || 'Not specified'}</td>
              <td className="py-3 pr-3 align-top text-ink-muted font-mono">{item.owner || 'Not specified'}</td>
              <td className="py-3 pr-3 align-top text-ink-muted font-mono">{item.due || 'Not specified'}</td>
              <td className="py-3 align-top">
                <PriorityBadge level={item.priority || 'low'} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function normaliseSegments(raw) {
  const segs = raw?.segments ?? raw?.words ?? raw
  if (!Array.isArray(segs) || !segs.length) return null

  if ('word' in (segs[0] ?? {})) {
    const out = []
    let current = null

    segs.forEach((word) => {
      const speaker = parseSpeakerNumber(word.speaker, 0)
      if (!current || current.speaker !== speaker) {
        if (current) out.push(current)
        current = {
          id: out.length,
          speaker,
          speakerTag: fmtSpeakerTag(word.speaker ?? speaker),
          speakerLabel: normaliseSpeaker(word.speaker ?? speaker),
          start: word.start,
          end: word.end,
          text: word.word,
        }
        return
      }

      current.text += ` ${word.word ?? ''}`
      current.end = word.end
    })

    if (current) out.push(current)
    return out
  }

  return segs.map((segment, i) => {
  const speakerNum = parseSpeakerNumber(
    segment.speaker ?? segment.speaker_id ?? segment.label,
    0
  )

  return {
    id: Number.isFinite(Number(segment.id)) ? Number(segment.id) : i,
    speaker: speakerNum,
    speakerTag: `Speaker ${speakerNum}`,
    speakerLabel: `Speaker ${speakerNum}`,
    start: segment.start,
    end: segment.end,
    text: segment.text,
  }
  })
}

function normaliseNlp(raw) {
  if (!raw) return null

  const painPointsRaw = raw.pain_points?.pain_points ?? raw.pain_points ?? []
  const actionItemsRaw = raw.action_items?.action_items ?? raw.action_items ?? []
  const topicsRaw = raw.topics?.topics ?? raw.topics?.keyphrases ?? raw.topics ?? []
  const decisionsRaw = raw.decisions?.decisions ?? raw.decisions ?? []

  const pain_points = painPointsRaw
    .map((item) => ({
      text: cleanText(item?.pain_point || item?.text || item?.description || item?.quote),
      severity: normaliseLevel(item?.severity, 'medium'),
      speaker: normaliseSpeaker(item?.speaker),
      quote: cleanText(item?.quote),
      category: cleanText(item?.category),
    }))
    .filter((item) => item.text)

  const action_items = actionItemsRaw
    .map((item) => {
      const due = cleanText(item?.deadline || item?.due)
      return {
        action: cleanText(item?.action || item?.task || item?.text),
        owner: normaliseSpeaker(item?.owner_name || item?.owner),
        due,
        priority: inferPriority(due, item?.priority, item?.confidence),
        quote: cleanText(item?.quote),
      }
    })
    .filter((item) => item.action)

  const topics = topicsRaw
    .map((item) => cleanText(typeof item === 'string' ? item : item?.phrase || item?.text))
    .filter(Boolean)

  const key_decisions = decisionsRaw
    .map((item) => cleanText(item?.decision || item?.text || item?.quote || item?.summary))
    .filter(Boolean)

  return { pain_points, action_items, topics, key_decisions }
}

function Section({ title, badge, defaultOpen = false, children }) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className="border-b border-bg-border last:border-0">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-bg-elevated/50 transition-colors text-left"
      >
        <span className="font-display font-semibold text-sm text-ink">{title}</span>
        <div className="flex items-center gap-2">
          {badge}
          <svg
            className={`w-4 h-4 text-ink-muted transition-transform ${open ? 'rotate-180' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>
      {open && <div className="px-4 pb-4">{children}</div>}
    </div>
  )
}

export default function ReportScreen({ sessionId }) {
  const [transcript, setTranscript] = useState([])
  const [nlp, setNlp] = useState({})
  const [reportSections, setReportSections] = useState(buildSectionScaffold())
  const [streaming, setStreaming] = useState(false)
  const [reportDone, setReportDone] = useState(false)
  const [reportElapsedMs, setReportElapsedMs] = useState(0)
  const [activeSectionIdx, setActiveSectionIdx] = useState(0)
  const [cursorVisible, setCursorVisible] = useState(false)
  const [searchQ, setSearchQ] = useState('')
  const [searchHits, setSearchHits] = useState([])
  const [seekTime, setSeekTime] = useState(null)
  const [activeSegId, setActiveSegId] = useState(0)
  const [playTime, setPlayTime] = useState(0)

  const transcriptRef = useRef(null)
  const closeStreamRef = useRef(null)
  const retryTimerRef = useRef(null)
  const flushTimerRef = useRef(null)
  const cursorTimerRef = useRef(null)
  const retryCountRef = useRef(0)
  const streamVisibleRef = useRef('')
  const streamPendingRef = useRef('')
  const reportStartedAtRef = useRef(null)
  const isDemo = !sessionId

  useEffect(() => {
    if (isDemo) {
      setTranscript([])
      return
    }

    fetchDiarisedTranscript(sessionId)
      .then((raw) => {
        const segs = normaliseSegments(raw)
        if (segs?.length) setTranscript(segs)
      })
      .catch(() => {})
  }, [sessionId, isDemo])

  useEffect(() => {
    if (isDemo) {
      setNlp({})
      return
    }

    fetchNlpResults(sessionId)
      .then((raw) => setNlp(normaliseNlp(raw) ?? {}))
      .catch(() => setNlp({}))
  }, [sessionId, isDemo])

  useEffect(() => {
    if (isDemo) {
      setReportSections(buildSectionScaffold())
      setStreaming(false)
      setReportDone(false)
      setReportElapsedMs(0)
      reportStartedAtRef.current = null
      return undefined
    }

    let cancelled = false

    const clearPendingRetry = () => {
      if (retryTimerRef.current) {
        clearTimeout(retryTimerRef.current)
        retryTimerRef.current = null
      }
    }

    const clearPendingFlush = () => {
      if (flushTimerRef.current) {
        clearTimeout(flushTimerRef.current)
        flushTimerRef.current = null
      }
    }

    const pulseCursor = () => {
      setCursorVisible(true)
      if (cursorTimerRef.current) clearTimeout(cursorTimerRef.current)
      cursorTimerRef.current = setTimeout(() => {
        cursorTimerRef.current = null
        setCursorVisible(false)
      }, 900)
    }

    const applyCachedReport = async () => {
      try {
        const data = await fetchReport(sessionId)
        const sections = normaliseReportSections(data)
        setReportSections(sections)
        setStreaming(false)
        setReportDone(true)
        return true
      } catch (_) {
        return false
      }
    }

    const scheduleReconnect = () => {
      if (cancelled || reportDone) return
      clearPendingRetry()
      const delay = Math.min(8000, 1000 * (2 ** Math.min(retryCountRef.current, 3)))
      retryCountRef.current += 1
      retryTimerRef.current = setTimeout(async () => {
        if (cancelled) return
        if (await applyCachedReport()) return
        connect()
      }, delay)
    }

    const connect = () => {
      if (cancelled) return
      clearPendingRetry()
      closeStreamRef.current?.()
      setStreaming(true)
      reportStartedAtRef.current = Date.now()
      setReportElapsedMs(0)
      streamVisibleRef.current = ''
      streamPendingRef.current = ''

      const flushStream = (force = false) => {
        const pending = streamPendingRef.current
        if (!pending) return

        const flushAt = force ? pending.length : pending.lastIndexOf('\n') + 1
        if (flushAt <= 0) return

        streamVisibleRef.current += pending.slice(0, flushAt)
        streamPendingRef.current = pending.slice(flushAt)
        setReportSections(parseStreamedReportSections(streamVisibleRef.current))
      }

      const schedulePartialFlush = () => {
        if (flushTimerRef.current) return
        flushTimerRef.current = setTimeout(() => {
          flushTimerRef.current = null
          flushStream(true)
        }, 60)
      }

      closeStreamRef.current = openReportStream(
        sessionId,
        (chunk) => {
          retryCountRef.current = 0
          streamPendingRef.current += chunk
          pulseCursor()
          flushStream(streamPendingRef.current.length >= 48)
          schedulePartialFlush()
        },
        async () => {
          retryCountRef.current = 0
          clearPendingFlush()
          if (cursorTimerRef.current) {
            clearTimeout(cursorTimerRef.current)
            cursorTimerRef.current = null
          }
          setCursorVisible(false)
          flushStream(true)
          setStreaming(false)
          setReportDone(true)
          await applyCachedReport()
        },
        async (err) => {
          console.warn('[Report stream]', err)
          if (cancelled) return
          clearPendingFlush()
          if (cursorTimerRef.current) {
            clearTimeout(cursorTimerRef.current)
            cursorTimerRef.current = null
          }
          setCursorVisible(false)
          setStreaming(false)
          if (await applyCachedReport()) return
          scheduleReconnect()
        },
        (event) => {
          if (event?.type === 'section' && Number.isFinite(Number(event.index))) {
            setActiveSectionIdx(Number(event.index))
          }
        }
      )
    }

    retryCountRef.current = 0
    setReportDone(false)
    setActiveSectionIdx(0)
    setCursorVisible(false)
    setReportElapsedMs(0)
    setReportSections(buildSectionScaffold())
    streamVisibleRef.current = ''
    streamPendingRef.current = ''

    connect()

    return () => {
      cancelled = true
      clearPendingRetry()
      if (flushTimerRef.current) {
        clearTimeout(flushTimerRef.current)
        flushTimerRef.current = null
      }
      if (cursorTimerRef.current) {
        clearTimeout(cursorTimerRef.current)
        cursorTimerRef.current = null
      }
      closeStreamRef.current?.()
    }
  }, [sessionId, isDemo])

  useEffect(() => {
    if (!streaming || !reportStartedAtRef.current) return undefined

    const tick = () => {
      setReportElapsedMs(Date.now() - reportStartedAtRef.current)
    }

    tick()
    const interval = setInterval(tick, 500)
    return () => clearInterval(interval)
  }, [streaming])

  useEffect(() => {
    const active = transcript.find((segment) => playTime >= segment.start && playTime <= segment.end)
    if (active && active.id !== activeSegId) {
      setActiveSegId(active.id)
      document.getElementById(`seg-${active.id}`)?.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      })
    }
  }, [playTime, transcript, activeSegId])

  const handleSearch = useCallback(() => {
    const query = searchQ.trim()
    if (!query) {
      setSearchHits([])
      return
    }

    const hits = transcript.filter((segment) => segment.text?.toLowerCase().includes(query.toLowerCase()))
    setSearchHits(hits.map((segment) => segment.id))

    if (hits.length > 0) {
      document.getElementById(`seg-${hits[0].id}`)?.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      })
      setSeekTime(hits[0].start)
    }
  }, [searchQ, transcript])

  const report = {
    key_decisions: [],
    pain_points: [],
    action_items: [],
    topics: [],
    ...(nlp ?? {}),
  }

  const summarySection = reportSections[0] ?? { title: 'Executive Summary', header: '## 1. Executive Summary', content: '' }
  const decisionsSection = reportSections[1] ?? { title: 'Key Decisions Made', header: '## 2. Key Decisions Made', content: '' }
  const painPointsSection = reportSections[2] ?? { title: 'Pain Points & Blockers', header: '## 3. Pain Points & Blockers', content: '' }
  const actionItemsSection = reportSections[3] ?? { title: 'Action Items', header: '## 4. Action Items', content: '' }
  const sentimentSection = reportSections[4] ?? { title: 'Meeting Sentiment Arc', header: '## 5. Meeting Sentiment Arc', content: '' }
  const topicsSection = reportSections[5] ?? { title: 'Key Topics Discussed', header: '## 6. Key Topics Discussed', content: '' }
  const followupsSection = reportSections[6] ?? { title: 'Recommended Follow-ups', header: '## 7. Recommended Follow-ups', content: '' }

  const fallbackDecisions = extractListItems(decisionsSection.content)
  const fallbackTopics = extractListItems(topicsSection.content)
  const fallbackFollowups = extractListItems(followupsSection.content)

  const isLiveReport = streaming && !reportDone
  const summaryContent = trimEmbeddedSections(summarySection.content)
  const decisionsContent = trimEmbeddedSections(decisionsSection.content)
  const painPointsContent = trimEmbeddedSections(painPointsSection.content)
  const actionItemsContent = trimEmbeddedSections(actionItemsSection.content)
  const sentimentContent = trimEmbeddedSections(sentimentSection.content)
  const topicsContent = trimEmbeddedSections(topicsSection.content)
  const followupsContent = trimEmbeddedSections(followupsSection.content)

  const displayDecisions = report.key_decisions?.length ? report.key_decisions : fallbackDecisions
  const displayTopics = reportDone && report.topics?.length ? report.topics : fallbackTopics
  const displayFollowups = reportDone ? fallbackFollowups : []
  const isSectionStreaming = (index) => isLiveReport && activeSectionIdx === index && cursorVisible
  const displayActionItems = report.action_items.length ? report.action_items : extractActionItemsFromContent(actionItemsContent)
  const decisionsCount = displayDecisions.length
  const painPointsCount = report.pain_points.length || extractListItems(painPointsSection.content).length
  const actionItemsCount = displayActionItems.length || extractListItems(actionItemsSection.content).length
  const topicsCount = displayTopics.length
  const sectionHasStructuredData = (index) => (
    (index === 1 && displayDecisions.length > 0)
    || (index === 2 && report.pain_points.length > 0)
    || (index === 3 && displayActionItems.length > 0)
  )
  const sectionStarted = (index, content = '') => (
    reportDone
    || activeSectionIdx >= index
    || Boolean(content)
    || (Boolean(summaryContent) && sectionHasStructuredData(index))
  )

  return (
    <div className="h-screen flex flex-col pt-14 overflow-hidden">
      <div className="flex-1 flex overflow-hidden">
        <div className="w-[42%] flex-shrink-0 flex flex-col border-r border-bg-border overflow-hidden">
          <div className="p-4 border-b border-bg-border bg-bg-surface/50">
            {sessionId ? (
              <WavePlayer
                sessionId={sessionId}
                currentTime={seekTime}
                onTimeUpdate={setPlayTime}
                onSeek={setSeekTime}
              />
            ) : (
              <div className="rounded-xl border border-dashed border-bg-border bg-bg-elevated/40 px-4 py-10 text-center text-sm text-ink-muted">
                Upload a meeting to view waveform and transcript.
              </div>
            )}
          </div>

          <div className="px-4 py-3 border-b border-bg-border">
            <div className="relative">
              <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-ink-faint" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 15.803a7.5 7.5 0 0010.607 10.607z" />
              </svg>
              <input
                type="text"
                placeholder="Search transcript... (Enter to jump)"
                value={searchQ}
                onChange={(e) => setSearchQ(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                className="w-full pl-9 pr-16 py-2 bg-bg-elevated border border-bg-border rounded-lg text-sm text-ink placeholder-ink-faint focus:outline-none focus:border-amber-dim transition-colors"
              />
              {searchHits.length > 0 && (
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs font-mono text-amber-glow">
                  {searchHits.length} hit{searchHits.length !== 1 ? 's' : ''}
                </span>
              )}
            </div>
          </div>

          <div ref={transcriptRef} className="flex-1 overflow-y-auto p-4 space-y-2">
            {transcript.length === 0 && (
              <div className="rounded-xl border border-dashed border-bg-border bg-bg-elevated/30 px-4 py-10 text-center text-sm text-ink-muted">
                {sessionId ? 'Transcript is not ready yet.' : 'No transcript yet.'}
              </div>
            )}

            {transcript.map((segment) => {
              const speakerTag = segment.speakerTag || `Speaker ${segment.speaker}`
              const isActive = segment.id === activeSegId
              const isHit = searchHits.includes(segment.id)

              return (
                <div
                  key={segment.id}
                  id={`seg-${segment.id}`}
                  onClick={() => setSeekTime(segment.start)}
                  className={`flex items-start gap-3 px-3 py-2.5 rounded-lg cursor-pointer transition-all border ${
                    isActive ? 'transcript-active' : 'border-transparent hover:bg-bg-elevated'
                  } ${isHit ? 'ring-1 ring-amber-glow/40' : ''}`}
                >
                  <div className="flex-shrink-0 pt-0.5">
                    <span className="inline-flex min-w-[84px] items-center justify-center rounded-md border border-amber-glow/25 bg-amber-glow/12 px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-amber-glow font-mono">
                      {speakerTag}
                    </span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5 flex-wrap">
                      <span className="text-xs text-ink-faint font-mono">{fmtTime(segment.start)}</span>
                    </div>
                    <p className={`text-sm leading-relaxed ${isActive ? 'text-ink' : 'text-ink/80'}`}>
                      {searchQ && segment.text?.toLowerCase().includes(searchQ.toLowerCase())
                        ? segment.text.split(new RegExp(`(${searchQ})`, 'gi')).map((part, i) => (
                            part.toLowerCase() === searchQ.toLowerCase()
                              ? <mark key={i} className="bg-amber-glow/25 text-amber-glow rounded px-0.5">{part}</mark>
                              : part
                          ))
                        : segment.text}
                    </p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto">
            <div className="px-6 py-4 border-b border-bg-border bg-bg-surface/50 flex items-center justify-between sticky top-0 z-10 backdrop-blur-sm">
              <div>
                <p className="text-xs text-ink-muted font-mono uppercase tracking-widest">AI Intelligence Report</p>
                <h2 className="font-display font-bold text-ink text-lg">Meeting Analysis</h2>
              </div>
              {streaming && (
                <div className="flex items-center gap-3 text-xs text-amber-glow font-mono">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-glow ring-pulse" />
                  <span>Streaming...</span>
                  <span className="rounded-full border border-amber-glow/20 bg-amber-glow/10 px-2 py-0.5 text-[11px]">
                    {fmtElapsed(reportElapsedMs)}
                  </span>
                </div>
              )}
            </div>

            <div className="divide-y divide-bg-border">
              <Section title={sectionLabel(summarySection)} defaultOpen>
                <ReportContent content={summaryContent} streaming={isSectionStreaming(0)} />
              </Section>

              <Section
                title={sectionLabel(decisionsSection)}
                defaultOpen
                badge={
                  <span className="text-xs font-mono bg-teal-glow/10 text-teal-glow px-2 py-0.5 rounded-full">
                    {decisionsCount}
                  </span>
                }
              >
                {!sectionStarted(1, decisionsContent) ? (
                  <SectionPlaceholder>Waiting for key decisions...</SectionPlaceholder>
                ) : displayDecisions.length > 0 ? (
                  <ul className="space-y-2">
                    {displayDecisions.map((decision, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-ink/90">
                        <span>{decision}</span>
                      </li>
                    ))}
                  </ul>
                ) : isLiveReport && decisionsContent ? (
                  <ReportContent content={decisionsContent} streaming={isSectionStreaming(1)} />
                ) : decisionsContent ? (
                  <ReportContent content={decisionsContent} />
                ) : (
                  <SectionPlaceholder>No confirmed decisions detected.</SectionPlaceholder>
                )}
              </Section>

              <Section
                title={sectionLabel(painPointsSection)}
                defaultOpen
                badge={
                  <span className="text-xs font-mono bg-fail/10 text-fail px-2 py-0.5 rounded-full">
                    {painPointsCount}
                  </span>
                }
              >
                {!sectionStarted(2, painPointsContent) ? (
                  <SectionPlaceholder>Waiting for pain points and blockers...</SectionPlaceholder>
                ) : report.pain_points.length > 0 ? (
                  <div className="space-y-2">
                    {report.pain_points.map((painPoint, i) => (
                      <div key={i} className="flex items-start gap-2.5 rounded-lg bg-bg-elevated p-3">
                        <PriorityBadge level={painPoint.severity} />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm leading-snug text-ink/90">{painPoint.text}</p>
                          {(painPoint.speaker || painPoint.category) && (
                            <p className="mt-1 text-xs font-mono text-ink-muted">
                              {[painPoint.speaker, painPoint.category].filter(Boolean).join(' | ')}
                            </p>
                          )}
                          {painPoint.quote && painPoint.quote !== painPoint.text && (
                            <p className="mt-1 text-xs italic text-ink-muted">"{painPoint.quote}"</p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : painPointsContent ? (
                  <ReportContent content={painPointsContent} streaming={isSectionStreaming(2)} />
                ) : (
                  <SectionPlaceholder>No pain points detected.</SectionPlaceholder>
                )}
              </Section>

              <Section
                title={sectionLabel(actionItemsSection)}
                defaultOpen
                badge={
                  <span className="text-xs font-mono bg-amber-glow/10 text-amber-glow px-2 py-0.5 rounded-full">
                    {actionItemsCount}
                  </span>
                }
              >
                {!sectionStarted(3, actionItemsContent) ? (
                  <SectionPlaceholder>Waiting for action items...</SectionPlaceholder>
                ) : displayActionItems.length > 0 ? (
                  <ActionItemsTable items={displayActionItems} />
                ) : actionItemsContent ? (
                  <ReportContent content={actionItemsContent} streaming={isSectionStreaming(3)} />
                ) : (
                  <SectionPlaceholder>No action items detected.</SectionPlaceholder>
                )}
              </Section>

              <Section title={sectionLabel(sentimentSection)} defaultOpen>
                {!sectionStarted(4, sentimentContent) ? (
                  <SectionPlaceholder>Waiting for sentiment arc...</SectionPlaceholder>
                ) : sentimentContent ? (
                  <ReportContent content={sentimentContent} streaming={isSectionStreaming(4)} />
                ) : (
                  <SectionPlaceholder>No meeting sentiment arc generated.</SectionPlaceholder>
                )}
              </Section>

              <Section
                title={sectionLabel(topicsSection)}
                defaultOpen
                badge={
                  <span className="text-xs font-mono bg-teal-glow/10 text-teal-glow px-2 py-0.5 rounded-full">
                    {topicsCount}
                  </span>
                }
              >
                {!sectionStarted(5, topicsContent) ? (
                  <SectionPlaceholder>Waiting for key topics...</SectionPlaceholder>
                ) : topicsContent ? (
                  <ReportContent content={topicsContent} streaming={isSectionStreaming(5)} />
                ) : displayTopics.length > 0 ? (
                  <ol className="space-y-2">
                    {displayTopics.map((topic, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-ink/90">
                        <span className="mt-0.5 flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-teal-glow/15 text-xs font-bold text-teal-glow">
                          {i + 1}
                        </span>
                        <span>{topic}</span>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <SectionPlaceholder>No key topics detected.</SectionPlaceholder>
                )}
              </Section>

              <Section title={sectionLabel(followupsSection)} defaultOpen>
                {!sectionStarted(6, followupsContent) ? (
                  <SectionPlaceholder>Waiting for recommended follow-ups...</SectionPlaceholder>
                ) : followupsContent ? (
                  <ReportContent content={followupsContent} streaming={isSectionStreaming(6)} />
                ) : displayFollowups.length > 0 ? (
                  <ul className="space-y-2">
                    {displayFollowups.map((followup, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-ink/80">
                        <span className="mt-0.5 text-amber-glow">-&gt;</span>
                        <span>{followup}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <SectionPlaceholder>No recommended follow-ups yet.</SectionPlaceholder>
                )}
              </Section>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
