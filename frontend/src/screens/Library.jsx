import React, { useState, useEffect, useMemo, useCallback } from 'react'
import { fetchSessions, clearAllSessions } from '../api'
import { StatusBadge, Spinner } from '../components/UI'

const PER_PAGE = 10
const SEVERITIES = ['all', 'high', 'medium', 'low']

function fmtDuration(s) {
  const sec = parseInt(s, 10)
  if (!sec || Number.isNaN(sec)) return '—'
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  return h ? `${h}h ${m}m` : `${m}m`
}

function fmtDate(ts) {
  if (!ts) return '—'
  const d = new Date(parseInt(ts, 10) * 1000)
  return Number.isNaN(d) ? '—' : d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
}

export default function LibraryScreen({ onOpenReport }) {
  const [allSessions, setAllSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [total, setTotal] = useState(0)
  const [search, setSearch] = useState('')
  const [severity, setSeverity] = useState('all')
  const [sortKey, setSortKey] = useState('created_at')
  const [sortDir, setSortDir] = useState('desc')
  const [page, setPage] = useState(0)
  const [clearing, setClearing] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchSessions({ limit: 100, offset: 0 })
      setAllSessions(data.sessions ?? [])
      setTotal(data.total ?? 0)
    } catch (err) {
      console.warn('[Library] Failed to load sessions:', err.message)
      setAllSessions([])
      setTotal(0)
      setError('Unable to load meetings right now.')
    } finally {
      setLoading(false)
    }
  }, [])

  const handleClearAll = useCallback(async () => {
    if (!confirm('Are you sure you want to clear all stored meetings? This action cannot be undone.')) return
    setClearing(true)
    try {
      await clearAllSessions()
      await load()
    } catch (err) {
      console.error('[Library] Failed to clear sessions:', err)
      setError('Failed to clear meetings.')
    } finally {
      setClearing(false)
    }
  }, [load])

  useEffect(() => {
    load()
  }, [load])

  const filtered = useMemo(() => {
    let out = allSessions

    if (search.trim()) {
      const q = search.toLowerCase()
      out = out.filter(s =>
        (s.filename ?? '').toLowerCase().includes(q) ||
        (s.session_id ?? '').toLowerCase().includes(q),
      )
    }

    if (severity !== 'all') {
      out = out.filter(s => {
        const pp = parseInt(s.pain_point_count, 10) || 0
        if (severity === 'high') return pp >= 8
        if (severity === 'medium') return pp >= 3 && pp < 8
        if (severity === 'low') return pp < 3
        return true
      })
    }

    out = [...out].sort((a, b) => {
      let va = a[sortKey] ?? ''
      let vb = b[sortKey] ?? ''
      if (!Number.isNaN(Number(va)) && !Number.isNaN(Number(vb))) {
        va = Number(va)
        vb = Number(vb)
      }
      if (sortDir === 'asc') return va > vb ? 1 : -1
      return va < vb ? 1 : -1
    })

    return out
  }, [allSessions, search, severity, sortKey, sortDir])

  const paged = filtered.slice(page * PER_PAGE, (page + 1) * PER_PAGE)
  const totalPages = Math.ceil(filtered.length / PER_PAGE)

  const toggleSort = (key) => {
    if (sortKey === key) setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    else {
      setSortKey(key)
      setSortDir('desc')
    }
    setPage(0)
  }

  const SortIcon = ({ k }) => {
    if (sortKey !== k) return <span className="text-ink-faint ml-1">↕</span>
    return <span className="text-amber-glow ml-1">{sortDir === 'asc' ? '↑' : '↓'}</span>
  }

  return (
    <div className="min-h-screen pt-14 px-6 py-8 animate-fade-in max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <p className="text-xs text-ink-muted font-mono uppercase tracking-widest mb-1">Archive</p>
          <h1 className="font-display text-2xl font-bold text-ink">Meeting Library</h1>
        </div>
        <div className="flex items-center gap-2">
          {error && (
            <span className="text-xs text-warn font-mono px-2 py-1 bg-warn/10 rounded-lg">{error}</span>
          )}
          <button
            onClick={handleClearAll}
            disabled={clearing || loading}
            className="text-xs text-fail hover:text-fail-dim font-mono px-3 py-1.5 bg-bg-elevated border border-bg-border rounded-lg transition-colors disabled:opacity-50"
          >
            {clearing ? 'Clearing...' : 'Clear All'}
          </button>
          <button
            onClick={load}
            disabled={loading}
            className="text-xs text-ink-muted hover:text-ink font-mono px-3 py-1.5 bg-bg-elevated border border-bg-border rounded-lg transition-colors"
          >
            {loading ? '...' : 'Refresh'}
          </button>
          <div className="px-3 py-1.5 bg-bg-elevated border border-bg-border rounded-lg text-xs font-mono text-ink-muted">
            {filtered.length}/{total} meeting{total !== 1 ? 's' : ''}
          </div>
        </div>
      </div>

      <div className="flex flex-wrap gap-3 mb-6">
        <div className="relative flex-1 min-w-48 max-w-sm">
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-ink-faint" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 15.803a7.5 7.5 0 0010.607 10.607z" />
          </svg>
          <input
            type="text"
            placeholder="Search by title or session ID..."
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(0) }}
            className="w-full pl-9 pr-3 py-2 bg-bg-elevated border border-bg-border rounded-lg text-sm text-ink placeholder-ink-faint focus:outline-none focus:border-amber-dim"
          />
        </div>

        <div className="flex items-center gap-1 bg-bg-elevated border border-bg-border rounded-lg p-1">
          {SEVERITIES.map(s => (
            <button
              key={s}
              onClick={() => { setSeverity(s); setPage(0) }}
              className={`px-3 py-1 rounded-md text-xs font-mono font-medium capitalize transition-all ${
                severity === s ? 'bg-amber-glow/15 text-amber-glow' : 'text-ink-muted hover:text-ink'
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-bg-surface border border-bg-border rounded-2xl overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-bg-border text-xs text-ink-muted font-mono uppercase tracking-wider">
              {[
                { key: 'filename', label: 'Title' },
                { key: 'created_at', label: 'Date' },
                { key: 'duration', label: 'Duration' },
                { key: 'num_speakers', label: 'Speakers' },
                { key: 'pain_point_count', label: 'Pain Pts' },
                { key: 'action_item_count', label: 'Actions' },
                { key: 'status', label: 'Status' },
              ].map(col => (
                <th
                  key={col.key}
                  onClick={() => toggleSort(col.key)}
                  className="text-left px-4 py-3 font-medium cursor-pointer hover:text-ink transition-colors whitespace-nowrap"
                >
                  {col.label}<SortIcon k={col.key} />
                </th>
              ))}
              <th className="px-4 py-3 font-medium text-right">Open</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-bg-border">
            {loading ? (
              <tr><td colSpan={8} className="text-center py-16"><div className="flex justify-center"><Spinner /></div></td></tr>
            ) : paged.length === 0 ? (
              <tr><td colSpan={8} className="text-center py-16 text-ink-muted text-sm">No meetings yet.</td></tr>
            ) : (
              paged.map(sess => (
                <tr
                  key={sess.session_id}
                  onClick={() => onOpenReport?.(sess.session_id)}
                  className="hover:bg-bg-elevated cursor-pointer transition-colors group"
                >
                  <td className="px-4 py-3">
                    <p className="text-sm text-ink font-medium truncate max-w-xs group-hover:text-amber-glow transition-colors">
                      {(sess.filename ?? 'Untitled').replace(/\.[^/.]+$/, '').replace(/_/g, ' ')}
                    </p>
                    <p className="text-xs text-ink-faint font-mono mt-0.5 truncate">{sess.session_id}</p>
                  </td>
                  <td className="px-4 py-3 text-sm text-ink-muted font-mono whitespace-nowrap">{fmtDate(sess.created_at)}</td>
                  <td className="px-4 py-3 text-sm text-ink-muted font-mono">{fmtDuration(sess.duration)}</td>
                  <td className="px-4 py-3 text-sm text-ink-muted font-mono">{sess.num_speakers ?? '—'}</td>
                  <td className="px-4 py-3">
                    <span className={`text-sm font-mono font-semibold ${
                      parseInt(sess.pain_point_count, 10) >= 8 ? 'text-fail' :
                      parseInt(sess.pain_point_count, 10) >= 3 ? 'text-warn' :
                      'text-good'
                    }`}
                    >
                      {sess.pain_point_count ?? '—'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-amber-glow font-mono font-semibold">{sess.action_item_count ?? '—'}</td>
                  <td className="px-4 py-3">
                    <StatusBadge status={sess.status ?? sess.pipeline_status ?? 'unknown'} />
                  </td>
                  <td className="px-4 py-3 text-right">
                    <button className="text-xs text-ink-muted hover:text-amber-glow transition-colors px-2 py-1 rounded hover:bg-amber-glow/10">
                      View →
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>

        {totalPages > 1 && (
          <div className="border-t border-bg-border px-4 py-3 flex items-center justify-between">
            <p className="text-xs text-ink-muted font-mono">
              Showing {page * PER_PAGE + 1}-{Math.min((page + 1) * PER_PAGE, filtered.length)} of {filtered.length}
            </p>
            <div className="flex gap-1">
              {Array.from({ length: totalPages }).map((_, i) => (
                <button
                  key={i}
                  onClick={() => setPage(i)}
                  className={`w-7 h-7 rounded-lg text-xs font-mono transition-colors ${
                    i === page ? 'bg-amber-glow/15 text-amber-glow' : 'text-ink-muted hover:bg-bg-elevated'
                  }`}
                >
                  {i + 1}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
