/* src/components/UI.jsx — shared atomic components */
import React from 'react'

/* ── Status badge ────────────────────────────────────────────────────────── */
export function StatusBadge({ status }) {
  const cfg = {
    complete: { bg: 'bg-good/10', text: 'text-good', dot: 'bg-good',   label: 'Complete' },
    running:  { bg: 'bg-amber-glow/10', text: 'text-amber-glow', dot: 'bg-amber-glow ring-pulse', label: 'Running'  },
    queued:   { bg: 'bg-ink-faint/20', text: 'text-ink-muted', dot: 'bg-ink-faint',  label: 'Queued'   },
    failed:   { bg: 'bg-fail/10', text: 'text-fail', dot: 'bg-fail',   label: 'Failed'   },
    skipped:  { bg: 'bg-ink-faint/20', text: 'text-ink-muted', dot: 'bg-ink-faint',  label: 'Skipped'  },
  }
  const c = cfg[status] ?? cfg.queued
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-mono font-medium ${c.bg} ${c.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${c.dot}`} />
      {c.label}
    </span>
  )
}

/* ── Priority / severity badge ───────────────────────────────────────────── */
export function PriorityBadge({ level }) {
  const cfg = {
    critical: { bg: 'bg-fail/15',  text: 'text-fail',       label: 'Critical' },
    high:     { bg: 'bg-fail/10',  text: 'text-fail',       label: 'High'     },
    medium:   { bg: 'bg-warn/10',  text: 'text-warn',       label: 'Medium'   },
    low:      { bg: 'bg-good/10',  text: 'text-good',       label: 'Low'      },
  }
  const c = cfg[level?.toLowerCase()] ?? cfg.low
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-mono font-semibold uppercase tracking-wide ${c.bg} ${c.text}`}>
      {c.label}
    </span>
  )
}

/* ── Section card ─────────────────────────────────────────────────────────── */
export function Card({ className = '', children, ...props }) {
  return (
    <div className={`bg-bg-surface border border-bg-border rounded-xl ${className}`} {...props}>
      {children}
    </div>
  )
}

/* ── Metric tile ─────────────────────────────────────────────────────────── */
export function MetricTile({ label, value, unit, accent = false, sublabel }) {
  return (
    <div className="bg-bg-elevated border border-bg-border rounded-xl p-4">
      <p className="text-xs text-ink-muted font-mono uppercase tracking-widest mb-1">{label}</p>
      <p className={`text-2xl font-display font-bold ${accent ? 'text-amber-glow' : 'text-ink'}`}>
        {value}<span className="text-sm font-normal text-ink-muted ml-1">{unit}</span>
      </p>
      {sublabel && <p className="text-xs text-ink-muted mt-1">{sublabel}</p>}
    </div>
  )
}

/* ── Progress bar ─────────────────────────────────────────────────────────── */
export function ProgressBar({ pct, color = 'amber', animated = false }) {
  const colors = { amber: 'bg-amber-glow', teal: 'bg-teal-glow', good: 'bg-good', fail: 'bg-fail' }
  return (
    <div className="h-1 w-full bg-bg-elevated rounded-full overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-700 ${colors[color]} ${animated ? 'animate-pulse-slow' : ''}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

/* ── Spinner ──────────────────────────────────────────────────────────────── */
export function Spinner({ size = 'md', color = 'amber' }) {
  const sz = { sm: 'w-4 h-4', md: 'w-6 h-6', lg: 'w-8 h-8' }
  const col = { amber: 'border-amber-glow', teal: 'border-teal-glow', white: 'border-ink' }
  return (
    <div className={`${sz[size]} border-2 ${col[color]} border-t-transparent rounded-full animate-spin`} />
  )
}

/* ── Empty state ──────────────────────────────────────────────────────────── */
export function EmptyState({ icon, title, subtitle }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <div className="text-5xl mb-4 opacity-30">{icon}</div>
      <p className="text-ink font-display font-semibold text-lg mb-1">{title}</p>
      <p className="text-ink-muted text-sm max-w-xs">{subtitle}</p>
    </div>
  )
}

/* ── Tooltip wrapper ──────────────────────────────────────────────────────── */
export function Chip({ label, className = '' }) {
  return (
    <span className={`inline-block px-2.5 py-1 bg-bg-elevated border border-bg-border rounded-lg text-xs font-mono text-ink-muted ${className}`}>
      {label}
    </span>
  )
}
