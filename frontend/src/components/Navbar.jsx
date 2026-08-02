import React from 'react'

const NAV = [
  { id: 'upload', label: 'Upload' },
  { id: 'pipeline', label: 'Pipeline' },
  { id: 'report', label: 'Report' },
  { id: 'library', label: 'Library' },
]

export default function Navbar({ screen, onNav, sessionId }) {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-14 flex items-center justify-between px-6 bg-bg-surface/90 backdrop-blur-md border-b border-bg-border">
      <button
        onClick={() => onNav('upload')}
        className="flex items-center gap-2.5 group"
      >
        <div className="w-7 h-7 rounded-lg bg-amber-glow/15 border border-amber-glow/30 flex items-center justify-center">
          <span className="text-amber-glow font-display font-bold text-sm">M</span>
        </div>
        <span className="font-display font-bold text-ink tracking-tight">
          Meeting<span className="text-amber-glow">_Insights</span>
        </span>
      </button>

      <nav className="flex items-center gap-1">
        {NAV.map((item) => (
          <button
            key={item.id}
            onClick={() => onNav(item.id)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              screen === item.id
                ? 'bg-amber-glow/10 text-amber-glow'
                : 'text-ink-muted hover:text-ink hover:bg-bg-elevated'
            }`}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <div className="flex items-center gap-3 text-xs text-ink-muted font-mono">
        {sessionId && (
          <span className="px-2.5 py-1 rounded-lg bg-bg-elevated border border-bg-border">
            Session {sessionId.slice(0, 8)}
          </span>
        )}
        <button
          onClick={() => onNav('observability')}
          className={`px-3 py-1.5 rounded-lg transition-all ${
            screen === 'observability'
              ? 'bg-amber-glow/10 text-amber-glow'
              : 'text-ink-muted hover:text-ink hover:bg-bg-elevated'
          }`}
        >
          Observability
        </button>
        <span className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-good" />
          Gateway
        </span>
      </div>
    </header>
  )
}
