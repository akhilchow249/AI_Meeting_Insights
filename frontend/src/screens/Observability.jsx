/* src/screens/Observability.jsx */
import React, { useState } from 'react'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Cell
} from 'recharts'
import { MetricTile } from '../components/UI'
import { MOCK_METRICS } from '../mockData'

const TIP_STYLE = {
  background: '#0e1527', border: '1px solid #1a2540',
  borderRadius: 8, fontSize: 11, color: '#f0f4ff'
}

function SectionHeader({ title, subtitle }) {
  return (
    <div className="mb-4">
      <h2 className="font-display font-bold text-ink text-base">{title}</h2>
      {subtitle && <p className="text-xs text-ink-muted mt-0.5">{subtitle}</p>}
    </div>
  )
}

export default function ObservabilityScreen() {
  const [tab, setTab] = useState('latency')

  const failData = MOCK_METRICS.failureRates.map(d => ({
    ...d,
    rate: parseFloat(((d.failures / d.total) * 100).toFixed(1))
  }))

  return (
    <div className="min-h-screen pt-14 px-6 py-8 animate-fade-in max-w-7xl mx-auto">

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <p className="text-xs text-ink-muted font-mono uppercase tracking-widest mb-1">Metrics · Prometheus</p>
          <h1 className="font-display text-2xl font-bold text-ink">Observability Dashboard</h1>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-good">
          <span className="w-1.5 h-1.5 rounded-full bg-good" />
          Scraping /metrics · 15s interval
        </div>
      </div>

      {/* KPI row — all 10 Prometheus metrics from metrics.py */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-8">
        <MetricTile label="Audio Extract p95"  value="22.0"  unit="s"   sublabel="Alert > 60s" />
        <MetricTile label="Transcription RTF"  value="1.4"   unit="×"   sublabel="Alert > 3×" accent />
        <MetricTile label="ASR Confidence"     value="0.87"  unit="avg" sublabel="Alert < 0.75" />
        <MetricTile label="Diarisation p95"    value="67"    unit="s"   sublabel="Alert > 120s" />
        <MetricTile label="Speakers Detected"  value="4.2"   unit="avg" sublabel="Last 50 meetings" />
        <MetricTile label="Pain Points Total"  value="312"   unit=""    sublabel="All time" accent />
        <MetricTile label="NLP p95"            value="24"    unit="s"   sublabel="Alert > 30s" />
        <MetricTile label="GenAI First Token"  value="3.2"   unit="s"   sublabel="Alert > 5s" />
        <MetricTile label="Pipeline Failures"  value="2.3"   unit="%"   sublabel="Alert > 5%" />
        <MetricTile label="Search p95"         value="180"   unit="ms"  sublabel="Alert > 500ms" />
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 bg-bg-elevated border border-bg-border rounded-xl p-1 mb-6 w-fit">
        {[
          { k: 'latency',    label: 'Stage Latency' },
          { k: 'confidence', label: 'ASR Confidence' },
          { k: 'failures',   label: 'Failure Rates' },
          { k: 'queue',      label: 'Queue Depth' },
        ].map(t => (
          <button
            key={t.k}
            onClick={() => setTab(t.k)}
            className={`px-4 py-1.5 rounded-lg text-xs font-mono font-medium transition-all
              ${tab === t.k ? 'bg-amber-glow/15 text-amber-glow' : 'text-ink-muted hover:text-ink'}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Chart panels */}
      <div className="bg-bg-surface border border-bg-border rounded-2xl p-6">

        {/* Stage latency — p50 vs p95 */}
        {tab === 'latency' && (
          <>
            <SectionHeader
              title="Processing Latency per Stage (ms)"
              subtitle="p50 median vs p95 tail latency — from stage:start timestamps in Redis"
            />
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={MOCK_METRICS.latency} margin={{ top: 0, right: 0, left: 0, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a2540" vertical={false} />
                <XAxis dataKey="stage" tick={{ fill: '#8899bb', fontSize: 11 }} angle={-35} textAnchor="end" interval={0} />
                <YAxis tick={{ fill: '#8899bb', fontSize: 11 }} tickFormatter={v => v >= 1000 ? `${v/1000}s` : `${v}ms`} />
                <Tooltip
                  contentStyle={TIP_STYLE}
                  formatter={(v, name) => [`${v >= 1000 ? (v/1000).toFixed(1)+'s' : v+'ms'}`, name]}
                />
                <Bar dataKey="p50" name="p50 median" fill="#1a2540" radius={[4,4,0,0]}>
                  {MOCK_METRICS.latency.map((_, i) => <Cell key={i} fill="#00b8ae" fillOpacity={0.7} />)}
                </Bar>
                <Bar dataKey="p95" name="p95 tail" fill="#f0a000" radius={[4,4,0,0]}>
                  {MOCK_METRICS.latency.map((_, i) => <Cell key={i} fill="#f0a000" fillOpacity={0.85} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="flex gap-4 mt-3">
              <span className="flex items-center gap-1.5 text-xs font-mono text-ink-muted">
                <span className="w-3 h-3 rounded-sm" style={{ background: '#00b8ae', opacity: 0.7 }} />p50 median
              </span>
              <span className="flex items-center gap-1.5 text-xs font-mono text-ink-muted">
                <span className="w-3 h-3 rounded-sm bg-amber-glow" />p95 tail
              </span>
            </div>
          </>
        )}

        {/* ASR confidence distribution */}
        {tab === 'confidence' && (
          <>
            <SectionHeader
              title="ASR Confidence Score Distribution"
              subtitle="Whisper word-level confidence histogram — alert threshold at mean < 0.75"
            />
            <div className="flex items-end gap-1 mb-2">
              <span className="text-xs text-fail font-mono">← Alert zone</span>
              <div className="flex-1 border-t border-dashed border-fail/30 mx-2 mb-0.5" />
              <span className="text-xs text-good font-mono">Good zone →</span>
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={MOCK_METRICS.confidence} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a2540" vertical={false} />
                <XAxis dataKey="bucket" tick={{ fill: '#8899bb', fontSize: 11 }} />
                <YAxis tick={{ fill: '#8899bb', fontSize: 11 }} />
                <Tooltip contentStyle={TIP_STYLE} />
                <Bar dataKey="count" name="Segments" radius={[3,3,0,0]}>
                  {MOCK_METRICS.confidence.map((d, i) => (
                    <Cell key={i} fill={parseFloat(d.bucket) < 0.75 ? '#ff4c6b' : '#00d97e'} fillOpacity={0.75} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </>
        )}

        {/* Failure rates */}
        {tab === 'failures' && (
          <>
            <SectionHeader
              title="Pipeline Stage Failure Rate (%)"
              subtitle="pipeline_stage_failure_total counter — alert threshold at > 5%"
            />
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={failData} layout="vertical" margin={{ top: 0, right: 40, left: 80, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a2540" horizontal={false} />
                <XAxis type="number" tick={{ fill: '#8899bb', fontSize: 11 }} tickFormatter={v => `${v}%`} domain={[0, 12]} />
                <YAxis type="category" dataKey="stage" tick={{ fill: '#8899bb', fontSize: 11 }} width={80} />
                <Tooltip
                  contentStyle={TIP_STYLE}
                  formatter={(v, _, props) => [`${v}% (${props.payload.failures}/${props.payload.total})`, 'Failure rate']}
                />
                <Bar dataKey="rate" name="Failure rate" radius={[0,4,4,0]}>
                  {failData.map((d, i) => (
                    <Cell key={i} fill={d.rate >= 5 ? '#ff4c6b' : d.rate >= 2 ? '#ffb547' : '#00d97e'} fillOpacity={0.8} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="flex gap-4 mt-3">
              {[['#00d97e','< 2% (healthy)'],['#ffb547','2–5% (watch)'],['#ff4c6b','> 5% (alert)']].map(([c, l]) => (
                <span key={l} className="flex items-center gap-1.5 text-xs font-mono text-ink-muted">
                  <span className="w-3 h-3 rounded-sm" style={{ background: c, opacity: 0.8 }} />{l}
                </span>
              ))}
            </div>
          </>
        )}

        {/* Queue depth */}
        {tab === 'queue' && (
          <>
            <SectionHeader
              title="Pipeline Queue Depth (24h)"
              subtitle="Active sessions in processing queue over the last 24 hours"
            />
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={MOCK_METRICS.queueDepth} margin={{ top: 5, right: 20, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a2540" />
                <XAxis dataKey="time" tick={{ fill: '#8899bb', fontSize: 11 }} />
                <YAxis tick={{ fill: '#8899bb', fontSize: 11 }} />
                <Tooltip contentStyle={TIP_STYLE} formatter={v => [`${v} sessions`, 'Queue depth']} />
                <Line type="monotone" dataKey="depth" stroke="#f0a000" strokeWidth={2} dot={{ fill: '#f0a000', r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </>
        )}

      </div>

      {/* Grafana embed note */}
      <div className="mt-4 px-4 py-3 bg-info/5 border border-info/15 rounded-xl flex items-center gap-3">
        <svg className="w-4 h-4 text-info flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
        </svg>
        <p className="text-xs text-info/80 font-mono">
          Live panels also available in Grafana — add <code className="bg-info/10 px-1 rounded">http://gateway:9000/metrics</code> as a Prometheus datasource.
          All 10 metrics from <code className="bg-info/10 px-1 rounded">metrics.py</code> are exposed at <code className="bg-info/10 px-1 rounded">/metrics</code>.
        </p>
      </div>
    </div>
  )
}
