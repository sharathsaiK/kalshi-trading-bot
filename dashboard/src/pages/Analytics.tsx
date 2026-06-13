import { useState } from 'react'
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, BarChart, Bar, Cell,
} from 'recharts'
import { mockWinRateHistory, mockPnLHistory, mockSpeakerStats, mockCalibration, mockSpeakerProfiles } from '../data/mockData'

const GlowCard = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`card-glow ${className}`}>
    <div className="card-glow-inner p-4 h-full">{children}</div>
  </div>
)

const TT = ({ active, payload, label, prefix = '', suffix = '' }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-[#0f1a2e] border border-white/10 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-gray-500 mb-1">{label}</p>
      {payload.map((p: any) => (
        <p key={p.name} style={{ color: p.color }} className="font-semibold">
          {p.name}: {prefix}{typeof p.value === 'number' ? p.value.toFixed(p.name === 'pnl' || p.name === 'drawdown' ? 2 : 1) : p.value}{suffix}
        </p>
      ))}
    </div>
  )
}

// Drawdown = $ below peak at each point
const drawdownData = (() => {
  let peak = 0
  return mockPnLHistory.filter((_, i) => i % 2 === 0).map(p => {
    peak = Math.max(peak, p.pnl)
    return { date: p.date.slice(5), drawdown: parseFloat((p.pnl - peak).toFixed(2)) }
  })
})()

const maxDrawdown = Math.min(...drawdownData.map(d => d.drawdown))
const currentDrawdown = drawdownData[drawdownData.length - 1]?.drawdown ?? 0

// ─── Heatmap ─────────────────────────────────────────────────────────────────
function HeatMap() {
  const [hovered, setHovered] = useState<{ speaker: string; word: string } | null>(null)

  const speakers = ['Trump', 'Powell', 'Vance', 'Rubio', 'Hegseth']
  const words    = ['Economy', 'Jobs', 'America', 'Wall', 'Border', 'China', 'Rate', 'Inflation', 'Employment', 'Military']

  // Build lookup: speaker+word → hitRate
  const lookup: Record<string, number> = {}
  mockSpeakerProfiles.forEach(p => {
    lookup[`${p.speaker}|${p.word}`] = p.hitRateLifetime
  })

  const cellColor = (rate: number | undefined) => {
    if (rate === undefined) return { bg: 'rgba(255,255,255,0.02)', text: '#374151' }
    if (rate >= 0.85) return { bg: 'rgba(52,211,153,0.55)',  text: '#fff' }
    if (rate >= 0.70) return { bg: 'rgba(16,185,129,0.38)',  text: '#6ee7b7' }
    if (rate >= 0.55) return { bg: 'rgba(59,130,246,0.30)',  text: '#93c5fd' }
    if (rate >= 0.40) return { bg: 'rgba(245,158,11,0.25)',  text: '#fcd34d' }
    return              { bg: 'rgba(239,68,68,0.18)',   text: '#fca5a5' }
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr>
            <th className="w-20 pb-2" />
            {words.map(w => (
              <th key={w} className="pb-2 text-[10px] font-medium text-gray-600 text-center px-1 whitespace-nowrap">
                {w}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {speakers.map(speaker => (
            <tr key={speaker}>
              <td className="pr-3 py-1 text-[11px] font-medium text-gray-400 text-right whitespace-nowrap">
                {speaker}
              </td>
              {words.map(word => {
                const rate = lookup[`${speaker}|${word}`]
                const { bg, text } = cellColor(rate)
                const isHovered = hovered?.speaker === speaker && hovered?.word === word
                return (
                  <td key={word} className="p-0.5">
                    <div
                      onMouseEnter={() => setHovered({ speaker, word })}
                      onMouseLeave={() => setHovered(null)}
                      className="relative rounded-md flex items-center justify-center transition-all duration-150 cursor-default select-none"
                      style={{
                        background: bg,
                        height: '36px',
                        minWidth: '52px',
                        transform: isHovered ? 'scale(1.12)' : 'scale(1)',
                        boxShadow: isHovered && rate !== undefined
                          ? `0 0 14px ${bg}, 0 0 4px ${bg}`
                          : 'none',
                        zIndex: isHovered ? 10 : 1,
                        position: 'relative',
                        border: isHovered ? `1px solid ${text}40` : '1px solid transparent',
                      }}
                    >
                      <span className="text-[11px] font-semibold" style={{ color: text }}>
                        {rate !== undefined ? `${(rate * 100).toFixed(0)}%` : '—'}
                      </span>
                    </div>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>

      {/* Legend */}
      <div className="flex items-center gap-3 mt-4 pt-3 border-t border-[#131e30] text-[10px] text-gray-600">
        <span>Hit rate:</span>
        {[
          { label: '≥85%', bg: 'rgba(52,211,153,0.55)' },
          { label: '≥70%', bg: 'rgba(16,185,129,0.38)' },
          { label: '≥55%', bg: 'rgba(59,130,246,0.30)' },
          { label: '≥40%', bg: 'rgba(245,158,11,0.25)' },
          { label: '<40%', bg: 'rgba(239,68,68,0.18)' },
          { label: 'N/A',  bg: 'rgba(255,255,255,0.02)' },
        ].map(({ label, bg }) => (
          <div key={label} className="flex items-center gap-1">
            <div className="w-5 h-3 rounded" style={{ background: bg, border: '1px solid rgba(255,255,255,0.06)' }} />
            <span>{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function Analytics() {
  const [speaker, setSpeaker] = useState('All')
  const speakers = ['All', 'Trump', 'Powell', 'Vance', 'Rubio', 'Hegseth']

  const pnlData = mockPnLHistory.filter((_, i) => i % 3 === 0).map(p => ({
    date: p.date.slice(5), pnl: p.pnl,
  }))

  return (
    <div className="space-y-4 page-enter">
      {/* Controls */}
      <div className="flex items-center gap-3">
        <select value={speaker} onChange={e => setSpeaker(e.target.value)}
          className="bg-[#111827] border border-[#1f2937] text-xs text-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:border-blue-500"
        >
          {speakers.map(s => <option key={s}>{s}</option>)}
        </select>
        <div className="flex items-center gap-0.5 bg-[#111827] border border-[#1f2937] rounded-lg p-0.5">
          {['30D', 'All'].map(r => (
            <button key={r} className="px-3 py-1.5 rounded-md text-xs text-gray-500 hover:text-white hover:bg-white/5 transition-colors">{r}</button>
          ))}
        </div>
      </div>

      {/* Row 1: Win rate + P&L */}
      <div className="grid grid-cols-2 gap-4">
        <GlowCard>
          <h3 className="text-[11px] text-gray-500 uppercase tracking-widest mb-4">Win Rate Over Time</h3>
          <ResponsiveContainer width="100%" height={185}>
            <AreaChart data={mockWinRateHistory} margin={{ top: 4, right: 4, left: -15, bottom: 0 }}>
              <defs>
                <linearGradient id="wrGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="#60a5fa" stopOpacity={0.5} />
                  <stop offset="50%"  stopColor="#3b82f6" stopOpacity={0.15} />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity={0}   />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#131e30" vertical={false} />
              <XAxis dataKey="date" tick={{ fill: '#374151', fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: '#374151', fontSize: 10 }} domain={[60, 100]} tickFormatter={v => `${v}%`} tickLine={false} axisLine={false} />
              <Tooltip content={<TT suffix="%" />} />
              <Area type="monotone" dataKey="wr" name="Win Rate" stroke="#3b82f6" strokeWidth={2}
                fill="url(#wrGrad)" dot={false}
                activeDot={{ r: 4, fill: '#3b82f6', strokeWidth: 0, filter: 'drop-shadow(0 0 6px #3b82f6)' }} />
            </AreaChart>
          </ResponsiveContainer>
        </GlowCard>

        <GlowCard>
          <h3 className="text-[11px] text-gray-500 uppercase tracking-widest mb-4">Cumulative P&L</h3>
          <ResponsiveContainer width="100%" height={185}>
            <AreaChart data={pnlData} margin={{ top: 4, right: 4, left: -10, bottom: 0 }}>
              <defs>
                <linearGradient id="pnlGrad2" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="#34d399" stopOpacity={0.5} />
                  <stop offset="50%"  stopColor="#10b981" stopOpacity={0.15} />
                  <stop offset="100%" stopColor="#10b981" stopOpacity={0}   />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#131e30" vertical={false} />
              <XAxis dataKey="date" tick={{ fill: '#374151', fontSize: 10 }} interval="preserveStartEnd" tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: '#374151', fontSize: 10 }} tickFormatter={v => `$${v}`} tickLine={false} axisLine={false} />
              <Tooltip content={<TT prefix="$" />} />
              <Area type="monotone" dataKey="pnl" name="pnl" stroke="#10b981" strokeWidth={2}
                fill="url(#pnlGrad2)" dot={false}
                activeDot={{ r: 4, fill: '#10b981', strokeWidth: 0, filter: 'drop-shadow(0 0 6px #10b981)' }} />
            </AreaChart>
          </ResponsiveContainer>
        </GlowCard>
      </div>

      {/* Row 2: Drawdown */}
      <GlowCard>
        <div className="flex items-center justify-between mb-1">
          <h3 className="text-[11px] text-gray-500 uppercase tracking-widest">Drawdown</h3>
          <div className="flex items-center gap-4 text-xs">
            <span className="text-gray-600">Max: <span className="text-red-400 font-medium">${maxDrawdown.toFixed(2)}</span></span>
            <span className="text-gray-600">Current: <span className={`font-medium ${currentDrawdown < 0 ? 'text-red-400' : 'text-gray-400'}`}>${currentDrawdown.toFixed(2)}</span></span>
          </div>
        </div>
        <p className="text-[10px] text-gray-700 mb-3">$ below running peak</p>
        <ResponsiveContainer width="100%" height={110}>
          <AreaChart data={drawdownData} margin={{ top: 4, right: 4, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor="#ef4444" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#ef4444" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#131e30" vertical={false} />
            <XAxis dataKey="date" tick={{ fill: '#374151', fontSize: 10 }} interval="preserveStartEnd" tickLine={false} axisLine={false} />
            <YAxis tick={{ fill: '#374151', fontSize: 10 }} tickFormatter={v => `$${v}`} tickLine={false} axisLine={false} />
            <Tooltip content={<TT prefix="$" />} />
            <Area type="monotone" dataKey="drawdown" name="drawdown"
              stroke="#ef4444" strokeWidth={1.5}
              fill="url(#ddGrad)" dot={false}
              activeDot={{ r: 3, fill: '#ef4444', strokeWidth: 0 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </GlowCard>

      {/* Row 3: Speaker accuracy + Calibration */}
      <div className="grid grid-cols-2 gap-4">
        <GlowCard>
          <h3 className="text-[11px] text-gray-500 uppercase tracking-widest mb-4">Accuracy & P&L by Speaker</h3>
          <div className="space-y-3 mb-5">
            {mockSpeakerStats.map(s => (
              <div key={s.speaker}>
                <div className="flex items-center justify-between text-xs mb-1.5">
                  <span className="text-gray-300 font-medium w-16">{s.speaker}</span>
                  <div className="flex-1 mx-3">
                    <div className="h-2 bg-[#0f1828] rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-1000 ${
                          s.winRate >= 80 ? 'bg-emerald-500' : s.winRate >= 70 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{
                          width: `${s.winRate}%`,
                          boxShadow: s.winRate >= 80 ? '0 0 8px rgba(16,185,129,0.5)' : 'none',
                        }}
                      />
                    </div>
                  </div>
                  <span className="text-gray-500 w-10 text-right">{s.winRate.toFixed(1)}%</span>
                  <span className={`w-14 text-right font-semibold ${s.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}
                    style={s.pnl > 5 ? { textShadow: '0 0 8px rgba(16,185,129,0.4)' } : {}}>
                    {s.pnl >= 0 ? '+' : ''}${s.pnl.toFixed(2)}
                  </span>
                </div>
              </div>
            ))}
          </div>
          <h3 className="text-[11px] text-gray-500 uppercase tracking-widest mb-3">Bets by Speaker</h3>
          <ResponsiveContainer width="100%" height={100}>
            <BarChart data={mockSpeakerStats} margin={{ top: 0, right: 0, left: -25, bottom: 0 }}>
              <XAxis dataKey="speaker" tick={{ fill: '#374151', fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: '#374151', fontSize: 10 }} tickLine={false} axisLine={false} />
              <Tooltip content={<TT />} />
              <Bar dataKey="bets" name="Bets" radius={[4, 4, 0, 0]}>
                {mockSpeakerStats.map((s, i) => (
                  <Cell key={i}
                    fill={s.winRate >= 80 ? '#10b981' : s.winRate >= 70 ? '#f59e0b' : '#ef4444'}
                    fillOpacity={0.75}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </GlowCard>

        <GlowCard>
          <h3 className="text-[11px] text-gray-500 uppercase tracking-widest mb-4">Calibration — Our Prob vs Actual</h3>
          <ResponsiveContainer width="100%" height={215}>
            <LineChart data={mockCalibration} margin={{ top: 4, right: 4, left: -15, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#131e30" />
              <XAxis dataKey="prob" tick={{ fill: '#374151', fontSize: 10 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: '#374151', fontSize: 10 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} domain={[0, 1]} tickLine={false} axisLine={false} />
              <Tooltip content={<TT />} />
              <Line type="monotone" dataKey="actual"  name="Model"   stroke="#3b82f6" strokeWidth={2}
                dot={{ r: 4, fill: '#3b82f6', strokeWidth: 0 }}
                activeDot={{ r: 5, filter: 'drop-shadow(0 0 6px #3b82f6)' }} />
              <Line type="monotone" dataKey="perfect" name="Perfect" stroke="#1f2937" strokeWidth={1}
                strokeDasharray="5 5" dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-5 mt-2 text-[11px] text-gray-600">
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-blue-500 inline-block rounded" style={{ boxShadow: '0 0 4px #3b82f6' }} /> Model
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-gray-700 inline-block rounded" /> Perfect
            </span>
          </div>
          <div className="grid grid-cols-3 gap-3 mt-4 pt-4 border-t border-[#131e30]">
            {[
              { label: 'AUC-ROC',  value: '0.808', color: 'text-blue-400',   glow: 'glow-blue' },
              { label: 'Brier',    value: '0.1774', color: 'text-purple-400', glow: '' },
              { label: 'Log Loss', value: '0.412',  color: 'text-gray-300',   glow: '' },
            ].map(({ label, value, color, glow }) => (
              <div key={label} className="text-center bg-[#0a1020] rounded-lg py-2">
                <p className="text-[10px] text-gray-600 uppercase tracking-wider">{label}</p>
                <p className={`text-lg font-bold mt-0.5 ${color} ${glow}`}>{value}</p>
              </div>
            ))}
          </div>
        </GlowCard>
      </div>

      {/* Row 4: Hit Rate Heatmap */}
      <GlowCard>
        <h3 className="text-[11px] text-gray-500 uppercase tracking-widest mb-1">Hit Rate Heatmap — Speaker × Word</h3>
        <p className="text-[10px] text-gray-700 mb-4">Lifetime hit rate per (speaker, word) pair. Brighter = higher probability of saying the word.</p>
        <HeatMap />
      </GlowCard>
    </div>
  )
}
