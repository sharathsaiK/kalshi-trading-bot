import { useState, useEffect, useRef } from 'react'
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid,
} from 'recharts'
import {
  mockPnLHistory, mockLiveMarkets, mockActivityFeed, newFeedPool,
  mockSpeakerStats, mockTrades, type ActivityItem,
} from '../data/mockData'
import { Flame, TrendingUp } from 'lucide-react'

// ─── Count-up hook ───────────────────────────────────────────────────────────
function useCountUp(target: number, duration = 1300) {
  const [val, setVal] = useState(0)
  useEffect(() => {
    const start = performance.now()
    const tick = (now: number) => {
      const t = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - t, 4)
      setVal(target * eased)
      if (t < 1) requestAnimationFrame(tick)
      else setVal(target)
    }
    requestAnimationFrame(tick)
  }, [target])
  return val
}

// ─── Sparkline ───────────────────────────────────────────────────────────────
const Sparkline = ({ data, color }: { data: number[]; color: string }) => {
  const d = data.map(v => ({ v }))
  return (
    <ResponsiveContainer width="100%" height={30}>
      <LineChart data={d} margin={{ top: 2, right: 0, left: 0, bottom: 2 }}>
        <Line type="monotone" dataKey="v" stroke={color} strokeWidth={1.5} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}

// ─── Metric card ─────────────────────────────────────────────────────────────
interface MetricProps {
  label: string
  value: number
  format: (v: number) => string
  sub: string
  sparkData: number[]
  color: 'green' | 'blue' | 'white'
  glowClass: string
}

const MetricCard = ({ label, value, format, sub, sparkData, color, glowClass }: MetricProps) => {
  const animated = useCountUp(value)
  const strokeColor = color === 'green' ? '#10b981' : color === 'blue' ? '#3b82f6' : '#a78bfa'
  const textColor   = color === 'green' ? 'text-emerald-400' : color === 'blue' ? 'text-blue-400' : 'text-purple-400'

  return (
    <div className="card-glow card-hover">
      <div className="card-glow-inner p-4">
        <p className="text-[10px] text-gray-600 uppercase tracking-widest mb-1">{label}</p>
        <p className={`text-2xl font-bold tracking-tight count-shimmer ${textColor} ${glowClass}`}>
          {format(animated)}
        </p>
        <p className="text-[10px] text-gray-600 mt-0.5 mb-1">{sub}</p>
        <Sparkline data={sparkData} color={strokeColor} />
      </div>
    </div>
  )
}

// ─── Portfolio arc ────────────────────────────────────────────────────────────
const PortfolioArc = ({ deployed, total }: { deployed: number; total: number }) => {
  const pct = Math.min(deployed / total, 1)
  const r = 38
  const cx = 56, cy = 56
  const startAngle = -220
  const sweep = 260
  const toRad = (deg: number) => (deg * Math.PI) / 180
  const arcPath = (angleDeg: number) => {
    const a = toRad(angleDeg)
    return `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`
  }
  const circumference = (sweep / 360) * 2 * Math.PI * r

  const trackPath = (() => {
    const steps = 60
    let d = `M ${arcPath(startAngle)}`
    for (let i = 1; i <= steps; i++) {
      d += ` L ${arcPath(startAngle + (sweep * i) / steps)}`
    }
    return d
  })()

  const fillOffset = circumference * (1 - pct)
  const fillColor = pct > 0.7 ? '#f59e0b' : '#10b981'

  return (
    <div className="flex flex-col items-center">
      <svg width={112} height={90} style={{ overflow: 'visible' }}>
        <defs>
          <filter id="arcGlow">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        {/* Track */}
        <path d={trackPath} fill="none" stroke="#1a2640" strokeWidth="6" strokeLinecap="round" />
        {/* Fill */}
        <path
          d={trackPath} fill="none" stroke={fillColor} strokeWidth="6" strokeLinecap="round"
          strokeDasharray={circumference} strokeDashoffset={fillOffset}
          filter="url(#arcGlow)" className="arc-fill"
        />
        {/* Center text */}
        <text x={cx} y={cy - 2} textAnchor="middle" fill="white" fontSize="14" fontWeight="700">
          {(pct * 100).toFixed(0)}%
        </text>
        <text x={cx} y={cy + 12} textAnchor="middle" fill="#4b5563" fontSize="9">
          deployed
        </text>
      </svg>
      <div className="text-center -mt-1">
        <p className="text-[10px] text-gray-600">
          <span className={pct > 0.7 ? 'text-yellow-400' : 'text-emerald-400'} style={{ textShadow: '0 0 10px currentColor' }}>
            ${deployed.toFixed(0)}
          </span>
          <span className="text-gray-700"> / ${total}</span>
        </p>
      </div>
    </div>
  )
}

// ─── Toast ────────────────────────────────────────────────────────────────────
interface ToastItem { id: number; side: 'YES' | 'NO'; word: string; contracts: number; ev: number; leaving: boolean }

const ToastNotif = ({ t }: { t: ToastItem }) => (
  <div className={`${t.leaving ? 'toast-out' : 'toast-in'} flex items-center gap-3 px-4 py-3 rounded-xl border shadow-2xl backdrop-blur-sm ${
    t.side === 'YES'
      ? 'bg-emerald-950/90 border-emerald-500/40'
      : 'bg-blue-950/90 border-blue-500/40'
  }`}>
    <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold ${
      t.side === 'YES' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-blue-500/20 text-blue-400'
    }`}>{t.side}</div>
    <div>
      <p className="text-xs font-semibold text-white">{t.word} ×{t.contracts}</p>
      <p className="text-[10px] text-gray-400">EV <span className="text-emerald-400">+{t.ev.toFixed(3)}</span></p>
    </div>
  </div>
)

// ─── Active event banner ──────────────────────────────────────────────────────
const ActiveEventBanner = () => (
  <div className="banner-pulse rounded-xl bg-emerald-950/40 px-4 py-2.5 flex items-center justify-between">
    <div className="flex items-center gap-3 text-xs">
      <span className="flex items-center gap-1.5 text-emerald-400 font-semibold">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        LIVE EVENT
      </span>
      <span className="text-gray-500">·</span>
      <span className="text-gray-300 font-medium">KXTRUMPSOTU-26APR14</span>
      <span className="text-gray-500">·</span>
      <span className="text-gray-400">5 markets open</span>
      <span className="text-gray-500">·</span>
      <span className="text-emerald-400">2 bets placed</span>
    </div>
    <div className="flex items-center gap-2 text-[11px] text-gray-500">
      <span>Speaker: <span className="text-gray-300">Trump</span></span>
      <span className="text-gray-700">·</span>
      <span>Transcript: <span className="text-gray-300">42,318 chars</span></span>
    </div>
  </div>
)

// ─── Chart tooltip ────────────────────────────────────────────────────────────
const ChartTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-[#0f1a2e] border border-emerald-500/20 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-gray-500 mb-0.5">{label}</p>
      <p className="text-emerald-400 font-bold" style={{ textShadow: '0 0 10px rgba(16,185,129,0.5)' }}>
        ${payload[0].value.toFixed(2)}
      </p>
    </div>
  )
}

// ─── Source colours ───────────────────────────────────────────────────────────
const sourceColors: Record<string, string> = {
  TRADE: 'text-emerald-400', GATE: 'text-gray-600', MODEL: 'text-blue-400',
  API: 'text-purple-400', PROFILE: 'text-yellow-400', NEWS: 'text-orange-400',
  HARVEST: 'text-cyan-400', TRAIN: 'text-indigo-400', BOT: 'text-gray-300',
}

// ─── Sparkline data for each metric card ─────────────────────────────────────
const sparklines = {
  winRate:   [76, 78, 79, 80, 81, 82, 83],
  pnl:       [18, 25, 31, 36, 40, 43, 45],
  roi:       [0.18, 0.20, 0.22, 0.23, 0.24, 0.25, 0.26],
  accuracy:  [76, 77, 78, 79, 80, 80, 80],
  auc:       [0.78, 0.79, 0.79, 0.80, 0.80, 0.81, 0.81],
  brier:     [0.21, 0.20, 0.20, 0.19, 0.18, 0.18, 0.18],
}

// Compute win streak from trades (newest first)
const winStreak = (() => {
  let s = 0
  for (const t of mockTrades) {
    if (t.result === 'OPEN') continue
    if (t.result === 'WIN') s++
    else break
  }
  return s
})()

// P&L chart data (30D)
const chartData = mockPnLHistory.slice(-30).filter((_, i) => i % 1 === 0).map(p => ({
  date: p.date.slice(5),
  pnl: p.pnl,
}))

// ─── Main component ───────────────────────────────────────────────────────────
export default function Overview() {
  const [range, setRange] = useState<'7D' | '30D' | 'All'>('30D')
  const [feed, setFeed] = useState<ActivityItem[]>(mockActivityFeed)
  const [toasts, setToasts] = useState<ToastItem[]>([])
  const nextFeedId  = useRef(mockActivityFeed.length + 1)
  const nextToastId = useRef(1)

  // Live feed + toast trigger
  useEffect(() => {
    const iv = setInterval(() => {
      const item = newFeedPool[Math.floor(Math.random() * newFeedPool.length)]
      const now  = new Date()
      const time = [now.getHours(), now.getMinutes(), now.getSeconds()]
        .map(n => String(n).padStart(2, '0')).join(':')
      const newItem = { ...item, id: nextFeedId.current++, time }
      setFeed(prev => [newItem, ...prev.slice(0, 19)])

      // Fire a toast when it's a trade
      if (item.source === 'TRADE') {
        const side: 'YES' | 'NO' = item.message.includes('NO') ? 'NO' : 'YES'
        const contracts = parseInt(item.message.match(/×(\d+)/)?.[1] || '1')
        const ev = parseFloat(item.message.match(/EV \+(\d+\.\d+)/)?.[1] || '0.15')
        const word = item.message.match(/\/(\w+) \(/)?.[1] || 'Word'
        const id = nextToastId.current++
        const toast: ToastItem = { id, side, word, contracts, ev, leaving: false }
        setToasts(prev => [...prev, toast])
        setTimeout(() => {
          setToasts(prev => prev.map(t => t.id === id ? { ...t, leaving: true } : t))
          setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 350)
        }, 4000)
      }
    }, 2800)
    return () => clearInterval(iv)
  }, [])

  // Filtered chart data
  const days = range === '7D' ? 7 : range === '30D' ? 30 : 999
  const filteredChart = (days >= 999
    ? mockPnLHistory.filter((_, i) => i % 3 === 0)
    : mockPnLHistory.slice(-days)
  ).map(p => ({ date: p.date.slice(5), pnl: p.pnl }))

  return (
    <div className="space-y-4">
      {/* Active event banner */}
      <ActiveEventBanner />

      {/* Metric cards */}
      <div className="grid grid-cols-3 gap-3">
        <MetricCard label="Win Rate"  value={83.1} format={v => `${v.toFixed(1)}%`}  sub="↑ +2.1% vs 7d"         sparkData={sparklines.winRate}  color="green" glowClass="glow-green" />
        <MetricCard label="P&L Total" value={45.01} format={v => `+$${v.toFixed(2)}`} sub="77 settled bets"       sparkData={sparklines.pnl}      color="green" glowClass="glow-green" />
        <MetricCard label="ROI / Bet" value={25.9}  format={v => `+${v.toFixed(1)}¢`} sub="avg expected value"    sparkData={sparklines.roi.map(x => x * 100)} color="green" glowClass="glow-green" />
      </div>

      {/* Win streak badge */}
      {winStreak >= 3 && (
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-orange-950/50 border border-orange-500/30 rounded-lg text-xs"
            style={{ boxShadow: '0 0 12px rgba(249,115,22,0.15)' }}>
            <Flame size={13} className="text-orange-400" />
            <span className="text-orange-400 font-semibold">{winStreak} win streak</span>
            <span className="text-gray-600">· keep going</span>
          </div>
        </div>
      )}

      {/* Charts row */}
      <div className="grid grid-cols-3 gap-4">
        {/* P&L area chart */}
        <div className="col-span-2 card-glow">
          <div className="card-glow-inner p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-[11px] text-gray-500 uppercase tracking-widest">Cumulative P&L</h3>
              <div className="flex gap-0.5 bg-[#0c1426] border border-[#1a2640] rounded-lg p-0.5">
                {(['7D', '30D', 'All'] as const).map(r => (
                  <button key={r} onClick={() => setRange(r)}
                    className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${range === r ? 'bg-blue-600 text-white' : 'text-gray-500 hover:text-white'}`}
                  >{r}</button>
                ))}
              </div>
            </div>
            <ResponsiveContainer width="100%" height={190}>
              <AreaChart data={filteredChart} margin={{ top: 4, right: 4, left: -10, bottom: 0 }}>
                <defs>
                  <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%"   stopColor="#10b981" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#10b981" stopOpacity={0}    />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#131e30" vertical={false} />
                <XAxis dataKey="date" tick={{ fill: '#374151', fontSize: 10 }} interval="preserveStartEnd" tickLine={false} axisLine={false} />
                <YAxis tick={{ fill: '#374151', fontSize: 10 }} tickFormatter={v => `$${v}`} tickLine={false} axisLine={false} />
                <Tooltip content={<ChartTooltip />} />
                <Area
                  type="monotone" dataKey="pnl"
                  stroke="#10b981" strokeWidth={2}
                  fill="url(#pnlGrad)"
                  dot={false}
                  activeDot={{ r: 4, fill: '#10b981', strokeWidth: 0, filter: 'drop-shadow(0 0 6px #10b981)' }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Live feed */}
        <div className="card-glow">
          <div className="card-glow-inner p-4 flex flex-col h-full">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-[11px] text-gray-500 uppercase tracking-widest">Live Feed</h3>
              <span className="flex items-center gap-1 text-[10px] text-emerald-400">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" /> live
              </span>
            </div>
            <div className="space-y-2 overflow-y-auto max-h-[205px] flex-1">
              {feed.map(item => (
                <div key={item.id} className="text-[11px] leading-relaxed">
                  <span className="text-gray-700 mr-1">{item.time}</span>
                  <span className={`font-semibold mr-1 ${sourceColors[item.source] || 'text-gray-400'}`}>[{item.source}]</span>
                  <span className={
                    item.level === 'ERROR' ? 'text-red-400' :
                    item.level === 'WARN'  ? 'text-yellow-400' :
                    item.source === 'TRADE' ? 'text-emerald-300' :
                    'text-gray-500'
                  }>{item.message}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Live markets + portfolio + speakers */}
      <div className="grid grid-cols-4 gap-4">
        {/* Live markets */}
        <div className="col-span-2 card-glow">
          <div className="card-glow-inner p-4">
            <h3 className="text-[11px] text-gray-500 uppercase tracking-widest mb-3">Live Markets</h3>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[#1a2030] text-gray-700">
                  <th className="text-left pb-2 font-medium">Ticker / Word</th>
                  <th className="text-right pb-2 font-medium">Prob</th>
                  <th className="text-right pb-2 font-medium">Mkt</th>
                  <th className="text-right pb-2 font-medium">EV</th>
                  <th className="text-right pb-2 font-medium">Call</th>
                </tr>
              </thead>
              <tbody>
                {mockLiveMarkets.map((m, i) => (
                  <tr key={i} className="border-b border-[#141d2e]/60 hover:bg-white/[0.02]">
                    <td className="py-2.5">
                      <span className="text-gray-600">{m.ticker.split('-')[0]}/</span>
                      <span className="text-white font-medium"> {m.word}</span>
                    </td>
                    <td className="py-2.5 text-right text-gray-400">{m.ourProb.toFixed(2)}</td>
                    <td className="py-2.5 text-right text-gray-400">{m.mktPrice.toFixed(2)}</td>
                    <td className={`py-2.5 text-right font-semibold ${m.ev >= 0.10 ? 'text-emerald-400' : m.ev < 0 ? 'text-red-400' : 'text-gray-600'}`}
                      style={m.ev >= 0.10 ? { textShadow: '0 0 8px rgba(16,185,129,0.4)' } : {}}>
                      {m.ev >= 0 ? '+' : ''}{m.ev.toFixed(3)}
                    </td>
                    <td className="py-2.5 text-right">
                      <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-medium ${
                        m.status === 'bet'  ? 'bg-emerald-500/15 text-emerald-400' :
                        m.status === 'skip' ? 'bg-gray-800 text-gray-600' :
                                              'bg-blue-500/15 text-blue-400'
                      }`}>{m.call}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Portfolio arc */}
        <div className="card-glow">
          <div className="card-glow-inner p-4 flex flex-col items-center justify-center">
            <h3 className="text-[11px] text-gray-500 uppercase tracking-widest mb-3 self-start">Portfolio</h3>
            <PortfolioArc deployed={182} total={1000} />
            <div className="w-full mt-3 space-y-1.5 text-xs">
              {[
                { label: 'Open positions', value: '9',      color: 'text-white' },
                { label: 'Bankroll',       value: '$1,000', color: 'text-gray-300' },
                { label: 'Deployed',       value: '$182',   color: 'text-emerald-400' },
                { label: 'Available',      value: '$818',   color: 'text-gray-400' },
              ].map(({ label, value, color }) => (
                <div key={label} className="flex justify-between">
                  <span className="text-gray-600">{label}</span>
                  <span className={`font-medium ${color}`}>{value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Speaker snapshot */}
        <div className="card-glow">
          <div className="card-glow-inner p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-[11px] text-gray-500 uppercase tracking-widest">By Speaker</h3>
              <TrendingUp size={12} className="text-gray-700" />
            </div>
            <div className="space-y-2">
              {mockSpeakerStats.map(s => (
                <div key={s.speaker}>
                  <div className="flex items-center justify-between mb-0.5 text-xs">
                    <span className="text-gray-300 font-medium">{s.speaker}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-gray-600">{s.bets}b</span>
                      <span className={`font-semibold ${s.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}
                        style={s.pnl > 5 ? { textShadow: '0 0 8px rgba(16,185,129,0.4)' } : {}}>
                        {s.pnl >= 0 ? '+' : ''}${s.pnl.toFixed(2)}
                      </span>
                    </div>
                  </div>
                  <div className="h-1.5 bg-[#0f1828] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-1000 ${
                        s.winRate >= 80 ? 'bg-emerald-500' : s.winRate >= 70 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{
                        width: `${s.winRate}%`,
                        boxShadow: s.winRate >= 80 ? '0 0 6px rgba(16,185,129,0.5)' : 'none',
                      }}
                    />
                  </div>
                  <div className="text-[10px] text-gray-600">{s.winRate.toFixed(1)}% WR</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Toast container */}
      <div className="fixed bottom-5 right-5 flex flex-col gap-2 z-50 pointer-events-none">
        {toasts.map(t => <ToastNotif key={t.id} t={t} />)}
      </div>
    </div>
  )
}
