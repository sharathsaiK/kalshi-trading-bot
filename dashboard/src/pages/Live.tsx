import { useState, useEffect, useRef } from 'react'
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid,
} from 'recharts'
import {
  mockPnLHistory, mockLiveMarkets, mockActivityFeed, newFeedPool,
  mockTrades, mockUpcomingEvents, type ActivityItem,
} from '../data/mockData'
import { Calendar } from 'lucide-react'

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

// ─── Arc gauge ───────────────────────────────────────────────────────────────
interface ArcProps { value: number; max: number; label: string; format: (v: number) => string; color: string; trailColor?: string }

const ArcGauge = ({ value, max, label, format, color, trailColor = '#1a2640' }: ArcProps) => {
  const animated   = useCountUp(value)
  const [ready, setReady] = useState(false)
  useEffect(() => { setReady(true) }, [])

  const gradId     = `arc-grad-${label.replace(/\s+/g, '-')}`
  const R = 44
  const cx = 56, cy = 56
  const startAngle = -220
  const sweep      = 260
  const toRad      = (deg: number) => (deg * Math.PI) / 180
  const targetPct  = Math.min(value / max, 1)
  const circumference = (sweep / 360) * 2 * Math.PI * R

  const arcPath = (() => {
    const steps = 60
    let d = `M ${cx + R * Math.cos(toRad(startAngle))},${cy + R * Math.sin(toRad(startAngle))}`
    for (let i = 1; i <= steps; i++) {
      const angle = startAngle + (sweep * i) / steps
      d += ` L ${cx + R * Math.cos(toRad(angle))},${cy + R * Math.sin(toRad(angle))}`
    }
    return d
  })()

  const fillOffset = ready ? circumference * (1 - targetPct) : circumference

  return (
    <div className="card-glow card-hover">
      <div className="card-glow-inner p-4 flex flex-col items-center">
        <p className="text-[10px] text-gray-600 uppercase tracking-widest mb-2 self-start">{label}</p>
        <svg width={112} height={100} viewBox="0 0 112 100">
          <defs>
            <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%"   stopColor={color} stopOpacity={0.6} />
              <stop offset="100%" stopColor={color} stopOpacity={1.0} />
            </linearGradient>
          </defs>
          {/* Track */}
          <path d={arcPath} fill="none" stroke={trailColor} strokeWidth={8} strokeLinecap="round" />
          {/* Fill — always rendered, CSS transition animates dashoffset from empty to targetPct */}
          <path
            d={arcPath}
            fill="none"
            stroke={`url(#${gradId})`}
            strokeWidth={8}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={fillOffset}
            className="arc-fill"
            style={{ filter: `drop-shadow(0 0 5px ${color}88)` }}
          />
          {/* Value */}
          <text x={cx} y={cy + 6} textAnchor="middle" fill="white" fontSize={18} fontWeight="bold"
            style={{ fontFamily: 'inherit' }}>
            {format(animated)}
          </text>
        </svg>
      </div>
    </div>
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
        <p className={`text-2xl font-bold tracking-tight count-shimmer ${
          color === 'green' ? 'gradient-text-green' : 'gradient-text-blue'
        } ${glowClass}`}>
          {format(animated)}
        </p>
        <p className="text-[10px] text-gray-600 mt-0.5 mb-1">{sub}</p>
        <Sparkline data={sparkData} color={strokeColor} />
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

// ─── Sparkline data ───────────────────────────────────────────────────────────
const sparklines = {
  winRate:   [76, 78, 79, 80, 81, 82, 83],
  pnl:       [18, 25, 31, 36, 40, 43, 45],
  roi:       [0.18, 0.20, 0.22, 0.23, 0.24, 0.25, 0.26],
  accuracy:  [76, 77, 78, 79, 80, 80, 80],
  auc:       [0.78, 0.79, 0.79, 0.80, 0.80, 0.81, 0.81],
  brier:     [0.21, 0.20, 0.20, 0.19, 0.18, 0.18, 0.18],
}

const openPositions = mockTrades.filter(t => t.result === 'OPEN')

// ─── Main component ───────────────────────────────────────────────────────────
export default function Live() {
  const [range, setRange] = useState<'7D' | '30D' | 'All'>('30D')
  const [feed, setFeed] = useState<ActivityItem[]>(mockActivityFeed)
  const [toasts, setToasts] = useState<ToastItem[]>([])
  const nextFeedId  = useRef(mockActivityFeed.length + 1)
  const nextToastId = useRef(1)

  useEffect(() => {
    const iv = setInterval(() => {
      const item = newFeedPool[Math.floor(Math.random() * newFeedPool.length)]
      const now  = new Date()
      const time = [now.getHours(), now.getMinutes(), now.getSeconds()]
        .map(n => String(n).padStart(2, '0')).join(':')
      const newItem = { ...item, id: nextFeedId.current++, time }
      setFeed(prev => [newItem, ...prev.slice(0, 19)])

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

  const days = range === '7D' ? 7 : range === '30D' ? 30 : 999
  const filteredChart = (days >= 999
    ? mockPnLHistory.filter((_, i) => i % 3 === 0)
    : mockPnLHistory.slice(-days)
  ).map(p => ({ date: p.date.slice(5), pnl: p.pnl }))

  return (
    <div className="space-y-4 page-enter">
      <ActiveEventBanner />

      {/* Metric cards — arc gauge for Win Rate, sparkline cards for P&L + ROI */}
      <div className="grid grid-cols-3 gap-3">
        <ArcGauge label="Win Rate"  value={83.1} max={100} format={v => `${v.toFixed(0)}%`}  color="#34d399" />
        <MetricCard label="P&L Total" value={45.01} format={v => `+$${v.toFixed(2)}`} sub="77 settled bets"    sparkData={sparklines.pnl}              color="green" glowClass="glow-green" />
        <MetricCard label="ROI / Bet" value={25.9}  format={v => `+${v.toFixed(1)}¢`} sub="avg expected value" sparkData={sparklines.roi.map(x => x * 100)} color="green" glowClass="glow-green" />
      </div>


      {/* P&L chart + live feed */}
      <div className="grid grid-cols-3 gap-4">
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
                    <stop offset="0%"   stopColor="#34d399" stopOpacity={0.5} />
                    <stop offset="50%"  stopColor="#10b981" stopOpacity={0.15} />
                    <stop offset="100%" stopColor="#10b981" stopOpacity={0}   />
                  </linearGradient>
                  <filter id="lineGlow">
                    <feGaussianBlur stdDeviation="2" result="blur" />
                    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                  </filter>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#131e30" vertical={false} />
                <XAxis dataKey="date" tick={{ fill: '#374151', fontSize: 10 }} interval="preserveStartEnd" tickLine={false} axisLine={false} />
                <YAxis tick={{ fill: '#374151', fontSize: 10 }} tickFormatter={v => `$${v}`} tickLine={false} axisLine={false} />
                <Tooltip content={<ChartTooltip />} />
                <Area
                  type="monotone" dataKey="pnl"
                  stroke="#34d399" strokeWidth={2.5}
                  fill="url(#pnlGrad)"
                  dot={false}
                  filter="url(#lineGlow)"
                  activeDot={{ r: 5, fill: '#34d399', strokeWidth: 0, filter: 'drop-shadow(0 0 8px #34d399)' }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

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

      {/* Live markets + open positions + upcoming events */}
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
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Open positions */}
        <div className="card-glow">
          <div className="card-glow-inner p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-[11px] text-gray-500 uppercase tracking-widest">Open Positions</h3>
              <span className="text-[10px] text-blue-400 font-medium">{openPositions.length} open</span>
            </div>
            <div className="space-y-0 overflow-y-auto" style={{ maxHeight: 220 }}>
              {openPositions.map(t => (
                <div key={t.id} className="flex items-center justify-between py-2 border-b border-[#141d2e]/60 last:border-0">
                  <div>
                    <span className="text-xs text-white font-medium">{t.word}</span>
                    <span className="text-[10px] text-gray-600 ml-1.5">{t.speaker}</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                      t.side === 'YES' ? 'bg-emerald-500/15 text-emerald-400' : 'bg-blue-500/15 text-blue-400'
                    }`}>{t.side}</span>
                    <span className="text-[10px] text-gray-600">×{t.contracts}</span>
                    <span className="text-[10px] text-gray-700">{t.date.slice(5)}</span>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-3 pt-2 border-t border-[#1a2640] flex justify-between text-[10px] text-gray-600">
              <span>Deployed</span>
              <span className="text-emerald-400 font-medium">$182</span>
            </div>
          </div>
        </div>

        {/* Upcoming events */}
        <div className="card-glow">
          <div className="card-glow-inner p-4">
            <div className="flex items-center gap-2 mb-3">
              <Calendar size={12} className="text-gray-700" />
              <h3 className="text-[11px] text-gray-500 uppercase tracking-widest">Upcoming</h3>
            </div>
            <div className="space-y-3">
              {mockUpcomingEvents.map(ev => (
                <div key={ev.ticker} className="flex items-start justify-between">
                  <div>
                    <div className="text-xs text-gray-200 font-medium">{ev.speaker}</div>
                    <div className="text-[10px] text-gray-600 mt-0.5 capitalize">{ev.type.replace('_', ' ')}</div>
                    <div className="text-[10px] text-gray-700 mt-0.5">{ev.markets} markets</div>
                  </div>
                  <div className="text-right">
                    <div className="text-[10px] text-gray-400">{ev.date.slice(5)}</div>
                    <div className={`text-xs font-semibold mt-0.5 ${ev.daysOut <= 7 ? 'text-yellow-400' : 'text-gray-500'}`}>
                      {ev.daysOut}d
                    </div>
                  </div>
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
