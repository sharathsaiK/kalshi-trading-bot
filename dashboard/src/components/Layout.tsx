import { useState, useEffect } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import {
  Radio, ListOrdered, BarChart2, Brain, Server,
  Play, Square, Pause, AlertOctagon, Circle, Users, CandlestickChart, BookOpen,
} from 'lucide-react'
import CommandPalette from './CommandPalette'
import logoSrc from '../assets/qourex_mark.svg'

// ─── Logo ─────────────────────────────────────────────────────────────────────
const Logo = () => (
  <img
    src={logoSrc}
    alt="logo"
    width={58}
    height={58}
    style={{
      display: 'block',
      filter: 'drop-shadow(0 0 6px rgba(255,255,255,0.55)) drop-shadow(0 0 14px rgba(59,130,246,0.45))',
    }}
  />
)

const navItems = [
  { to: '/live',      label: 'Live',          icon: Radio },
  { to: '/markets',   label: 'Markets',       icon: CandlestickChart },
  { to: '/history',   label: 'History',       icon: ListOrdered },
  { to: '/analytics', label: 'Analytics',     icon: BarChart2 },
  { to: '/model',     label: 'Model',         icon: Brain },
  { to: '/speakers',  label: 'Speakers',      icon: Users },
  { to: '/news',      label: 'News & Trans.', icon: BookOpen },
  { to: '/system',    label: 'System',        icon: Server },
]

type BotState = 'running' | 'paused' | 'stopped'

function useCountUp(target: number, duration = 1400) {
  const [val, setVal] = useState(0)
  useEffect(() => {
    const start = performance.now()
    const tick = (now: number) => {
      const t = Math.min((now - start) / duration, 1)
      setVal(target * (1 - Math.pow(1 - t, 4)))
      if (t < 1) requestAnimationFrame(tick)
      else setVal(target)
    }
    requestAnimationFrame(tick)
  }, [target, duration])
  return val
}

export default function Layout() {
  const [botState, setBotState]   = useState<BotState>('running')
  const [mode, setMode]           = useState<'paper' | 'live'>('paper')
  const [paletteOpen, setPalette] = useState(false)
  const animatedPnl = useCountUp(45.01)

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setPalette(p => !p)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const botColor = botState === 'running' ? 'text-emerald-400' : botState === 'paused' ? 'text-yellow-400' : 'text-gray-500'
  const botLabel = botState === 'running' ? 'RUNNING' : botState === 'paused' ? 'PAUSED' : 'STOPPED'

  return (
    <div className="flex h-screen dot-grid text-white overflow-hidden">
      <CommandPalette open={paletteOpen} onClose={() => setPalette(false)} />
      {/* Sidebar */}
      <aside className="w-[210px] flex-shrink-0 bg-[#0b1220] border-r border-[#1a2640]/60 flex flex-col" style={{ boxShadow: '1px 0 20px rgba(0,0,0,0.4)' }}>
        {/* Logo */}
        <div className="px-4 py-4 border-b border-[#1a2640]">
          <div className="flex items-center gap-2.5">
            <Logo />
            <div>
              <div className="text-sm font-bold text-white leading-tight">KALSHI BOT</div>
              <div className={`text-[10px] font-medium ${botColor} flex items-center gap-1`}>
                <span className={`inline-block w-1.5 h-1.5 rounded-full ${botState === 'running' ? 'bg-emerald-400 animate-pulse' : botState === 'paused' ? 'bg-yellow-400' : 'bg-gray-500'}`} />
                {botLabel}
              </div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-2 py-3 overflow-y-auto">
          <div className="space-y-0.5">
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to} to={to}
                className={({ isActive }) =>
                  `relative flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all ${
                    isActive
                      ? 'bg-blue-500/10 text-blue-400 font-medium nav-active'
                      : 'text-gray-500 hover:text-gray-200 hover:bg-white/5'
                  }`
                }
              >
                <Icon size={15} />
                {label}
              </NavLink>
            ))}
          </div>
        </nav>

        {/* Controls */}
        <div className="px-2 pb-3 pt-2 border-t border-[#1a2640]">
          <p className="text-[10px] text-gray-600 uppercase tracking-wider px-2 mb-2">Bot Control</p>
          <div className="space-y-1">
            {[
              { label: 'Start',  icon: Play,         state: 'running' as BotState,  activeClass: 'bg-emerald-500/15 text-emerald-400' },
              { label: 'Pause',  icon: Pause,        state: 'paused' as BotState,   activeClass: 'bg-yellow-500/15 text-yellow-400' },
              { label: 'Stop',   icon: Square,       state: 'stopped' as BotState,  activeClass: 'bg-gray-500/15 text-gray-400' },
            ].map(({ label, icon: Icon, state, activeClass }) => (
              <button
                key={label}
                onClick={() => setBotState(state)}
                className={`w-full flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  botState === state ? activeClass : 'text-gray-500 hover:bg-white/5 hover:text-gray-300'
                }`}
              >
                <Icon size={11} /> {label}
              </button>
            ))}
            <button
              onClick={() => setBotState('stopped')}
              className="w-full flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors mt-1"
            >
              <AlertOctagon size={11} /> Emergency Stop
            </button>
          </div>

          {/* Status dots */}
          <div className="mt-4 space-y-1.5 px-1">
            <p className="text-[10px] text-gray-600 uppercase tracking-wider px-1 mb-2">Services</p>
            {[
              { label: 'Kalshi API', ok: true,  sub: '0.4s' },
              { label: 'News Feed',  ok: true,  sub: '4m ago' },
              { label: 'Database',   ok: true,  sub: 'healthy' },
              { label: 'Model',      ok: true,  sub: 'v1.4' },
            ].map(({ label, ok, sub }) => (
              <div key={label} className="flex items-center justify-between text-[11px]">
                <span className="text-gray-500">{label}</span>
                <div className="flex items-center gap-1.5">
                  <span className="text-gray-600">{sub}</span>
                  <Circle size={6} className={ok ? 'fill-emerald-400 text-emerald-400' : 'fill-red-400 text-red-400'} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </aside>

      {/* Main */}
      <div className="flex flex-col flex-1 overflow-hidden">
        {/* Header */}
        <header className="h-[50px] bg-[#0b1220]/90 border-b border-[#1a2640]/60 flex items-center justify-between px-5 flex-shrink-0 backdrop-blur-sm" style={{ boxShadow: '0 1px 20px rgba(0,0,0,0.3)' }}>
          <div className="flex items-center gap-4 text-xs">
            <span className="text-gray-500">Uptime: <span className="text-gray-300">4h 23m 11s</span></span>
            <span className="text-gray-500">Bankroll: <span className="text-white font-semibold">$1,000.00</span></span>
            <span className="text-gray-500">P&L: <span className="gradient-text-green font-bold text-sm count-shimmer">+${animatedPnl.toFixed(2)}</span></span>
          </div>
          <div className="flex items-center gap-3">
            {/* Mode toggle */}
            <div className="flex items-center gap-1 bg-[#111827] border border-[#1a2640] rounded-lg p-0.5">
              <button
                onClick={() => setMode('paper')}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${mode === 'paper' ? 'bg-blue-600 text-white' : 'text-gray-500 hover:text-white'}`}
              >Paper</button>
              <button
                onClick={() => setMode('live')}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${mode === 'live' ? 'bg-emerald-600 text-white' : 'text-gray-500 hover:text-white'}`}
              >Live</button>
            </div>

            {/* Live Mode pill + P&L */}
            {mode === 'live' && (
              <div
                className="flex items-center gap-2 px-3 py-1 rounded-lg border border-emerald-500/40 bg-emerald-500/10 text-xs font-semibold text-emerald-400"
                style={{ boxShadow: '0 0 14px rgba(52,211,153,0.18)' }}
              >
                <span className="radar-ping w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block" />
                Live Mode
                <span className="ml-1 text-emerald-300 font-bold">+$45.01</span>
              </div>
            )}

          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-auto p-5">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
