import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Radio, ListOrdered, BarChart2, Brain, Server,
  Users, CandlestickChart, BookOpen, Search, ArrowRight,
  TrendingUp, Hash,
} from 'lucide-react'

interface Command {
  id: string
  label: string
  sub?: string
  icon: React.ElementType
  color: string
  action: () => void
  category: string
}

interface Props {
  open: boolean
  onClose: () => void
}

export default function CommandPalette({ open, onClose }: Props) {
  const [query, setQuery]     = useState('')
  const [cursor, setCursor]   = useState(0)
  const inputRef              = useRef<HTMLInputElement>(null)
  const navigate              = useNavigate()

  const go = (path: string) => { navigate(path); onClose() }

  const commands: Command[] = [
    { id: 'live',      label: 'Live',             sub: 'Real-time bot activity',         icon: Radio,             color: 'text-emerald-400', action: () => go('/live'),      category: 'Pages' },
    { id: 'markets',   label: 'Markets',           sub: 'Signal evaluation queue',        icon: CandlestickChart,  color: 'text-blue-400',    action: () => go('/markets'),   category: 'Pages' },
    { id: 'history',   label: 'Trade History',     sub: 'All logged trades',              icon: ListOrdered,       color: 'text-blue-400',    action: () => go('/history'),   category: 'Pages' },
    { id: 'analytics', label: 'Analytics',         sub: 'P&L, win rate, drawdown',        icon: BarChart2,         color: 'text-purple-400',  action: () => go('/analytics'), category: 'Pages' },
    { id: 'model',     label: 'Model',             sub: 'LightGBM training & calibration',icon: Brain,             color: 'text-blue-400',    action: () => go('/model'),     category: 'Pages' },
    { id: 'speakers',  label: 'Speaker Profiles',  sub: 'Hit rates, blocklist, history',  icon: Users,             color: 'text-yellow-400',  action: () => go('/speakers'),  category: 'Pages' },
    { id: 'news',      label: 'News & Transcripts',sub: 'Cached training data',           icon: BookOpen,          color: 'text-orange-400',  action: () => go('/news'),      category: 'Pages' },
    { id: 'system',    label: 'System',            sub: 'Services & health checks',       icon: Server,            color: 'text-gray-400',    action: () => go('/system'),    category: 'Pages' },
    { id: 'trump',     label: 'Donald Trump',      sub: '1,240 samples · 65% hit rate',   icon: Hash,              color: 'text-red-400',     action: () => { go('/speakers') }, category: 'Speakers' },
    { id: 'powell',    label: 'Jerome Powell',     sub: '420 samples · 81% hit rate',     icon: Hash,              color: 'text-blue-400',    action: () => { go('/speakers') }, category: 'Speakers' },
    { id: 'vance',     label: 'JD Vance',          sub: '96 samples · 58% hit rate',      icon: Hash,              color: 'text-gray-400',    action: () => { go('/speakers') }, category: 'Speakers' },
    { id: 'tariff',    label: 'TRUMP-TARIFF-SOTU', sub: 'YES · EV +24¢ · Kelly ×8',       icon: TrendingUp,        color: 'text-emerald-400', action: () => go('/markets'),   category: 'Markets' },
    { id: 'rate',      label: 'POWELL-RATE-FED',   sub: 'YES · EV +23¢ · Kelly ×9',       icon: TrendingUp,        color: 'text-emerald-400', action: () => go('/markets'),   category: 'Markets' },
  ]

  const filtered = query.trim() === ''
    ? commands
    : commands.filter(c =>
        c.label.toLowerCase().includes(query.toLowerCase()) ||
        (c.sub || '').toLowerCase().includes(query.toLowerCase()) ||
        c.category.toLowerCase().includes(query.toLowerCase())
      )

  // Group by category
  const groups = filtered.reduce<Record<string, Command[]>>((acc, c) => {
    acc[c.category] = acc[c.category] || []
    acc[c.category].push(c)
    return acc
  }, {})

  const flat = Object.values(groups).flat()

  useEffect(() => {
    if (open) {
      setCursor(0)
      setQuery('')
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [open])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!open) return
      if (e.key === 'Escape') { onClose(); return }
      if (e.key === 'ArrowDown') { e.preventDefault(); setCursor(c => Math.min(c + 1, flat.length - 1)) }
      if (e.key === 'ArrowUp')   { e.preventDefault(); setCursor(c => Math.max(c - 1, 0)) }
      if (e.key === 'Enter' && flat[cursor]) { flat[cursor].action() }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, flat, cursor, onClose])

  if (!open) return null

  let globalIdx = 0

  return (
    <div
      className="palette-backdrop fixed inset-0 z-50 flex items-start justify-center pt-[15vh]"
      style={{ background: 'rgba(4,8,18,0.85)', backdropFilter: 'blur(8px)' }}
      onClick={onClose}
    >
      <div
        className="palette-panel w-full max-w-xl mx-4 rounded-2xl overflow-hidden"
        style={{
          background: 'linear-gradient(160deg, #111c2e 0%, #0d1525 100%)',
          border: '1px solid rgba(96,165,250,0.2)',
          boxShadow: '0 0 0 1px rgba(255,255,255,0.04), 0 32px 80px rgba(0,0,0,0.6), 0 0 60px rgba(59,130,246,0.08)',
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 py-3.5 border-b border-white/[0.06]">
          <Search size={15} className="text-gray-500 flex-shrink-0" />
          <input
            ref={inputRef}
            value={query}
            onChange={e => { setQuery(e.target.value); setCursor(0) }}
            placeholder="Search pages, markets, speakers..."
            className="flex-1 bg-transparent text-sm text-white placeholder-gray-600 outline-none"
          />
          <kbd className="text-[10px] text-gray-600 bg-white/5 border border-white/10 rounded px-1.5 py-0.5">ESC</kbd>
        </div>

        {/* Results */}
        <div className="max-h-[360px] overflow-y-auto py-2">
          {flat.length === 0 ? (
            <p className="px-4 py-8 text-xs text-gray-600 text-center">No results for "{query}"</p>
          ) : (
            Object.entries(groups).map(([category, items]) => (
              <div key={category}>
                <p className="px-4 pt-3 pb-1.5 text-[10px] font-semibold text-gray-600 uppercase tracking-widest">
                  {category}
                </p>
                {items.map(cmd => {
                  const idx = globalIdx++
                  const active = idx === cursor
                  const Icon = cmd.icon
                  return (
                    <button
                      key={cmd.id}
                      onMouseEnter={() => setCursor(idx)}
                      onClick={cmd.action}
                      className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                        active ? 'bg-blue-500/10' : 'hover:bg-white/[0.03]'
                      }`}
                    >
                      <div className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 ${
                        active ? 'bg-blue-500/20' : 'bg-white/5'
                      }`}>
                        <Icon size={13} className={active ? cmd.color : 'text-gray-500'} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className={`text-xs font-medium ${active ? 'text-white' : 'text-gray-300'}`}>{cmd.label}</p>
                        {cmd.sub && <p className="text-[10px] text-gray-600 truncate">{cmd.sub}</p>}
                      </div>
                      {active && <ArrowRight size={12} className="text-gray-600 flex-shrink-0" />}
                    </button>
                  )
                })}
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center gap-4 px-4 py-2.5 border-t border-white/[0.04] text-[10px] text-gray-700">
          <span className="flex items-center gap-1"><kbd className="bg-white/5 border border-white/10 rounded px-1 py-0.5">↑↓</kbd> navigate</span>
          <span className="flex items-center gap-1"><kbd className="bg-white/5 border border-white/10 rounded px-1 py-0.5">↵</kbd> open</span>
          <span className="flex items-center gap-1"><kbd className="bg-white/5 border border-white/10 rounded px-1 py-0.5">esc</kbd> close</span>
        </div>
      </div>
    </div>
  )
}
