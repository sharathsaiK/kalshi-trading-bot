import { useState } from 'react'
import { CheckCircle, XCircle, Clock, TrendingUp, TrendingDown, RefreshCw } from 'lucide-react'
import { mockMarkets, type MarketRow } from '../data/mockData'

interface ActionToast { id: number; text: string; type: 'approve' | 'skip'; leaving: boolean }

type Filter = 'all' | 'bet' | 'watch' | 'skip'

const StatusBadge = ({ status, reason }: { status: MarketRow['status'], reason?: string }) => {
  if (status === 'bet')   return <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/15 text-emerald-400">BET</span>
  if (status === 'watch') return <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-yellow-500/15 text-yellow-400">WATCH</span>
  return (
    <div className="flex flex-col gap-0.5">
      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-gray-500/15 text-gray-500 w-fit">SKIP</span>
      {reason && <span className="text-[9px] text-gray-600">{reason}</span>}
    </div>
  )
}

const SideBadge = ({ side }: { side: 'YES' | 'NO' }) => (
  <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${side === 'YES' ? 'bg-emerald-500/15 text-emerald-400' : 'bg-blue-500/15 text-blue-400'}`}>
    {side}
  </span>
)

let toastId = 0

export default function Markets() {
  const [filter, setFilter]   = useState<Filter>('all')
  const [markets, setMarkets] = useState<MarketRow[]>(mockMarkets)
  const [toasts, setToasts]   = useState<ActionToast[]>([])

  const filtered = filter === 'all' ? markets : markets.filter(m => m.status === filter)

  const betCount   = markets.filter(m => m.status === 'bet').length
  const watchCount = markets.filter(m => m.status === 'watch').length
  const skipCount  = markets.filter(m => m.status === 'skip').length
  const totalEv    = markets.filter(m => m.status === 'bet').reduce((s, m) => s + m.ev, 0)

  const fireToast = (text: string, type: 'approve' | 'skip') => {
    const id = ++toastId
    setToasts(prev => [...prev, { id, text, type, leaving: false }])
    setTimeout(() => {
      setToasts(prev => prev.map(t => t.id === id ? { ...t, leaving: true } : t))
      setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 280)
    }, 2800)
  }

  const approve = (ticker: string, word: string) => {
    setMarkets(prev => prev.map(m =>
      m.ticker === ticker && m.word === word ? { ...m, status: 'bet' } : m
    ))
    fireToast(`Approved ${word}`, 'approve')
  }

  const skip = (ticker: string, word: string) => {
    setMarkets(prev => prev.map(m =>
      m.ticker === ticker && m.word === word ? { ...m, status: 'skip', skipReason: 'manual skip' } : m
    ))
    fireToast(`Skipped ${word}`, 'skip')
  }

  return (
    <div className="space-y-4 page-enter">
      {/* Action toasts */}
      <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 flex flex-col items-center gap-2 pointer-events-none">
        {toasts.map(t => (
          <div key={t.id} className={`${t.leaving ? 'action-toast-out' : 'action-toast-in'} flex items-center gap-2 px-4 py-2 rounded-xl border text-xs font-semibold shadow-xl backdrop-blur-sm ${
            t.type === 'approve'
              ? 'bg-emerald-950/90 border-emerald-500/40 text-emerald-400'
              : 'bg-red-950/90 border-red-500/40 text-red-400'
          }`}>
            {t.type === 'approve' ? <CheckCircle size={12} /> : <XCircle size={12} />}
            {t.text}
          </div>
        ))}
      </div>

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-base font-semibold text-white">Markets</h1>
          <p className="text-xs text-gray-500 mt-0.5">Live signal evaluation queue — approve or skip each market</p>
        </div>
        <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#1a2640] text-gray-400 hover:text-white text-xs transition-colors">
          <RefreshCw size={11} /> Refresh
        </button>
      </div>

      {/* Stat strip */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { label: 'Queued',   value: markets.length,  gradient: 'gradient-text-blue' },
          { label: 'Bet',      value: betCount,        gradient: 'gradient-text-green' },
          { label: 'Watching', value: watchCount,      gradient: 'gradient-text-gold' },
          { label: 'Avg EV',   value: `+${(totalEv / Math.max(betCount, 1) * 100).toFixed(1)}¢`, gradient: 'gradient-text-blue' },
        ].map(({ label, value, gradient }) => (
          <div key={label} className="card-glow card-hover">
            <div className="card-glow-inner px-4 py-3">
              <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-1">{label}</p>
              <p className={`text-xl font-bold ${gradient}`}>{value}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Filter tabs */}
      <div className="flex items-center gap-1 bg-[#0e1521] border border-[#1a2640] rounded-lg p-1 w-fit">
        {(['all', 'bet', 'watch', 'skip'] as Filter[]).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded-md text-xs font-medium capitalize transition-colors ${
              filter === f ? 'bg-blue-600 text-white' : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            {f === 'all' ? `All (${markets.length})` : f === 'bet' ? `Bet (${betCount})` : f === 'watch' ? `Watch (${watchCount})` : `Skip (${skipCount})`}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="bg-[#0e1521] border border-[#1a2640] rounded-xl overflow-auto max-h-[calc(100vh-280px)]">
        <table className="w-full">
          <thead className="sticky-thead">
            <tr className="border-b border-[#1a2640]">
              {['MARKET', 'SPEAKER', 'OUR PROB', 'MKT PRICE', 'EV', 'SIDE', 'CONTRACTS', 'VOLUME', 'SPREAD', 'CLOSES', 'STATUS', 'ACTION'].map(col => (
                <th key={col} className="px-4 py-3 text-left text-[10px] font-semibold text-gray-500 uppercase tracking-widest whitespace-nowrap">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((m, i) => (
              <tr
                key={`${m.ticker}-${m.word}`}
                className={`border-b border-[#1a2640]/50 transition-colors hover:bg-white/[0.02] ${i === filtered.length - 1 ? 'border-b-0' : ''}`}
              >
                {/* Market */}
                <td className="px-4 py-3">
                  <p className="text-xs font-semibold text-blue-400">{m.word}</p>
                  <p className="text-[10px] text-gray-600 mt-0.5 truncate max-w-[140px]">{m.ticker}</p>
                </td>

                {/* Speaker */}
                <td className="px-4 py-3">
                  <span className="text-xs text-gray-400">{m.speaker.split(' ').pop()}</span>
                </td>

                {/* Our prob */}
                <td className="px-4 py-3">
                  <span className={`text-xs font-semibold ${m.ourProb >= 0.70 ? 'text-emerald-400' : m.ourProb <= 0.30 ? 'text-blue-400' : 'text-gray-300'}`}>
                    {(m.ourProb * 100).toFixed(0)}%
                  </span>
                </td>

                {/* Mkt price */}
                <td className="px-4 py-3">
                  <span className="text-xs text-gray-400">{(m.mktPrice * 100).toFixed(0)}¢</span>
                </td>

                {/* EV */}
                <td className="px-4 py-3">
                  <div className="flex items-center gap-1">
                    {m.ev > 0
                      ? <TrendingUp size={11} className="text-emerald-400" />
                      : <TrendingDown size={11} className="text-red-400" />
                    }
                    <span className={`text-xs font-semibold ${m.ev >= 0.15 ? 'text-emerald-400' : m.ev >= 0.05 ? 'text-yellow-400' : 'text-gray-500'}`}>
                      {m.ev >= 0 ? '+' : ''}{(m.ev * 100).toFixed(1)}¢
                    </span>
                  </div>
                </td>

                {/* Side */}
                <td className="px-4 py-3">
                  <SideBadge side={m.evSide} />
                </td>

                {/* Contracts */}
                <td className="px-4 py-3">
                  <span className={`text-xs font-semibold ${m.contracts > 0 ? 'text-white' : 'text-gray-600'}`}>
                    {m.contracts > 0 ? `×${m.contracts}` : '—'}
                  </span>
                </td>

                {/* Volume */}
                <td className="px-4 py-3">
                  <span className={`text-xs ${m.volume >= 2000 ? 'text-gray-300' : m.volume >= 500 ? 'text-gray-500' : 'text-red-400/70'}`}>
                    ${(m.volume).toLocaleString()}
                  </span>
                </td>

                {/* Spread */}
                <td className="px-4 py-3">
                  <span className={`text-xs ${m.spread <= 0.04 ? 'text-emerald-400/70' : m.spread <= 0.08 ? 'text-yellow-400/70' : 'text-red-400/70'}`}>
                    {(m.spread * 100).toFixed(0)}¢
                  </span>
                </td>

                {/* Closes */}
                <td className="px-4 py-3">
                  <div className="flex items-center gap-1 text-[11px] text-gray-500">
                    <Clock size={10} />
                    {m.closeTime}
                  </div>
                </td>

                {/* Status */}
                <td className="px-4 py-3">
                  <StatusBadge status={m.status} reason={m.skipReason} />
                </td>

                {/* Action */}
                <td className="px-4 py-3">
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => approve(m.ticker, m.word)}
                      disabled={m.status === 'bet'}
                      className={`flex items-center gap-1 px-2 py-1 rounded text-[11px] font-medium transition-colors ${
                        m.status === 'bet'
                          ? 'bg-emerald-500/10 text-emerald-600 cursor-default'
                          : 'bg-emerald-500/15 text-emerald-400 hover:bg-emerald-500/25'
                      }`}
                    >
                      <CheckCircle size={10} /> Approve
                    </button>
                    <button
                      onClick={() => skip(m.ticker, m.word)}
                      disabled={m.status === 'skip'}
                      className={`flex items-center gap-1 px-2 py-1 rounded text-[11px] font-medium transition-colors ${
                        m.status === 'skip'
                          ? 'bg-red-500/10 text-red-700 cursor-default'
                          : 'bg-red-500/15 text-red-400 hover:bg-red-500/25'
                      }`}
                    >
                      <XCircle size={10} /> Skip
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
