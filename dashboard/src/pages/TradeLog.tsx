import { useState, useMemo } from 'react'
import { mockTrades, type Trade } from '../data/mockData'
import { Download, ChevronLeft, ChevronRight } from 'lucide-react'

const PAGE_SIZE = 12

const Card = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`bg-[#111827] border border-[#1f2937] rounded-xl ${className}`}>{children}</div>
)

const SideBadge = ({ side }: { side: 'YES' | 'NO' }) => (
  <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
    side === 'YES' ? 'bg-emerald-500/15 text-emerald-400' : 'bg-blue-500/15 text-blue-400'
  }`}>{side}</span>
)

const ResultBadge = ({ result }: { result: Trade['result'] }) => (
  <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-medium ${
    result === 'WIN'  ? 'bg-emerald-500/15 text-emerald-400' :
    result === 'LOSS' ? 'bg-red-500/15 text-red-400' :
                        'bg-blue-500/10 text-blue-400'
  }`}>
    {result === 'WIN' ? '✓ WIN' : result === 'LOSS' ? '✗ LOSS' : '⏳ OPEN'}
  </span>
)

export default function TradeLog() {
  const [rangeFilter, setRangeFilter] = useState<'1D' | '7D' | '30D' | 'All'>('All')
  const [speakerFilter, setSpeakerFilter] = useState('All')
  const [sideFilter, setSideFilter] = useState('All')
  const [resultFilter, setResultFilter] = useState('All')
  const [page, setPage] = useState(0)

  const today = new Date()
  const speakers = ['All', 'Trump', 'Powell', 'Vance', 'Rubio', 'Hegseth']

  const filtered = useMemo(() => {
    const cutoff = new Date()
    if (rangeFilter === '1D') cutoff.setDate(today.getDate() - 1)
    else if (rangeFilter === '7D') cutoff.setDate(today.getDate() - 7)
    else if (rangeFilter === '30D') cutoff.setDate(today.getDate() - 30)
    else cutoff.setFullYear(2000)

    return mockTrades.filter(t => {
      const d = new Date(t.date)
      if (d < cutoff) return false
      if (speakerFilter !== 'All' && t.speaker !== speakerFilter) return false
      if (sideFilter !== 'All' && t.side !== sideFilter) return false
      if (resultFilter !== 'All' && t.result !== resultFilter) return false
      return true
    })
  }, [rangeFilter, speakerFilter, sideFilter, resultFilter])

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE)
  const pageData = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  const wins  = filtered.filter(t => t.result === 'WIN').length
  const pnl   = filtered.reduce((a, t) => a + t.pnl, 0)
  const evAvg = filtered.length ? (filtered.reduce((a, t) => a + t.ev, 0) / filtered.length) : 0

  const bySpkr = speakers.slice(1).map(sp => {
    const ts = filtered.filter(t => t.speaker === sp)
    const w = ts.filter(t => t.result === 'WIN').length
    return { speaker: sp, bets: ts.length, wins: w, pnl: ts.reduce((a, t) => a + t.pnl, 0) }
  }).filter(s => s.bets > 0)

  return (
    <div className="space-y-4 page-enter">
      {/* Filters */}
      <div className="flex items-center gap-3 flex-wrap">
        {/* Range */}
        <div className="flex items-center gap-0.5 bg-[#111827] border border-[#1f2937] rounded-lg p-0.5">
          {(['1D', '7D', '30D', 'All'] as const).map(r => (
            <button key={r} onClick={() => { setRangeFilter(r); setPage(0) }}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${rangeFilter === r ? 'bg-blue-600 text-white' : 'text-gray-500 hover:text-white'}`}
            >{r}</button>
          ))}
        </div>

        {/* Speaker */}
        <select value={speakerFilter} onChange={e => { setSpeakerFilter(e.target.value); setPage(0) }}
          className="bg-[#111827] border border-[#1f2937] text-xs text-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:border-blue-500"
        >
          {speakers.map(s => <option key={s}>{s}</option>)}
        </select>

        {/* Side */}
        <select value={sideFilter} onChange={e => { setSideFilter(e.target.value); setPage(0) }}
          className="bg-[#111827] border border-[#1f2937] text-xs text-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:border-blue-500"
        >
          {['All', 'YES', 'NO'].map(s => <option key={s}>{s}</option>)}
        </select>

        {/* Result */}
        <select value={resultFilter} onChange={e => { setResultFilter(e.target.value); setPage(0) }}
          className="bg-[#111827] border border-[#1f2937] text-xs text-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:border-blue-500"
        >
          {['All', 'WIN', 'LOSS', 'OPEN'].map(s => <option key={s}>{s}</option>)}
        </select>

        <div className="flex-1" />
        <button className="flex items-center gap-1.5 px-3 py-2 bg-[#111827] border border-[#1f2937] text-xs text-gray-400 rounded-lg hover:text-white transition-colors">
          <Download size={12} /> Export CSV
        </button>
      </div>

      <div className="grid grid-cols-4 gap-4">
        {/* Table */}
        <Card className="col-span-3">
          <table className="w-full text-xs">
            <thead className="sticky-thead">
              <tr className="border-b border-[#1f2937] text-gray-600">
                <th className="text-left px-4 py-3 font-medium">ID</th>
                <th className="text-left px-4 py-3 font-medium">Ticker / Word</th>
                <th className="text-left px-4 py-3 font-medium">Speaker</th>
                <th className="text-center px-4 py-3 font-medium">Side</th>
                <th className="text-right px-4 py-3 font-medium">Prob</th>
                <th className="text-right px-4 py-3 font-medium">Odds</th>
                <th className="text-right px-4 py-3 font-medium">EV</th>
                <th className="text-right px-4 py-3 font-medium">×</th>
                <th className="text-right px-4 py-3 font-medium">P&L</th>
                <th className="text-center px-4 py-3 font-medium">Result</th>
              </tr>
            </thead>
            <tbody>
              {pageData.map(t => (
                <tr key={t.id} className="border-b border-[#1a2030]/40 hover:bg-white/[0.02] transition-colors">
                  <td className="px-4 py-2.5 text-gray-600">#{t.id}</td>
                  <td className="px-4 py-2.5">
                    <span className="text-gray-500">{t.ticker.split('-')[0]}/</span>
                    <span className="text-white font-medium"> {t.word}</span>
                  </td>
                  <td className="px-4 py-2.5 text-gray-400">{t.speaker}</td>
                  <td className="px-4 py-2.5 text-center"><SideBadge side={t.side} /></td>
                  <td className="px-4 py-2.5 text-right text-gray-300">{t.prob.toFixed(2)}</td>
                  <td className="px-4 py-2.5 text-right text-gray-300">{t.odds.toFixed(2)}</td>
                  <td className={`px-4 py-2.5 text-right font-medium ${t.ev >= 0.15 ? 'text-emerald-400' : 'text-gray-400'}`}>
                    +{t.ev.toFixed(3)}
                  </td>
                  <td className="px-4 py-2.5 text-right text-gray-400">{t.contracts}</td>
                  <td className={`px-4 py-2.5 text-right font-medium ${
                    t.pnl > 0 ? 'text-emerald-400' : t.pnl < 0 ? 'text-red-400' : 'text-gray-600'
                  }`}>
                    {t.pnl === 0 ? '—' : `${t.pnl >= 0 ? '+' : ''}$${t.pnl.toFixed(2)}`}
                  </td>
                  <td className="px-4 py-2.5 text-center"><ResultBadge result={t.result} /></td>
                </tr>
              ))}
              {pageData.length === 0 && (
                <tr><td colSpan={10} className="px-4 py-8 text-center text-gray-600">No trades match filters</td></tr>
              )}
            </tbody>
          </table>

          {/* Pagination */}
          <div className="flex items-center justify-between px-4 py-3 border-t border-[#1f2937]">
            <div className="text-xs text-gray-600">
              Showing {Math.min(filtered.length, page * PAGE_SIZE + 1)}–{Math.min(filtered.length, (page + 1) * PAGE_SIZE)} of {filtered.length} trades
            </div>
            <div className="flex items-center gap-1">
              <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}
                className="p-1.5 rounded-lg text-gray-500 hover:text-white hover:bg-white/5 disabled:opacity-30 transition-colors">
                <ChevronLeft size={14} />
              </button>
              {Array.from({ length: Math.min(totalPages, 5) }).map((_, i) => (
                <button key={i} onClick={() => setPage(i)}
                  className={`w-7 h-7 rounded-lg text-xs font-medium transition-colors ${i === page ? 'bg-blue-600 text-white' : 'text-gray-500 hover:text-white hover:bg-white/5'}`}
                >{i + 1}</button>
              ))}
              <button onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))} disabled={page >= totalPages - 1}
                className="p-1.5 rounded-lg text-gray-500 hover:text-white hover:bg-white/5 disabled:opacity-30 transition-colors">
                <ChevronRight size={14} />
              </button>
            </div>
          </div>
        </Card>

        {/* Sidebar breakdown */}
        <div className="space-y-3">
          <Card className="p-4">
            <h3 className="text-[10px] text-gray-500 uppercase tracking-wider mb-3">Summary</h3>
            <div className="space-y-2 text-xs">
              {[
                { label: 'Bets',    value: String(filtered.length) },
                { label: 'Wins',    value: `${wins} (${filtered.length ? Math.round(100*wins/filtered.length) : 0}%)` },
                { label: 'P&L',     value: `${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}`, color: pnl >= 0 ? 'text-emerald-400' : 'text-red-400' },
                { label: 'Avg EV',  value: `+${evAvg.toFixed(3)}` },
              ].map(({ label, value, color }) => (
                <div key={label} className="flex justify-between">
                  <span className="text-gray-500">{label}</span>
                  <span className={`font-medium ${color || 'text-white'}`}>{value}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card className="p-4">
            <h3 className="text-[10px] text-gray-500 uppercase tracking-wider mb-3">By Speaker</h3>
            <div className="space-y-2.5">
              {bySpkr.map(s => (
                <div key={s.speaker} className="text-xs">
                  <div className="flex justify-between mb-1">
                    <span className="text-gray-300">{s.speaker}</span>
                    <span className={s.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                      {s.pnl >= 0 ? '+' : ''}${s.pnl.toFixed(2)}
                    </span>
                  </div>
                  <div className="h-1 bg-[#1a2640] rounded-full">
                    <div className="h-full bg-blue-500 rounded-full" style={{ width: `${(s.bets / filtered.length) * 100}%` }} />
                  </div>
                  <div className="text-[10px] text-gray-600 mt-0.5">{s.bets} bets · {s.bets ? Math.round(100*s.wins/s.bets) : 0}% WR</div>
                </div>
              ))}
            </div>
          </Card>

          <Card className="p-4">
            <h3 className="text-[10px] text-gray-500 uppercase tracking-wider mb-3">By Side</h3>
            {(['YES', 'NO'] as const).map(side => {
              const ts = filtered.filter(t => t.side === side)
              const w  = ts.filter(t => t.result === 'WIN').length
              return (
                <div key={side} className="mb-2 text-xs">
                  <div className="flex justify-between mb-1">
                    <span className={side === 'YES' ? 'text-emerald-400' : 'text-blue-400'}>{side}</span>
                    <span className="text-gray-400">{ts.length} bets · {ts.length ? Math.round(100*w/ts.length) : 0}% WR</span>
                  </div>
                  <div className="h-1 bg-[#1a2640] rounded-full">
                    <div className={`h-full rounded-full ${side === 'YES' ? 'bg-emerald-500' : 'bg-blue-500'}`}
                      style={{ width: `${filtered.length ? (ts.length / filtered.length) * 100 : 0}%` }} />
                  </div>
                </div>
              )
            })}
          </Card>
        </div>
      </div>
    </div>
  )
}
