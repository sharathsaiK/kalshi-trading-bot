import { useState, useEffect, useRef } from 'react'
import { mockLogs, newFeedPool, type ActivityItem } from '../data/mockData'
import { Search, ChevronDown } from 'lucide-react'

const levelColor: Record<string, string> = {
  INFO:  'text-gray-500',
  WARN:  'text-yellow-400',
  ERROR: 'text-red-400',
}

const levelBg: Record<string, string> = {
  INFO:  'bg-gray-500/10 text-gray-400',
  WARN:  'bg-yellow-500/15 text-yellow-400',
  ERROR: 'bg-red-500/15 text-red-400',
}

const sourceColor: Record<string, string> = {
  TRADE:   'text-emerald-400',
  GATE:    'text-gray-500',
  MODEL:   'text-blue-400',
  API:     'text-purple-400',
  PROFILE: 'text-yellow-400',
  NEWS:    'text-orange-400',
  HARVEST: 'text-cyan-400',
  TRAIN:   'text-indigo-400',
  BOT:     'text-gray-300',
  TRANS:   'text-pink-400',
}

export default function Logs() {
  const [logs, setLogs] = useState<ActivityItem[]>(mockLogs)
  const [levelFilter, setLevelFilter] = useState<'All' | 'INFO' | 'WARN' | 'ERROR'>('All')
  const [sourceFilter, setSourceFilter] = useState('All')
  const [search, setSearch] = useState('')
  const [autoScroll, setAutoScroll] = useState(true)
  const bottomRef = useRef<HTMLDivElement>(null)
  const nextId = useRef(mockLogs.length + 1)

  const sources = ['All', 'TRADE', 'GATE', 'MODEL', 'API', 'PROFILE', 'NEWS', 'HARVEST', 'TRAIN', 'TRANS', 'BOT']

  useEffect(() => {
    const iv = setInterval(() => {
      const item = newFeedPool[Math.floor(Math.random() * newFeedPool.length)]
      const now = new Date()
      const time = [now.getHours(), now.getMinutes(), now.getSeconds()]
        .map(n => String(n).padStart(2, '0')).join(':')
      setLogs(prev => [...prev, { ...item, id: nextId.current++, time }].slice(-200))
    }, 3500)
    return () => clearInterval(iv)
  }, [])

  useEffect(() => {
    if (autoScroll) bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs, autoScroll])

  const filtered = logs.filter(l => {
    if (levelFilter !== 'All' && l.level !== levelFilter) return false
    if (sourceFilter !== 'All' && l.source !== sourceFilter) return false
    if (search && !l.message.toLowerCase().includes(search.toLowerCase())) return false
    return true
  })

  return (
    <div className="flex flex-col h-full space-y-3" style={{ height: 'calc(100vh - 120px)' }}>
      {/* Filters */}
      <div className="flex items-center gap-3 flex-wrap flex-shrink-0">
        {/* Level */}
        <div className="flex items-center gap-0.5 bg-[#111827] border border-[#1f2937] rounded-lg p-0.5">
          {(['All', 'INFO', 'WARN', 'ERROR'] as const).map(l => (
            <button key={l} onClick={() => setLevelFilter(l)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                levelFilter === l
                  ? l === 'ERROR' ? 'bg-red-600 text-white'
                  : l === 'WARN'  ? 'bg-yellow-600 text-white'
                  : 'bg-blue-600 text-white'
                  : 'text-gray-500 hover:text-white'
              }`}
            >{l}</button>
          ))}
        </div>

        {/* Source */}
        <select value={sourceFilter} onChange={e => setSourceFilter(e.target.value)}
          className="bg-[#111827] border border-[#1f2937] text-xs text-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:border-blue-500"
        >
          {sources.map(s => <option key={s}>{s}</option>)}
        </select>

        {/* Search */}
        <div className="flex items-center gap-2 bg-[#111827] border border-[#1f2937] rounded-lg px-3 py-2 flex-1 max-w-xs">
          <Search size={12} className="text-gray-600 flex-shrink-0" />
          <input
            value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Search logs..."
            className="bg-transparent text-xs text-gray-300 placeholder-gray-600 flex-1 focus:outline-none"
          />
        </div>

        <div className="flex-1" />

        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span>{filtered.length} entries</span>
        </div>

        {/* Auto-scroll */}
        <button
          onClick={() => setAutoScroll(!autoScroll)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
            autoScroll ? 'bg-blue-600/20 border-blue-500/40 text-blue-400' : 'bg-[#111827] border-[#1f2937] text-gray-500 hover:text-white'
          }`}
        >
          <ChevronDown size={12} className={autoScroll ? 'animate-bounce' : ''} />
          Auto-scroll {autoScroll ? 'on' : 'off'}
        </button>
      </div>

      {/* Log panel */}
      <div className="flex-1 bg-[#080d1a] border border-[#1a2640] rounded-xl overflow-y-auto font-mono">
        <div className="p-4 space-y-0.5">
          {filtered.map(item => (
            <div key={item.id} className="flex items-start gap-3 py-0.5 hover:bg-white/[0.02] px-2 -mx-2 rounded text-[11px] leading-relaxed">
              <span className="text-gray-700 flex-shrink-0 w-16">{item.time}</span>
              <span className={`flex-shrink-0 w-5 font-bold ${levelColor[item.level]}`}>
                {item.level === 'ERROR' ? '✗' : item.level === 'WARN' ? '⚠' : '·'}
              </span>
              <span className={`flex-shrink-0 px-1.5 py-0.5 rounded text-[10px] font-medium ${levelBg[item.level]}`}>
                {item.level}
              </span>
              <span className={`flex-shrink-0 w-16 font-semibold ${sourceColor[item.source] || 'text-gray-400'}`}>
                [{item.source}]
              </span>
              <span className={`flex-1 ${
                item.level === 'ERROR' ? 'text-red-300' :
                item.level === 'WARN'  ? 'text-yellow-300' :
                item.source === 'TRADE' ? 'text-emerald-300' :
                'text-gray-400'
              }`}>
                {item.message}
              </span>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </div>
    </div>
  )
}
