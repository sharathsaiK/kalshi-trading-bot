import { useState, useEffect, useRef, useCallback } from 'react'
import { Save, RotateCcw, Info, Search, ChevronDown, RefreshCw } from 'lucide-react'
import { api, type LogEntry, type Settings } from '../api/client'

type Tab = 'logs' | 'settings'

const Card = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`bg-[#111827] border border-[#1f2937] rounded-xl ${className}`}>{children}</div>
)

// ─── Settings helpers ─────────────────────────────────────────────────────────
const Field = ({
  label, hint, value, onChange, min = 0, max = 1, step = 0.01, unit = '',
}: {
  label: string; hint?: string; value: number; onChange: (v: number) => void
  min?: number; max?: number; step?: number; unit?: string
}) => (
  <div className="mb-4">
    <div className="flex items-center justify-between mb-1.5">
      <div className="flex items-center gap-1.5">
        <label className="text-xs text-gray-300">{label}</label>
        {hint && <Info size={11} className="text-gray-600 cursor-help" aria-label={hint} />}
      </div>
      <div className="flex items-center gap-1">
        <input type="number" value={value} step={step} min={min} max={max}
          onChange={e => onChange(parseFloat(e.target.value) || 0)}
          className="w-20 bg-[#0c1426] border border-[#1a2640] text-xs text-white rounded-md px-2 py-1 text-right focus:outline-none focus:border-blue-500" />
        {unit && <span className="text-xs text-gray-500">{unit}</span>}
      </div>
    </div>
    <input type="range" min={min} max={max} step={step} value={value}
      onChange={e => onChange(parseFloat(e.target.value))}
      className="w-full" />
    <div className="flex justify-between text-[10px] text-gray-700 mt-0.5">
      <span>{min}{unit}</span><span>{max}{unit}</span>
    </div>
  </div>
)

const TextInput = ({ label, hint, value, onChange, unit }: {
  label: string; hint?: string; value: string | number; onChange: (v: string) => void; unit?: string
}) => (
  <div className="flex items-center justify-between mb-3">
    <div className="flex items-center gap-1.5">
      <label className="text-xs text-gray-300">{label}</label>
      {hint && <Info size={11} className="text-gray-600" aria-label={hint} />}
    </div>
    <div className="flex items-center gap-1.5">
      <input value={value} onChange={e => onChange(e.target.value)}
        className="w-24 bg-[#0c1426] border border-[#1a2640] text-xs text-white rounded-md px-2 py-1.5 text-right focus:outline-none focus:border-blue-500" />
      {unit && <span className="text-xs text-gray-500 w-6">{unit}</span>}
    </div>
  </div>
)

const Toggle = ({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) => (
  <div className="flex items-center justify-between mb-3">
    <label className="text-xs text-gray-300">{label}</label>
    <button onClick={() => onChange(!value)}
      className={`w-9 h-5 rounded-full transition-colors relative ${value ? 'bg-blue-600' : 'bg-gray-700'}`}>
      <span className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-all shadow-sm ${value ? 'left-4' : 'left-0.5'}`} />
    </button>
  </div>
)

const SectionTitle = ({ children }: { children: React.ReactNode }) => (
  <h3 className="text-[11px] text-gray-500 uppercase tracking-wider mb-4">{children}</h3>
)

// ─── Logs tab ─────────────────────────────────────────────────────────────────
const levelColor: Record<string, string> = {
  INFO: 'text-gray-500', WARN: 'text-yellow-400', ERROR: 'text-red-400',
}
const levelBg: Record<string, string> = {
  INFO: 'bg-gray-500/10 text-gray-400', WARN: 'bg-yellow-500/15 text-yellow-400', ERROR: 'bg-red-500/15 text-red-400',
}
const sourceColor: Record<string, string> = {
  TRADE: 'text-emerald-400', GATE: 'text-gray-500', MODEL: 'text-blue-400',
  API: 'text-purple-400', PROFILE: 'text-yellow-400', NEWS: 'text-orange-400',
  HARVEST: 'text-cyan-400', TRAIN: 'text-indigo-400', BOT: 'text-gray-300', TRANS: 'text-pink-400',
}

function LogsTab() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [levelFilter, setLevelFilter] = useState<'All' | 'INFO' | 'WARN' | 'ERROR'>('All')
  const [sourceFilter, setSourceFilter] = useState('All')
  const [search, setSearch] = useState('')
  const [autoScroll, setAutoScroll] = useState(true)
  const bottomRef = useRef<HTMLDivElement>(null)
  const sources = ['All', 'TRADE', 'GATE', 'MODEL', 'API', 'PROFILE', 'NEWS', 'HARVEST', 'TRAIN', 'TRANS', 'BOT']

  const fetchLogs = useCallback(async () => {
    try {
      const data = await api.getLogs()
      setLogs(data)
    } catch {
      // backend not running — leave current logs
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchLogs()
    const iv = setInterval(fetchLogs, 5000)
    return () => clearInterval(iv)
  }, [fetchLogs])

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
    <div className="flex flex-col space-y-3" style={{ height: 'calc(100vh - 175px)' }}>
      <div className="flex items-center gap-3 flex-wrap flex-shrink-0">
        <div className="flex items-center gap-0.5 bg-[#111827] border border-[#1f2937] rounded-lg p-0.5">
          {(['All', 'INFO', 'WARN', 'ERROR'] as const).map(l => (
            <button key={l} onClick={() => setLevelFilter(l)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                levelFilter === l
                  ? l === 'ERROR' ? 'bg-red-600 text-white' : l === 'WARN' ? 'bg-yellow-600 text-white' : 'bg-blue-600 text-white'
                  : 'text-gray-500 hover:text-white'
              }`}
            >{l}</button>
          ))}
        </div>
        <select value={sourceFilter} onChange={e => setSourceFilter(e.target.value)}
          className="bg-[#111827] border border-[#1f2937] text-xs text-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:border-blue-500">
          {sources.map(s => <option key={s}>{s}</option>)}
        </select>
        <div className="flex items-center gap-2 bg-[#111827] border border-[#1f2937] rounded-lg px-3 py-2 flex-1 max-w-xs">
          <Search size={12} className="text-gray-600 flex-shrink-0" />
          <input value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Search logs..."
            className="bg-transparent text-xs text-gray-300 placeholder-gray-600 flex-1 focus:outline-none" />
        </div>
        <div className="flex-1" />
        <span className="text-xs text-gray-500">{filtered.length} entries</span>
        <button onClick={fetchLogs}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border bg-[#111827] border-[#1f2937] text-gray-500 hover:text-white transition-colors">
          <RefreshCw size={12} /> Refresh
        </button>
        <button onClick={() => setAutoScroll(!autoScroll)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
            autoScroll ? 'bg-blue-600/20 border-blue-500/40 text-blue-400' : 'bg-[#111827] border-[#1f2937] text-gray-500 hover:text-white'
          }`}>
          <ChevronDown size={12} className={autoScroll ? 'animate-bounce' : ''} />
          Auto-scroll {autoScroll ? 'on' : 'off'}
        </button>
      </div>

      <div className="flex-1 bg-[#080d1a] border border-[#1a2640] rounded-xl overflow-y-auto font-mono">
        <div className="p-4 space-y-0.5">
          {loading && (
            <div className="text-xs text-gray-600 py-4 text-center">Loading logs…</div>
          )}
          {!loading && filtered.length === 0 && (
            <div className="text-xs text-gray-600 py-4 text-center">
              No log entries. Is api_server.py running?
            </div>
          )}
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
                item.source === 'TRADE' ? 'text-emerald-300' : 'text-gray-400'
              }`}>{item.message}</span>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </div>
    </div>
  )
}

// ─── Settings tab ─────────────────────────────────────────────────────────────
const DEFAULT_SETTINGS: Settings = {
  yesMinEdge: 0.22, noMinEdge: 0.10, yesProbFloor: 0.72, noProbCeil: 0.30,
  confMaxMult: 3.0, bankroll: 1000, kellyFraction: 0.25, maxPositionPct: 0.10,
  baseRatePct: 0.05, nSamplesMin: 3, maxSpread: 0.10, minVolume: 100,
  minTimeClose: 30, noOddsCeil: 0.65, apiPing: '0.4', newsPing: '4',
  harvestPing: '24', retrainRows: '25', transcriptRefresh: '6',
  minTranscriptChars: '5000', autoRetrain: true, liveMode: false,
}

function SettingsTab() {
  const [s, setS] = useState<Settings>(DEFAULT_SETTINGS)
  const [loaded, setLoaded] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const set = (k: keyof Settings) => (v: any) => setS(prev => ({ ...prev, [k]: v }))

  useEffect(() => {
    api.getSettings().then(data => {
      setS(data)
      setLoaded(true)
    }).catch(() => setLoaded(true))
  }, [])

  const handleSave = async () => {
    setSaving(true)
    try {
      const res = await api.saveSettings(s)
      setS(res.settings)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } finally {
      setSaving(false)
    }
  }

  const handleReset = async () => {
    try {
      const res = await api.saveSettings(DEFAULT_SETTINGS)
      setS(res.settings)
    } catch {
      setS(DEFAULT_SETTINGS)
    }
  }

  if (!loaded) return <div className="text-xs text-gray-600 py-8 text-center">Loading settings…</div>

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-xs text-gray-500">Changes apply on Save. Bot must be restarted for ping interval changes.</p>
        <div className="flex gap-2">
          <button onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-[#111827] border border-[#1f2937] text-xs text-gray-400 rounded-lg hover:text-white transition-colors">
            <RotateCcw size={12} /> Reset to defaults
          </button>
          <button onClick={handleSave} disabled={saving}
            className={`flex items-center gap-1.5 px-4 py-1.5 text-xs text-white rounded-lg font-medium transition-colors ${
              saved ? 'bg-emerald-600' : 'bg-blue-600 hover:bg-blue-700'
            } disabled:opacity-60`}>
            <Save size={12} /> {saved ? 'Saved!' : saving ? 'Saving…' : 'Save Changes'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <Card className="p-5">
          <SectionTitle>Edge & Probability Gates</SectionTitle>
          <Field label="YES min edge"    hint="Minimum EV required to log a YES bet"    value={s.yesMinEdge}   onChange={set('yesMinEdge')}   min={0} max={0.5} step={0.01} />
          <Field label="NO min edge"     hint="Minimum EV required to log a NO bet"     value={s.noMinEdge}    onChange={set('noMinEdge')}    min={0} max={0.5} step={0.01} />
          <Field label="YES prob floor"  hint="Skip YES bets below this model prob"     value={s.yesProbFloor} onChange={set('yesProbFloor')} min={0.5} max={1.0} step={0.01} />
          <Field label="NO prob ceiling" hint="Skip NO bets above this model prob"      value={s.noProbCeil}   onChange={set('noProbCeil')}   min={0} max={0.5} step={0.01} />
          <Field label="Confidence mult" hint="Max Kelly scaling multiplier"            value={s.confMaxMult}  onChange={set('confMaxMult')}  min={1} max={5} step={0.5} unit="×" />
          <Field label="NO odds ceiling" hint="Skip NO bets when YES ask is above this" value={s.noOddsCeil}  onChange={set('noOddsCeil')}   min={0.3} max={0.9} step={0.05} />
        </Card>

        <Card className="p-5">
          <SectionTitle>Kelly & Position Sizing</SectionTitle>
          <TextInput label="Bankroll"        unit="$"  value={s.bankroll}        onChange={v => set('bankroll')(parseFloat(v))} />
          <Field     label="Kelly fraction"  hint="0 = no bet, 1 = full Kelly"  value={s.kellyFraction}  onChange={set('kellyFraction')}  min={0} max={1} step={0.05} />
          <Field     label="Max position %"  hint="Max % of bankroll per trade" value={s.maxPositionPct} onChange={set('maxPositionPct')} min={0.01} max={0.25} step={0.01} unit="%" />
          <Field     label="Base rate %"     hint="Baseline bet fraction"       value={s.baseRatePct}    onChange={set('baseRatePct')}    min={0.01} max={0.20} step={0.01} unit="%" />
          <TextInput label="n_samples min"   hint="Ignore speakers below this"  value={s.nSamplesMin}    onChange={v => set('nSamplesMin')(parseInt(v))} />
          <div className="mt-5 pt-4 border-t border-[#1f2937]">
            <SectionTitle>Liquidity Gates</SectionTitle>
            <Field     label="Max bid-ask spread" value={s.maxSpread}    onChange={set('maxSpread')}    min={0.02} max={0.25} step={0.01} />
            <TextInput label="Min volume ($)"     value={s.minVolume}    onChange={v => set('minVolume')(parseFloat(v))} unit="$" />
            <TextInput label="Min time to close"  value={s.minTimeClose} onChange={v => set('minTimeClose')(parseInt(v))} unit="s" />
          </div>
        </Card>

        <Card className="p-5">
          <SectionTitle>Ping Intervals</SectionTitle>
          <TextInput label="Kalshi API ping"    value={s.apiPing}           onChange={set('apiPing')}           unit="s" />
          <TextInput label="News collection"    value={s.newsPing}          onChange={set('newsPing')}          unit="h" />
          <TextInput label="Data harvest"       value={s.harvestPing}       onChange={set('harvestPing')}       unit="h" />
          <TextInput label="Retrain trigger"    hint="Retrain after N new real rows" value={s.retrainRows} onChange={set('retrainRows')} unit="rows" />
          <TextInput label="Transcript refresh" value={s.transcriptRefresh} onChange={set('transcriptRefresh')} unit="h" />
          <div className="mt-5 pt-4 border-t border-[#1f2937]">
            <SectionTitle>Misc</SectionTitle>
            <TextInput label="Min transcript chars" value={s.minTranscriptChars} onChange={set('minTranscriptChars')} />
            <Toggle    label="Auto-retrain on threshold" value={s.autoRetrain} onChange={set('autoRetrain')} />
            <Toggle    label="Live mode"                  value={s.liveMode}   onChange={set('liveMode')} />
          </div>
        </Card>
      </div>
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function System() {
  const [tab, setTab] = useState<Tab>('logs')

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-0.5 bg-[#111827] border border-[#1f2937] rounded-lg p-1 w-fit">
        {([
          { key: 'logs',     label: 'Logs'     },
          { key: 'settings', label: 'Settings' },
        ] as { key: Tab; label: string }[]).map(({ key, label }) => (
          <button key={key} onClick={() => setTab(key)}
            className={`px-4 py-1.5 rounded-md text-xs font-medium transition-colors ${
              tab === key ? 'bg-blue-600 text-white' : 'text-gray-500 hover:text-white'
            }`}
          >{label}</button>
        ))}
      </div>

      {tab === 'logs'     && <LogsTab />}
      {tab === 'settings' && <SettingsTab />}
    </div>
  )
}
