import { useState } from 'react'
import { ShieldOff, ShieldCheck, TrendingUp, TrendingDown, X, FileText, Newspaper, ExternalLink, UserPlus } from 'lucide-react'
import { mockTranscripts, mockNews } from '../data/mockData'

interface Speaker {
  name: string
  role: string
  samples: number
  hitRate: number
  avgWordsPerTalk: number
  pnl: number
  blocklisted: boolean
}

const mockSpeakers: Speaker[] = [
  { name: 'Jerome Powell',    role: 'Fed Chair',       samples: 420,  hitRate: 0.81, avgWordsPerTalk: 14, pnl:  425.00, blocklisted: false },
  { name: 'Donald Trump',     role: 'President',       samples: 1240, hitRate: 0.65, avgWordsPerTalk: 22, pnl:  627.00, blocklisted: false },
  { name: 'Janet Yellen',     role: 'Sec. Treasury',   samples: 180,  hitRate: 0.52, avgWordsPerTalk: 11, pnl:  208.00, blocklisted: false },
  { name: 'Elizabeth Warren', role: 'Senator',         samples: 218,  hitRate: 0.44, avgWordsPerTalk:  8, pnl: -867.00, blocklisted: true  },
  { name: 'JD Vance',         role: 'Vice President',  samples: 96,   hitRate: 0.58, avgWordsPerTalk: 17, pnl:  112.00, blocklisted: false },
  { name: 'Marco Rubio',      role: 'Sec. of State',   samples: 74,   hitRate: 0.61, avgWordsPerTalk: 13, pnl:   88.00, blocklisted: false },
  { name: 'Pete Hegseth',     role: 'Sec. of Defense', samples: 42,   hitRate: 0.55, avgWordsPerTalk:  9, pnl:  -34.00, blocklisted: false },
]

type PanelTab = 'transcripts' | 'news'

const BLANK_FORM = { name: '', role: '', hitRate: '0.60', avgWordsPerTalk: '12' }

// ─── Add Speaker Modal ────────────────────────────────────────────────────────
function AddSpeakerModal({ onAdd, onClose }: { onAdd: (s: Speaker) => void; onClose: () => void }) {
  const [form, setForm] = useState(BLANK_FORM)
  const [error, setError] = useState('')

  const set = (k: keyof typeof BLANK_FORM) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm(prev => ({ ...prev, [k]: e.target.value }))

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!form.name.trim()) { setError('Name is required'); return }
    if (!form.role.trim()) { setError('Role is required'); return }
    const hr = parseFloat(form.hitRate)
    if (isNaN(hr) || hr < 0 || hr > 1) { setError('Hit rate must be between 0 and 1'); return }
    const avg = parseInt(form.avgWordsPerTalk)
    if (isNaN(avg) || avg < 1) { setError('Avg words must be a positive number'); return }
    onAdd({ name: form.name.trim(), role: form.role.trim(), hitRate: hr, avgWordsPerTalk: avg, samples: 0, pnl: 0, blocklisted: false })
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Modal */}
      <div className="relative bg-[#0e1521] border border-[#1a2640] rounded-2xl shadow-2xl w-full max-w-md mx-4 animate-[fadeSlideIn_0.18s_ease-out]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#1a2640]">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-blue-500/15 flex items-center justify-center">
              <UserPlus size={14} className="text-blue-400" />
            </div>
            <h2 className="text-sm font-semibold text-white">Add Speaker</h2>
          </div>
          <button onClick={onClose} className="text-gray-600 hover:text-gray-300 transition-colors">
            <X size={15} />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={submit} className="px-6 py-5 space-y-4">
          {/* Name */}
          <div>
            <label className="block text-[10px] text-gray-500 uppercase tracking-widest mb-1.5">Full Name *</label>
            <input
              autoFocus
              value={form.name}
              onChange={set('name')}
              placeholder="e.g. Elon Musk"
              className="w-full bg-[#0c1426] border border-[#1a2640] rounded-lg px-3 py-2.5 text-sm text-white placeholder-gray-700 focus:outline-none focus:border-blue-500/60 transition-colors"
            />
          </div>

          {/* Role */}
          <div>
            <label className="block text-[10px] text-gray-500 uppercase tracking-widest mb-1.5">Role / Title *</label>
            <input
              value={form.role}
              onChange={set('role')}
              placeholder="e.g. CEO, Tesla"
              className="w-full bg-[#0c1426] border border-[#1a2640] rounded-lg px-3 py-2.5 text-sm text-white placeholder-gray-700 focus:outline-none focus:border-blue-500/60 transition-colors"
            />
          </div>

          {/* Hit Rate + Avg Words (side by side) */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-[10px] text-gray-500 uppercase tracking-widest mb-1.5">Est. Hit Rate</label>
              <div className="relative">
                <input
                  value={form.hitRate}
                  onChange={set('hitRate')}
                  placeholder="0.60"
                  className="w-full bg-[#0c1426] border border-[#1a2640] rounded-lg px-3 py-2.5 text-sm text-white placeholder-gray-700 focus:outline-none focus:border-blue-500/60 transition-colors"
                />
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-600">0–1</span>
              </div>
            </div>
            <div>
              <label className="block text-[10px] text-gray-500 uppercase tracking-widest mb-1.5">Avg Words / Talk</label>
              <input
                value={form.avgWordsPerTalk}
                onChange={set('avgWordsPerTalk')}
                placeholder="12"
                className="w-full bg-[#0c1426] border border-[#1a2640] rounded-lg px-3 py-2.5 text-sm text-white placeholder-gray-700 focus:outline-none focus:border-blue-500/60 transition-colors"
              />
            </div>
          </div>

          {/* Error */}
          {error && (
            <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">{error}</p>
          )}

          {/* Hint */}
          <p className="text-[10px] text-gray-600">Samples and P&amp;L start at 0 and will update as the bot collects data for this speaker.</p>

          {/* Actions */}
          <div className="flex gap-2 pt-1">
            <button type="button" onClick={onClose}
              className="flex-1 px-4 py-2 rounded-lg border border-[#1a2640] text-xs text-gray-400 hover:text-white hover:border-[#243050] transition-colors">
              Cancel
            </button>
            <button type="submit"
              className="flex-1 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white transition-colors">
              Add Speaker
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function SpeakerProfiles() {
  const [speakers, setSpeakers] = useState<Speaker[]>(mockSpeakers)
  const [selected, setSelected] = useState<Speaker | null>(null)
  const [panelTab, setPanelTab] = useState<PanelTab>('transcripts')
  const [showAdd, setShowAdd] = useState(false)

  const addSpeaker = (s: Speaker) => setSpeakers(prev => [...prev, s])

  const toggleBlocklist = (name: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setSpeakers(prev => prev.map(s => s.name === name ? { ...s, blocklisted: !s.blocklisted } : s))
  }

  const selectSpeaker = (s: Speaker) => {
    if (selected?.name === s.name) {
      setSelected(null)
    } else {
      setSelected(s)
      setPanelTab('transcripts')
    }
  }

  const totalPnl = speakers.reduce((sum, s) => sum + s.pnl, 0)

  const speakerTranscripts = selected
    ? mockTranscripts.filter(t => t.speaker === selected.name)
    : []

  const speakerNews = selected
    ? mockNews.filter(n => n.speaker === selected.name)
    : []

  return (
    <div className="flex gap-4 h-full page-enter">
      {/* Add speaker modal */}
      {showAdd && <AddSpeakerModal onAdd={addSpeaker} onClose={() => setShowAdd(false)} />}

      {/* Left: table */}
      <div className={`flex flex-col gap-4 transition-all duration-300 ${selected ? 'flex-1' : 'w-full'}`}>
        {/* Header row */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-base font-semibold text-white">Speaker Profiles</h1>
            <p className="text-xs text-gray-500 mt-0.5">Click a row to view transcripts and news · toggle blocklist per speaker</p>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <span className="text-gray-500">
              Total P&L: <span className={`font-semibold ${totalPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
              </span>
            </span>
            <span className="text-gray-500">
              Speakers: <span className="text-white font-semibold">{speakers.length}</span>
            </span>
            <span className="text-gray-500">
              Blocklisted: <span className="text-red-400 font-semibold">{speakers.filter(s => s.blocklisted).length}</span>
            </span>
            <button
              onClick={() => setShowAdd(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded-lg text-white font-medium transition-colors"
            >
              <UserPlus size={12} />
              Add Speaker
            </button>
          </div>
        </div>

        {/* Table */}
        <div className="bg-[#0e1521] border border-[#1a2640] rounded-xl overflow-hidden">
          <table className="w-full">
            <thead className="sticky-thead">
              <tr className="border-b border-[#1a2640]">
                {['SPEAKER', 'ROLE', 'SAMPLES', 'HIT RATE', 'AVG WORDS / TALK', 'P&L', 'BLOCKLISTED'].map(col => (
                  <th key={col} className="px-5 py-3 text-left text-[10px] font-semibold text-gray-500 uppercase tracking-widest whitespace-nowrap">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {speakers.map((s, i) => (
                <tr
                  key={s.name}
                  onClick={() => selectSpeaker(s)}
                  className={`border-b border-[#1a2640]/50 cursor-pointer transition-colors
                    ${selected?.name === s.name ? 'bg-blue-500/10 border-blue-500/20' : 'hover:bg-white/[0.02]'}
                    ${s.blocklisted ? 'opacity-50' : ''}
                    ${i === speakers.length - 1 ? 'border-b-0' : ''}
                  `}
                >
                  {/* Speaker */}
                  <td className="px-5 py-3.5">
                    <div className="flex items-center gap-2.5">
                      <div className="w-7 h-7 rounded-full bg-[#1a2640] flex items-center justify-center text-[11px] font-bold text-gray-400 flex-shrink-0">
                        {s.name.split(' ').map(n => n[0]).join('').slice(0, 2)}
                      </div>
                      <span className="text-sm font-medium text-white">{s.name}</span>
                    </div>
                  </td>

                  {/* Role */}
                  <td className="px-5 py-3.5">
                    <span className="text-xs text-gray-400">{s.role}</span>
                  </td>

                  {/* Samples */}
                  <td className="px-5 py-3.5">
                    <span className="text-xs text-gray-300 font-medium">{s.samples.toLocaleString()}</span>
                  </td>

                  {/* Hit Rate */}
                  <td className="px-5 py-3.5">
                    <div className="flex items-center gap-2">
                      <div className="w-14 h-1 bg-[#1a2640] rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${s.hitRate >= 0.70 ? 'bg-emerald-400' : s.hitRate >= 0.55 ? 'bg-blue-400' : 'bg-gray-500'}`}
                          style={{ width: `${s.hitRate * 100}%` }}
                        />
                      </div>
                      <span className={`text-xs font-semibold ${s.hitRate >= 0.70 ? 'text-emerald-400' : s.hitRate >= 0.55 ? 'text-blue-400' : 'text-gray-400'}`}>
                        {(s.hitRate * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>

                  {/* Avg words */}
                  <td className="px-5 py-3.5">
                    <span className="text-xs text-gray-300">{s.avgWordsPerTalk}</span>
                  </td>

                  {/* P&L */}
                  <td className="px-5 py-3.5">
                    <div className="flex items-center gap-1">
                      {s.pnl >= 0
                        ? <TrendingUp size={12} className="text-emerald-400" />
                        : <TrendingDown size={12} className="text-red-400" />
                      }
                      <span className={`text-xs font-semibold ${s.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {s.pnl >= 0 ? '+' : ''}${s.pnl.toFixed(2)}
                      </span>
                    </div>
                  </td>

                  {/* Blocklisted toggle */}
                  <td className="px-5 py-3.5">
                    <button
                      onClick={(e) => toggleBlocklist(s.name, e)}
                      className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors ${
                        s.blocklisted
                          ? 'bg-red-500/15 text-red-400 hover:bg-red-500/25'
                          : 'bg-[#1a2640] text-gray-500 hover:bg-[#1f2f4a] hover:text-gray-300'
                      }`}
                    >
                      {s.blocklisted ? <><ShieldOff size={10} /> Yes</> : <><ShieldCheck size={10} /> No</>}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="text-[11px] text-gray-600">
          Blocklisted speakers are skipped during YES bet evaluation. NO bets are still considered if edge exists.
        </p>
      </div>

      {/* Right: detail panel */}
      {selected && (
        <div className="w-[380px] flex-shrink-0 bg-[#0e1521] border border-[#1a2640] rounded-xl flex flex-col overflow-hidden">
          {/* Panel header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-[#1a2640]">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-full bg-[#1a2640] flex items-center justify-center text-[11px] font-bold text-gray-400">
                {selected.name.split(' ').map(n => n[0]).join('').slice(0, 2)}
              </div>
              <div>
                <p className="text-sm font-semibold text-white leading-tight">{selected.name}</p>
                <p className="text-[11px] text-gray-500">{selected.role}</p>
              </div>
            </div>
            <button onClick={() => setSelected(null)} className="text-gray-600 hover:text-gray-300 transition-colors">
              <X size={15} />
            </button>
          </div>

          {/* Tabs */}
          <div className="flex border-b border-[#1a2640]">
            {([
              { id: 'transcripts', label: 'Transcripts', icon: FileText,  count: speakerTranscripts.length },
              { id: 'news',        label: 'News',         icon: Newspaper, count: speakerNews.length },
            ] as const).map(({ id, label, icon: Icon, count }) => (
              <button
                key={id}
                onClick={() => setPanelTab(id)}
                className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium transition-colors border-b-2 ${
                  panelTab === id
                    ? 'border-blue-500 text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-300'
                }`}
              >
                <Icon size={12} />
                {label}
                <span className={`ml-0.5 px-1.5 py-0.5 rounded text-[10px] ${panelTab === id ? 'bg-blue-500/20 text-blue-400' : 'bg-[#1a2640] text-gray-500'}`}>
                  {count}
                </span>
              </button>
            ))}
          </div>

          {/* Panel body */}
          <div className="flex-1 overflow-y-auto">
            {/* Transcripts tab */}
            {panelTab === 'transcripts' && (
              <div className="divide-y divide-[#1a2640]/60">
                {speakerTranscripts.length === 0 ? (
                  <p className="px-4 py-8 text-xs text-gray-600 text-center">No transcripts cached for this speaker</p>
                ) : speakerTranscripts.map(t => (
                  <div key={t.id} className="px-4 py-3 hover:bg-white/[0.02] transition-colors">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <p className="text-[11px] font-semibold text-blue-400 truncate">{t.ticker}</p>
                        <p className="text-[11px] text-gray-400 mt-0.5 leading-relaxed line-clamp-2">{t.preview}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 mt-2 text-[10px] text-gray-600">
                      <span>{t.date}</span>
                      <span>·</span>
                      <span>{t.source}</span>
                      <span>·</span>
                      <span>{(t.chars / 1000).toFixed(1)}k chars</span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* News tab */}
            {panelTab === 'news' && (
              <div className="divide-y divide-[#1a2640]/60">
                {speakerNews.length === 0 ? (
                  <p className="px-4 py-8 text-xs text-gray-600 text-center">No news articles cached for this speaker</p>
                ) : speakerNews.map(n => (
                  <div key={n.id} className="px-4 py-3 hover:bg-white/[0.02] transition-colors group">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5 mb-1">
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#1a2640] text-gray-400 font-medium">{n.word}</span>
                          <span className={`text-[10px] font-semibold ${n.relevance >= 0.85 ? 'text-emerald-400' : n.relevance >= 0.70 ? 'text-blue-400' : 'text-gray-500'}`}>
                            {(n.relevance * 100).toFixed(0)}% rel
                          </span>
                        </div>
                        <p className="text-[11px] text-gray-300 leading-relaxed">{n.title}</p>
                      </div>
                      <a href={n.url} className="text-gray-700 hover:text-gray-400 transition-colors flex-shrink-0 mt-0.5">
                        <ExternalLink size={11} />
                      </a>
                    </div>
                    <div className="flex items-center gap-3 mt-1.5 text-[10px] text-gray-600">
                      <span>{n.source}</span>
                      <span>·</span>
                      <span>{n.date}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
