import { useState } from 'react'
import { mockSpeakerProfiles, mockTranscripts } from '../data/mockData'
import { Search, ChevronUp, ChevronDown } from 'lucide-react'

const Card = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`bg-[#111827] border border-[#1f2937] rounded-xl ${className}`}>{children}</div>
)

type Tab = 'profiles' | 'transcripts' | 'news'

const MomentumBadge = ({ v }: { v: number }) => (
  <span className={`inline-flex items-center gap-0.5 text-[11px] font-medium ${v > 0 ? 'text-emerald-400' : v < 0 ? 'text-red-400' : 'text-gray-500'}`}>
    {v > 0 ? <ChevronUp size={11} /> : v < 0 ? <ChevronDown size={11} /> : null}
    {v > 0 ? '+' : ''}{v.toFixed(2)}
  </span>
)

const HitBar = ({ v }: { v: number }) => (
  <div className="flex items-center gap-2">
    <div className="flex-1 h-1.5 bg-[#1a2640] rounded-full overflow-hidden">
      <div className="h-full bg-blue-500/70 rounded-full" style={{ width: `${v * 100}%` }} />
    </div>
    <span className="text-gray-300 w-8 text-right">{(v * 100).toFixed(0)}%</span>
  </div>
)

export default function DataBrowser() {
  const [tab, setTab] = useState<Tab>('profiles')
  const [speakerFilter, setSpeakerFilter] = useState('All')
  const [search, setSearch] = useState('')

  const speakers = ['All', 'Trump', 'Powell', 'Vance', 'Rubio', 'Hegseth']

  const filteredProfiles = mockSpeakerProfiles.filter(p => {
    if (speakerFilter !== 'All' && p.speaker !== speakerFilter) return false
    if (search && !p.word.toLowerCase().includes(search.toLowerCase()) && !p.speaker.toLowerCase().includes(search.toLowerCase())) return false
    return true
  })

  return (
    <div className="space-y-4">
      {/* Tab bar */}
      <div className="flex items-center gap-0.5 bg-[#111827] border border-[#1f2937] rounded-lg p-1 w-fit">
        {([
          { key: 'profiles',    label: 'Speaker Profiles' },
          { key: 'transcripts', label: 'Transcripts' },
          { key: 'news',        label: 'News Articles' },
        ] as { key: Tab; label: string }[]).map(({ key, label }) => (
          <button key={key} onClick={() => setTab(key)}
            className={`px-4 py-1.5 rounded-md text-xs font-medium transition-colors ${
              tab === key ? 'bg-blue-600 text-white' : 'text-gray-500 hover:text-white'
            }`}
          >{label}</button>
        ))}
      </div>

      {/* Speaker Profiles */}
      {tab === 'profiles' && (
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-[#111827] border border-[#1f2937] rounded-lg px-3 py-2">
              <Search size={12} className="text-gray-600" />
              <input
                value={search} onChange={e => setSearch(e.target.value)}
                placeholder="Search word or speaker..."
                className="bg-transparent text-xs text-gray-300 placeholder-gray-600 w-48 focus:outline-none"
              />
            </div>
            <select value={speakerFilter} onChange={e => setSpeakerFilter(e.target.value)}
              className="bg-[#111827] border border-[#1f2937] text-xs text-gray-300 rounded-lg px-3 py-2 focus:outline-none"
            >
              {speakers.map(s => <option key={s}>{s}</option>)}
            </select>
            <span className="text-xs text-gray-600">{filteredProfiles.length} profiles</span>
          </div>

          <Card>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[#1f2937] text-gray-600">
                  <th className="text-left px-5 py-3 font-medium">Speaker</th>
                  <th className="text-left px-5 py-3 font-medium">Word</th>
                  <th className="px-5 py-3 font-medium text-right">Hit Rate (lifetime)</th>
                  <th className="px-5 py-3 font-medium text-right">Hit Rate (recent)</th>
                  <th className="px-5 py-3 font-medium text-right">Momentum</th>
                  <th className="px-5 py-3 font-medium text-right">N Samples</th>
                  <th className="px-5 py-3 font-medium text-right">Updated</th>
                </tr>
              </thead>
              <tbody>
                {filteredProfiles.map((p, i) => (
                  <tr key={i} className="border-b border-[#1a2030]/40 hover:bg-white/[0.02] transition-colors">
                    <td className="px-5 py-3 font-medium text-gray-200">{p.speaker}</td>
                    <td className="px-5 py-3 text-gray-300">{p.word}</td>
                    <td className="px-5 py-3"><HitBar v={p.hitRateLifetime} /></td>
                    <td className="px-5 py-3"><HitBar v={p.hitRateRecent} /></td>
                    <td className="px-5 py-3 text-right"><MomentumBadge v={p.momentum} /></td>
                    <td className="px-5 py-3 text-right">
                      <span className={`font-medium ${p.nSamples >= 100 ? 'text-white' : p.nSamples >= 30 ? 'text-yellow-400' : 'text-red-400'}`}>
                        {p.nSamples}
                      </span>
                    </td>
                    <td className="px-5 py-3 text-right text-gray-600">{p.updatedAt}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* Transcripts */}
      {tab === 'transcripts' && (
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-[#111827] border border-[#1f2937] rounded-lg px-3 py-2">
              <Search size={12} className="text-gray-600" />
              <input placeholder="Search transcripts..." className="bg-transparent text-xs text-gray-300 placeholder-gray-600 w-48 focus:outline-none" />
            </div>
            <select className="bg-[#111827] border border-[#1f2937] text-xs text-gray-300 rounded-lg px-3 py-2 focus:outline-none">
              {speakers.map(s => <option key={s}>{s}</option>)}
            </select>
          </div>

          <Card>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[#1f2937] text-gray-600">
                  <th className="text-left px-5 py-3 font-medium">ID</th>
                  <th className="text-left px-5 py-3 font-medium">Ticker</th>
                  <th className="text-left px-5 py-3 font-medium">Speaker</th>
                  <th className="text-left px-5 py-3 font-medium">Date</th>
                  <th className="text-right px-5 py-3 font-medium">Size</th>
                  <th className="text-left px-5 py-3 font-medium">Preview</th>
                </tr>
              </thead>
              <tbody>
                {mockTranscripts.map(t => (
                  <tr key={t.id} className="border-b border-[#1a2030]/40 hover:bg-white/[0.02] cursor-pointer transition-colors group">
                    <td className="px-5 py-3 text-gray-600">#{t.id}</td>
                    <td className="px-5 py-3 text-gray-300 font-medium">{t.ticker}</td>
                    <td className="px-5 py-3 text-gray-400">{t.speaker}</td>
                    <td className="px-5 py-3 text-gray-400">{t.date}</td>
                    <td className="px-5 py-3 text-right text-gray-400">{(t.chars / 1000).toFixed(1)}k chars</td>
                    <td className="px-5 py-3 text-gray-600 group-hover:text-gray-400 transition-colors truncate max-w-xs">
                      {t.preview}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* News */}
      {tab === 'news' && (
        <Card className="p-8">
          <div className="text-center text-gray-600">
            <p className="text-sm mb-2">News article cache</p>
            <p className="text-xs">Articles are fetched per-event and stored temporarily. Run a harvest to populate.</p>
            <button className="mt-4 px-4 py-2 bg-blue-600/20 border border-blue-500/30 text-blue-400 text-xs rounded-lg hover:bg-blue-600/30 transition-colors">
              Fetch Latest News
            </button>
          </div>
        </Card>
      )}
    </div>
  )
}
