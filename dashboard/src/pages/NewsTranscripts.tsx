import { useState } from 'react'
import { Newspaper, FileText, ExternalLink, Search } from 'lucide-react'
import { mockNews, mockTranscripts } from '../data/mockData'

type Tab = 'news' | 'transcripts'

const speakers = ['All', 'Donald Trump', 'Jerome Powell', 'JD Vance', 'Marco Rubio', 'Elizabeth Warren', 'Pete Hegseth']

export default function NewsTranscripts() {
  const [tab, setTab]               = useState<Tab>('news')
  const [speakerFilter, setSpeaker] = useState('All')
  const [search, setSearch]         = useState('')

  const filteredNews = mockNews.filter(n => {
    const matchSpeaker = speakerFilter === 'All' || n.speaker === speakerFilter
    const matchSearch  = !search || n.title.toLowerCase().includes(search.toLowerCase()) || n.word.toLowerCase().includes(search.toLowerCase())
    return matchSpeaker && matchSearch
  })

  const filteredTranscripts = mockTranscripts.filter(t => {
    const matchSpeaker = speakerFilter === 'All' || t.speaker === speakerFilter
    const matchSearch  = !search || t.ticker.toLowerCase().includes(search.toLowerCase()) || t.preview.toLowerCase().includes(search.toLowerCase())
    return matchSpeaker && matchSearch
  })

  return (
    <div className="space-y-4 page-enter">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-base font-semibold text-white">News & Transcripts</h1>
          <p className="text-xs text-gray-500 mt-0.5">Cached data used to build speaker profiles and model predictions</p>
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span>{mockNews.length} articles</span>
          <span className="text-gray-700">·</span>
          <span>{mockTranscripts.length} transcripts</span>
        </div>
      </div>

      {/* Controls row */}
      <div className="flex items-center gap-3">
        {/* Tabs */}
        <div className="flex items-center gap-1 bg-[#0e1521] border border-[#1a2640] rounded-lg p-1">
          <button
            onClick={() => setTab('news')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${tab === 'news' ? 'bg-blue-600 text-white' : 'text-gray-500 hover:text-gray-300'}`}
          >
            <Newspaper size={11} /> News <span className={`ml-0.5 px-1.5 py-0.5 rounded text-[10px] ${tab === 'news' ? 'bg-white/20' : 'bg-[#1a2640] text-gray-600'}`}>{filteredNews.length}</span>
          </button>
          <button
            onClick={() => setTab('transcripts')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${tab === 'transcripts' ? 'bg-blue-600 text-white' : 'text-gray-500 hover:text-gray-300'}`}
          >
            <FileText size={11} /> Transcripts <span className={`ml-0.5 px-1.5 py-0.5 rounded text-[10px] ${tab === 'transcripts' ? 'bg-white/20' : 'bg-[#1a2640] text-gray-600'}`}>{filteredTranscripts.length}</span>
          </button>
        </div>

        {/* Speaker filter */}
        <select
          value={speakerFilter}
          onChange={e => setSpeaker(e.target.value)}
          className="bg-[#0e1521] border border-[#1a2640] rounded-lg px-3 py-2 text-xs text-gray-300 focus:outline-none focus:border-blue-500/50"
        >
          {speakers.map(s => <option key={s} value={s}>{s}</option>)}
        </select>

        {/* Search */}
        <div className="relative flex-1 max-w-xs">
          <Search size={11} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-600" />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search..."
            className="w-full bg-[#0e1521] border border-[#1a2640] rounded-lg pl-8 pr-3 py-2 text-xs text-gray-300 placeholder-gray-600 focus:outline-none focus:border-blue-500/50"
          />
        </div>
      </div>

      {/* News tab */}
      {tab === 'news' && (
        <div className="bg-[#0e1521] border border-[#1a2640] rounded-xl overflow-hidden">
          <table className="w-full">
            <thead className="sticky-thead">
              <tr className="border-b border-[#1a2640]">
                {['HEADLINE', 'SPEAKER', 'WORD', 'RELEVANCE', 'SOURCE', 'DATE', ''].map(col => (
                  <th key={col} className="px-5 py-3 text-left text-[10px] font-semibold text-gray-500 uppercase tracking-widest">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredNews.length === 0 ? (
                <tr><td colSpan={7} className="px-5 py-10 text-center text-xs text-gray-600">No articles match your filters</td></tr>
              ) : filteredNews.map((n, i) => (
                <tr key={n.id} className={`border-b border-[#1a2640]/50 hover:bg-white/[0.02] transition-colors ${i === filteredNews.length - 1 ? 'border-b-0' : ''}`}>
                  <td className="px-5 py-3.5 max-w-[320px]">
                    <p className="text-xs text-gray-300 leading-relaxed">{n.title}</p>
                  </td>
                  <td className="px-5 py-3.5">
                    <span className="text-xs text-gray-400">{n.speaker.split(' ').pop()}</span>
                  </td>
                  <td className="px-5 py-3.5">
                    <span className="px-2 py-0.5 rounded bg-[#1a2640] text-[10px] text-gray-400 font-medium">{n.word}</span>
                  </td>
                  <td className="px-5 py-3.5">
                    <div className="flex items-center gap-2">
                      <div className="w-12 h-1 bg-[#1a2640] rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${n.relevance >= 0.85 ? 'bg-emerald-400' : n.relevance >= 0.70 ? 'bg-blue-400' : 'bg-gray-500'}`}
                          style={{ width: `${n.relevance * 100}%` }}
                        />
                      </div>
                      <span className={`text-xs font-semibold ${n.relevance >= 0.85 ? 'text-emerald-400' : n.relevance >= 0.70 ? 'text-blue-400' : 'text-gray-500'}`}>
                        {(n.relevance * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-5 py-3.5">
                    <span className="text-xs text-gray-500">{n.source}</span>
                  </td>
                  <td className="px-5 py-3.5">
                    <span className="text-xs text-gray-600">{n.date}</span>
                  </td>
                  <td className="px-5 py-3.5">
                    <a href={n.url} className="text-gray-700 hover:text-gray-400 transition-colors">
                      <ExternalLink size={11} />
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Transcripts tab */}
      {tab === 'transcripts' && (
        <div className="bg-[#0e1521] border border-[#1a2640] rounded-xl overflow-hidden">
          <table className="w-full">
            <thead className="sticky-thead">
              <tr className="border-b border-[#1a2640]">
                {['EVENT', 'SPEAKER', 'DATE', 'SOURCE', 'SIZE', 'PREVIEW'].map(col => (
                  <th key={col} className="px-5 py-3 text-left text-[10px] font-semibold text-gray-500 uppercase tracking-widest">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredTranscripts.length === 0 ? (
                <tr><td colSpan={6} className="px-5 py-10 text-center text-xs text-gray-600">No transcripts match your filters</td></tr>
              ) : filteredTranscripts.map((t, i) => (
                <tr key={t.id} className={`border-b border-[#1a2640]/50 hover:bg-white/[0.02] transition-colors ${i === filteredTranscripts.length - 1 ? 'border-b-0' : ''}`}>
                  <td className="px-5 py-3.5">
                    <span className="text-xs font-semibold text-blue-400">{t.ticker}</span>
                  </td>
                  <td className="px-5 py-3.5">
                    <span className="text-xs text-gray-400">{t.speaker}</span>
                  </td>
                  <td className="px-5 py-3.5">
                    <span className="text-xs text-gray-600">{t.date}</span>
                  </td>
                  <td className="px-5 py-3.5">
                    <span className="text-xs text-gray-500">{t.source}</span>
                  </td>
                  <td className="px-5 py-3.5">
                    <span className="text-xs text-gray-500">{(t.chars / 1000).toFixed(1)}k chars</span>
                  </td>
                  <td className="px-5 py-3.5 max-w-[360px]">
                    <p className="text-[11px] text-gray-500 truncate">{t.preview}</p>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
