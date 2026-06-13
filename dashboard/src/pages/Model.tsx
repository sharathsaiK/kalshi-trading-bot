import { useState, useEffect, useCallback, useRef } from 'react'
import { RefreshCw, Download, Play, CheckCircle, Search, ChevronUp, ChevronDown, AlertCircle } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { api, type TrainingStats, type SpeakerProfile, type Transcript, type FeatureImportance, type ModelMetrics, type JobName, type JobStatus } from '../api/client'

const Card = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`bg-[#111827] border border-[#1f2937] rounded-xl ${className}`}>{children}</div>
)

type Tab = 'training' | 'profiles' | 'transcripts' | 'features'

const categoryColor: Record<string, string> = {
  profile: '#3b82f6',
  word:    '#10b981',
  news:    '#f59e0b',
  event:   '#a78bfa',
  market:  '#6b7280',
  other:   '#374151',
}

// ─── Shared sub-components ────────────────────────────────────────────────────
const MomentumBadge = ({ v }: { v: number }) => (
  <span className={`inline-flex items-center gap-0.5 text-[11px] font-medium ${v > 0 ? 'text-emerald-400' : v < 0 ? 'text-red-400' : 'text-gray-500'}`}>
    {v > 0 ? <ChevronUp size={11} /> : v < 0 ? <ChevronDown size={11} /> : null}
    {v > 0 ? '+' : ''}{v.toFixed(2)}
  </span>
)

const HitBar = ({ v }: { v: number }) => (
  <div className="flex items-center gap-2">
    <div className="flex-1 h-1.5 bg-[#1a2640] rounded-full overflow-hidden">
      <div className="h-full bg-blue-500/70 rounded-full" style={{ width: `${Math.min(v * 100, 100)}%` }} />
    </div>
    <span className="text-gray-300 w-8 text-right">{(v * 100).toFixed(0)}%</span>
  </div>
)

const Spinner = () => (
  <div className="flex items-center justify-center py-10">
    <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
  </div>
)

// ─── Training tab ─────────────────────────────────────────────────────────────
function TrainingTab() {
  const [stats, setStats] = useState<TrainingStats | null>(null)
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [training, setTraining] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentSeed, setCurrentSeed] = useState(0)
  const [trainDone, setTrainDone] = useState(false)
  const [job, setJob] = useState<JobStatus | null>(null)
  const outputRef = useRef<HTMLPreElement>(null)

  const fetchStats = useCallback(async () => {
    try {
      const [s, m] = await Promise.all([api.getTrainingStats(), api.getMetrics()])
      setStats(s)
      setMetrics(m)
    } catch { /* backend not running */ }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { fetchStats() }, [fetchStats])

  // Sync job state on mount (pick up any job that started before this render)
  useEffect(() => {
    api.getJobStatus().then(st => { if (st.running || st.done) setJob(st) }).catch(() => {})
  }, [])

  // Poll retrain progress
  useEffect(() => {
    if (!training) return
    const iv = setInterval(async () => {
      try {
        const st = await api.getTrainStatus()
        setCurrentSeed(st.seed)
        setProgress(st.progress)
        if (!st.running && st.done) {
          setTraining(false)
          setTrainDone(true)
          fetchStats()
        }
      } catch { /* ignore */ }
    }, 800)
    return () => clearInterval(iv)
  }, [training, fetchStats])

  // Poll background job status
  useEffect(() => {
    if (!job?.running) return
    const iv = setInterval(async () => {
      try {
        const st = await api.getJobStatus()
        setJob(st)
        if (outputRef.current) outputRef.current.scrollTop = outputRef.current.scrollHeight
        if (!st.running && st.done) fetchStats()
      } catch { /* ignore */ }
    }, 800)
    return () => clearInterval(iv)
  }, [job?.running, fetchStats])

  const startTraining = async () => {
    setTraining(true)
    setTrainDone(false)
    setProgress(0)
    setCurrentSeed(0)
    try { await api.startTraining() } catch { setTraining(false) }
  }

  const runJob = async (name: JobName) => {
    try {
      await api.runJob(name)
      setJob({ running: true, job: name, output: '', done: false, error: null })
    } catch { /* conflict or server down */ }
  }

  const seeds = stats?.seeds ?? 11
  const jobBusy = training || (job?.running ?? false)

  const ACTION_BTNS: { label: string; job: JobName; icon: typeof Play }[] = [
    { label: 'Run Holdout Evaluation',   job: 'evaluate',       icon: Play     },
    { label: 'Harvest New Training Data', job: 'harvest',        icon: Download },
    { label: 'Backfill News Features',   job: 'backfill-news',  icon: RefreshCw },
    { label: 'Backfill Topic Match',     job: 'backfill-topic', icon: RefreshCw },
  ]

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-4">
        {/* Current Model */}
        <Card className="p-5">
          <h3 className="text-[11px] text-gray-500 uppercase tracking-wider mb-4">Current Model</h3>
          {loading ? <Spinner /> : (
            <>
              <div className="space-y-2.5 text-xs">
                {[
                  { label: 'Architecture',   value: stats?.architecture ?? '—' },
                  { label: 'Last trained',   value: stats?.modelMtime ?? '—' },
                  { label: 'Training rows',  value: stats ? stats.preCutoffRows.toLocaleString() + ' (pre-cutoff)' : '—' },
                  { label: 'Seeds',          value: stats ? String(stats.seeds) : '—' },
                  { label: 'Holdout cutoff', value: stats?.holdoutCutoff ?? '—' },
                ].map(({ label, value }) => (
                  <div key={label} className="flex justify-between">
                    <span className="text-gray-500">{label}</span>
                    <span className="font-medium text-gray-300 text-right max-w-[55%] truncate">{value}</span>
                  </div>
                ))}
              </div>
              {metrics && (metrics.accuracy != null) && (
                <div className="mt-5 pt-4 border-t border-[#1f2937]">
                  <h4 className="text-[10px] text-gray-600 uppercase tracking-wider mb-3">Holdout Performance</h4>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    {[
                      { label: 'Accuracy', value: `${metrics.accuracy.toFixed(1)}%`, color: 'text-emerald-400' },
                      { label: 'AUC',      value: metrics.auc!.toFixed(3),            color: 'text-blue-400'    },
                      { label: 'Brier',    value: metrics.brier!.toFixed(4),           color: 'text-purple-400'  },
                    ].map(({ label, value, color }) => (
                      <div key={label} className="bg-[#0c1426] rounded-lg py-2 px-1">
                        <div className={`text-base font-bold ${color}`}>{value}</div>
                        <div className="text-[10px] text-gray-600">{label}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </Card>

        {/* Training Data */}
        <Card className="p-5">
          <h3 className="text-[11px] text-gray-500 uppercase tracking-wider mb-4">Training Data</h3>
          {loading ? <Spinner /> : (
            <>
              <div className="space-y-2 text-xs mb-5">
                {[
                  { label: 'Total rows',     value: stats?.totalRows.toLocaleString() ?? '—' },
                  { label: 'Pre-cutoff',     value: stats?.preCutoffRows.toLocaleString() ?? '—' },
                  { label: 'Holdout rows',   value: stats?.holdoutRows.toLocaleString() ?? '—', color: 'text-purple-400' },
                  { label: 'Real rows',      value: stats?.realRows.toLocaleString() ?? '—',    color: 'text-emerald-400' },
                  { label: 'Synthetic rows', value: stats?.syntheticRows.toLocaleString() ?? '—', color: 'text-gray-500' },
                ].map(({ label, value, color }) => (
                  <div key={label} className="flex justify-between">
                    <span className="text-gray-500">{label}</span>
                    <span className={`font-medium ${color || 'text-gray-300'}`}>{value}</span>
                  </div>
                ))}
              </div>
              <h4 className="text-[10px] text-gray-600 uppercase tracking-wider mb-3">By Speaker</h4>
              <div className="space-y-2.5">
                {(stats?.bySpeaker ?? []).map(s => {
                  const total = stats?.preCutoffRows ?? 1
                  return (
                    <div key={s.speaker}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-400">{s.speaker}</span>
                        <span className="text-gray-500">{s.rows.toLocaleString()}</span>
                      </div>
                      <div className="h-1.5 bg-[#1a2640] rounded-full overflow-hidden">
                        <div className="h-full bg-blue-500/60 rounded-full" style={{ width: `${(s.rows / total) * 100}%` }} />
                      </div>
                    </div>
                  )
                })}
              </div>
            </>
          )}
        </Card>

        {/* Actions */}
        <Card className="p-5">
          <h3 className="text-[11px] text-gray-500 uppercase tracking-wider mb-4">Actions</h3>

          {/* Retrain progress */}
          {training && (
            <div className="mb-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
              <div className="flex items-center justify-between text-xs mb-2">
                <span className="text-blue-400 font-medium">Training in progress…</span>
                <span className="text-gray-500">Seed {currentSeed}/{seeds}</span>
              </div>
              <div className="h-2 bg-[#1a2640] rounded-full overflow-hidden">
                <div className="h-full bg-blue-500 rounded-full transition-all duration-500" style={{ width: `${progress}%` }} />
              </div>
              <div className="text-[10px] text-gray-600 mt-1.5">{progress}% complete</div>
            </div>
          )}
          {trainDone && !training && (
            <div className="mb-3 p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg flex items-center gap-2 text-xs text-emerald-400">
              <CheckCircle size={13} /> Training complete
            </div>
          )}

          {/* Background job status */}
          {job && !job.running && job.done && (
            <div className={`mb-3 p-3 rounded-lg flex items-center gap-2 text-xs ${
              job.error
                ? 'bg-red-500/10 border border-red-500/30 text-red-400'
                : 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400'
            }`}>
              {job.error ? <AlertCircle size={13} /> : <CheckCircle size={13} />}
              {job.error ? `${job.job} failed: ${job.error}` : `${job.job} complete`}
            </div>
          )}

          {/* Live output while job runs */}
          {job?.running && job.output && (
            <pre ref={outputRef}
              className="mb-3 p-2 bg-[#080d1a] border border-[#1a2640] rounded-lg text-[10px] text-gray-500 font-mono overflow-y-auto max-h-28 whitespace-pre-wrap">
              {job.output}
            </pre>
          )}

          <div className="space-y-2">
            <button onClick={startTraining} disabled={jobBusy}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-xs text-white rounded-lg font-medium transition-colors">
              <RefreshCw size={13} className={training ? 'animate-spin' : ''} />
              {training ? 'Training…' : 'Retrain Model Now'}
            </button>

            <button onClick={fetchStats}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-[#0c1426] border border-[#1a2640] hover:border-blue-500/50 text-xs text-gray-400 hover:text-white rounded-lg font-medium transition-colors">
              <RefreshCw size={13} /> Refresh Stats
            </button>

            {ACTION_BTNS.map(({ label, job: jobName, icon: Icon }) => {
              const isThis = job?.job === jobName
              const running = isThis && job?.running
              return (
                <button key={jobName} onClick={() => runJob(jobName)} disabled={jobBusy}
                  className={`w-full flex items-center justify-center gap-2 px-4 py-2.5 border text-xs rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                    running
                      ? 'bg-blue-600/20 border-blue-500/40 text-blue-400'
                      : 'bg-[#0c1426] border-[#1a2640] hover:border-blue-500/50 text-gray-400 hover:text-white'
                  }`}>
                  <Icon size={13} className={running ? 'animate-spin' : ''} />
                  {running ? `${label}…` : label}
                </button>
              )
            })}
          </div>
        </Card>
      </div>
      {/* Version History */}
      {metrics && metrics.history.length > 0 && (
        <Card className="p-5">
          <h3 className="text-[11px] text-gray-500 uppercase tracking-wider mb-4">Model Version History</h3>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[#1f2937] text-gray-600">
                <th className="text-left pb-2 font-medium">Version</th>
                <th className="text-left pb-2 font-medium">Date Trained</th>
                <th className="text-right pb-2 font-medium">AUC</th>
                <th className="text-right pb-2 font-medium">Brier</th>
                <th className="text-right pb-2 font-medium">Accuracy</th>
                <th className="text-right pb-2 font-medium">Rows</th>
                <th className="text-center pb-2 font-medium">Status</th>
                <th className="text-right pb-2 font-medium"></th>
              </tr>
            </thead>
            <tbody>
              {metrics.history.map((m, i) => (
                <tr key={i} className="border-b border-[#1a2030]/40 hover:bg-white/[0.02] transition-colors">
                  <td className="py-2.5 font-medium text-white">{m.version}</td>
                  <td className="py-2.5 text-gray-400">{m.date}</td>
                  <td className="py-2.5 text-right text-gray-300">{m.auc.toFixed(3)}</td>
                  <td className="py-2.5 text-right text-gray-300">{m.brier.toFixed(4)}</td>
                  <td className="py-2.5 text-right text-gray-300">{m.acc.toFixed(1)}%</td>
                  <td className="py-2.5 text-right text-gray-400">{m.rows.toLocaleString()}</td>
                  <td className="py-2.5 text-center">
                    {m.active
                      ? <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-500/15 text-emerald-400 font-medium">● ACTIVE</span>
                      : <span className="text-gray-600">archived</span>}
                  </td>
                  <td className="py-2.5 text-right">
                    {!m.active && <button className="text-[10px] text-gray-600 hover:text-blue-400 transition-colors">Rollback</button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  )
}

// ─── Profiles tab ─────────────────────────────────────────────────────────────
function ProfilesTab() {
  const [profiles, setProfiles] = useState<SpeakerProfile[]>([])
  const [loading, setLoading] = useState(true)
  const [speakerFilter, setSpeakerFilter] = useState('All')
  const [search, setSearch] = useState('')
  const speakers = ['All', 'Trump', 'Powell', 'Vance', 'Rubio', 'Hegseth']

  const fetchProfiles = useCallback(async () => {
    setLoading(true)
    try {
      const data = await api.getProfiles({ speaker: speakerFilter, search })
      setProfiles(data)
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }, [speakerFilter, search])

  useEffect(() => {
    const t = setTimeout(fetchProfiles, search ? 300 : 0)
    return () => clearTimeout(t)
  }, [fetchProfiles, search])

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 bg-[#111827] border border-[#1f2937] rounded-lg px-3 py-2">
          <Search size={12} className="text-gray-600" />
          <input value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Search word or speaker..."
            className="bg-transparent text-xs text-gray-300 placeholder-gray-600 w-48 focus:outline-none" />
        </div>
        <select value={speakerFilter} onChange={e => setSpeakerFilter(e.target.value)}
          className="bg-[#111827] border border-[#1f2937] text-xs text-gray-300 rounded-lg px-3 py-2 focus:outline-none">
          {speakers.map(s => <option key={s}>{s}</option>)}
        </select>
        <span className="text-xs text-gray-600">{profiles.length} profiles</span>
      </div>
      <Card>
        {loading ? <Spinner /> : (
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
              {profiles.length === 0 && (
                <tr><td colSpan={7} className="px-5 py-8 text-center text-gray-600">
                  No profiles found. Is api_server.py running?
                </td></tr>
              )}
              {profiles.map(p => (
                <tr key={p.id} className="border-b border-[#1a2030]/40 hover:bg-white/[0.02] transition-colors">
                  <td className="px-5 py-3 font-medium text-gray-200">{p.speaker}</td>
                  <td className="px-5 py-3 text-gray-300">{p.word}</td>
                  <td className="px-5 py-3"><HitBar v={p.hit_rate_lifetime} /></td>
                  <td className="px-5 py-3"><HitBar v={p.hit_rate_recent} /></td>
                  <td className="px-5 py-3 text-right"><MomentumBadge v={p.momentum} /></td>
                  <td className="px-5 py-3 text-right">
                    <span className={`font-medium ${p.n_samples_lifetime >= 100 ? 'text-white' : p.n_samples_lifetime >= 30 ? 'text-yellow-400' : 'text-red-400'}`}>
                      {p.n_samples_lifetime}
                    </span>
                  </td>
                  <td className="px-5 py-3 text-right text-gray-600">{p.updated_at?.slice(0, 10) ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  )
}

// ─── Transcripts tab ──────────────────────────────────────────────────────────
function TranscriptsTab() {
  const [transcripts, setTranscripts] = useState<Transcript[]>([])
  const [loading, setLoading] = useState(true)
  const [speakerFilter, setSpeakerFilter] = useState('All')
  const [search, setSearch] = useState('')
  const speakers = ['All', 'Trump', 'Powell', 'Vance', 'Rubio', 'Hegseth']

  const fetchTranscripts = useCallback(async () => {
    setLoading(true)
    try {
      const data = await api.getTranscripts({ speaker: speakerFilter, search })
      setTranscripts(data)
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }, [speakerFilter, search])

  useEffect(() => {
    const t = setTimeout(fetchTranscripts, search ? 300 : 0)
    return () => clearTimeout(t)
  }, [fetchTranscripts, search])

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 bg-[#111827] border border-[#1f2937] rounded-lg px-3 py-2">
          <Search size={12} className="text-gray-600" />
          <input value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Search transcripts..."
            className="bg-transparent text-xs text-gray-300 placeholder-gray-600 w-48 focus:outline-none" />
        </div>
        <select value={speakerFilter} onChange={e => setSpeakerFilter(e.target.value)}
          className="bg-[#111827] border border-[#1f2937] text-xs text-gray-300 rounded-lg px-3 py-2 focus:outline-none">
          {speakers.map(s => <option key={s}>{s}</option>)}
        </select>
        <span className="text-xs text-gray-600">{transcripts.length} transcripts</span>
      </div>
      <Card>
        {loading ? <Spinner /> : (
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
              {transcripts.length === 0 && (
                <tr><td colSpan={6} className="px-5 py-8 text-center text-gray-600">
                  No transcripts found.
                </td></tr>
              )}
              {transcripts.map(t => (
                <tr key={t.id} className="border-b border-[#1a2030]/40 hover:bg-white/[0.02] cursor-pointer transition-colors group">
                  <td className="px-5 py-3 text-gray-600">#{t.id}</td>
                  <td className="px-5 py-3 text-gray-300 font-medium">{t.event_ticker || t.event_type || '—'}</td>
                  <td className="px-5 py-3 text-gray-400">{t.speaker}</td>
                  <td className="px-5 py-3 text-gray-400">{t.event_date?.slice(0, 10) ?? t.fetched_at?.slice(0, 10) ?? '—'}</td>
                  <td className="px-5 py-3 text-right text-gray-400">{(t.chars / 1000).toFixed(1)}k chars</td>
                  <td className="px-5 py-3 text-gray-600 group-hover:text-gray-400 transition-colors truncate max-w-xs">{t.preview}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  )
}

// ─── Features tab ─────────────────────────────────────────────────────────────
function FeaturesTab() {
  const [features, setFeatures] = useState<FeatureImportance[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.getFeatures()
      .then(setFeatures)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const TT = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null
    return (
      <div className="bg-[#0f1a2e] border border-white/10 rounded-lg px-3 py-2 text-xs shadow-xl">
        <p className="text-gray-500 mb-1">{label}</p>
        <p className="font-semibold text-white">{payload[0].value.toFixed(1)}%</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <Card className="p-5">
          <h3 className="text-[11px] text-gray-500 uppercase tracking-wider mb-1">Feature Importance</h3>
          <p className="text-[10px] text-gray-700 mb-4">Mean gain from seed-1 model</p>
          {loading ? <Spinner /> : (
            <ResponsiveContainer width="100%" height={Math.max(220, features.length * 22)}>
              <BarChart data={features} layout="vertical" margin={{ top: 0, right: 16, left: 8, bottom: 0 }}>
                <XAxis type="number" tick={{ fill: '#374151', fontSize: 10 }} tickFormatter={v => `${v}%`} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="feature" tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} width={160} />
                <Tooltip content={<TT />} />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                  {features.map((f, i) => (
                    <Cell key={i} fill={categoryColor[f.category] ?? categoryColor.other} fillOpacity={0.8} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card className="p-5">
          <h3 className="text-[11px] text-gray-500 uppercase tracking-wider mb-4">Legend & Notes</h3>
          <div className="space-y-2 mb-5">
            {Object.entries(categoryColor).filter(([c]) => c !== 'other').map(([cat, color]) => (
              <div key={cat} className="flex items-center gap-2 text-xs">
                <span className="w-3 h-3 rounded-sm flex-shrink-0" style={{ background: color, opacity: 0.8 }} />
                <span className="text-gray-400 capitalize">{cat}</span>
              </div>
            ))}
          </div>
          <div className="pt-4 border-t border-[#1f2937] space-y-3 text-xs text-gray-500">
            <p><span className="text-gray-300">hit_rate_recent</span> — rolling hit rate over last 10 events, by speaker+word</p>
            <p><span className="text-gray-300">hit_rate_word_global</span> — unconditional frequency of word, Bayesian-shrunk (K=15)</p>
            <p><span className="text-gray-300">rel_max / rel_mean</span> — news relevancy TF-IDF scores at event time</p>
            <p><span className="text-gray-300">momentum</span> — hit_rate_recent minus hit_rate_lifetime</p>
            <p><span className="text-gray-300">topic_match</span> — transformer similarity to event topic</p>
          </div>
        </Card>
      </div>
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function Model() {
  const [tab, setTab] = useState<Tab>('training')

  const tabs: { key: Tab; label: string }[] = [
    { key: 'training',    label: 'Training'    },
    { key: 'profiles',    label: 'Profiles'    },
    { key: 'transcripts', label: 'Transcripts' },
    { key: 'features',    label: 'Features'    },
  ]

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-0.5 bg-[#111827] border border-[#1f2937] rounded-lg p-1 w-fit">
        {tabs.map(({ key, label }) => (
          <button key={key} onClick={() => setTab(key)}
            className={`px-4 py-1.5 rounded-md text-xs font-medium transition-colors ${
              tab === key ? 'bg-blue-600 text-white' : 'text-gray-500 hover:text-white'
            }`}
          >{label}</button>
        ))}
      </div>

      {tab === 'training'    && <TrainingTab />}
      {tab === 'profiles'    && <ProfilesTab />}
      {tab === 'transcripts' && <TranscriptsTab />}
      {tab === 'features'    && <FeaturesTab />}
    </div>
  )
}
