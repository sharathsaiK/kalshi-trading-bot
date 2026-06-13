// Typed fetch wrappers for the kalshi api_server.py backend (port 8765, proxied via /api)

const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`)
  return res.json()
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`)
  return res.json()
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface LogEntry {
  id: number
  time: string
  level: 'INFO' | 'WARN' | 'ERROR'
  source: string
  message: string
}

export interface Settings {
  yesMinEdge: number
  noMinEdge: number
  yesProbFloor: number
  noProbCeil: number
  confMaxMult: number
  bankroll: number
  kellyFraction: number
  maxPositionPct: number
  baseRatePct: number
  nSamplesMin: number
  maxSpread: number
  minVolume: number
  minTimeClose: number
  noOddsCeil: number
  apiPing: string
  newsPing: string
  harvestPing: string
  retrainRows: string
  transcriptRefresh: string
  minTranscriptChars: string
  autoRetrain: boolean
  liveMode: boolean
}

export interface SpeakerRow { speaker: string; rows: number }

export interface TrainingStats {
  totalRows: number
  preCutoffRows: number
  holdoutRows: number
  realRows: number
  syntheticRows: number
  bySpeaker: SpeakerRow[]
  holdoutBySpeaker: SpeakerRow[]
  modelMtime: string
  seeds: number
  architecture: string
  holdoutCutoff: string
}

export interface SpeakerProfile {
  id: number
  speaker: string
  word: string
  event_type: string
  hit_rate_lifetime: number
  hit_rate_recent: number
  momentum: number
  avg_freq: number
  recency: number
  n_samples_lifetime: number
  n_samples_recent: number
  updated_at: string
}

export interface Transcript {
  id: number
  speaker: string
  event_type: string
  event_ticker: string
  source: string
  event_date: string | null
  fetched_at: string
  chars: number
  preview: string
}

export interface FeatureImportance {
  feature: string
  importance: number
  category: string
}

export interface ModelHistoryEntry {
  version: string
  date: string
  auc: number
  brier: number
  acc: number
  rows: number
  active: boolean
}

export interface ModelMetrics {
  accuracy: number | null
  auc: number | null
  brier: number | null
  history: ModelHistoryEntry[]
}

export interface TrainStatus {
  running: boolean
  seed: number
  progress: number
  done: boolean
}

export type JobName = 'evaluate' | 'harvest' | 'backfill-news' | 'backfill-topic'

export interface JobStatus {
  running: boolean
  job: JobName | null
  output: string
  done: boolean
  error: string | null
}

// ── API calls ─────────────────────────────────────────────────────────────────

export const api = {
  // System
  getLogs: ()                    => get<LogEntry[]>('/system/logs'),
  getSettings: ()                => get<Settings>('/system/settings'),
  saveSettings: (s: Settings)   => post<{ ok: boolean; settings: Settings }>('/system/settings', s),

  // Model
  getTrainingStats: ()           => get<TrainingStats>('/model/training-stats'),
  getProfiles: (params?: { speaker?: string; search?: string }) => {
    const qs = new URLSearchParams()
    if (params?.speaker) qs.set('speaker', params.speaker)
    if (params?.search)  qs.set('search',  params.search)
    const q = qs.toString()
    return get<SpeakerProfile[]>(`/model/profiles${q ? `?${q}` : ''}`)
  },
  getTranscripts: (params?: { speaker?: string; search?: string; offset?: number }) => {
    const qs = new URLSearchParams()
    if (params?.speaker) qs.set('speaker', params.speaker)
    if (params?.search)  qs.set('search',  params.search)
    if (params?.offset)  qs.set('offset',  String(params.offset))
    const q = qs.toString()
    return get<Transcript[]>(`/model/transcripts${q ? `?${q}` : ''}`)
  },
  getMetrics: ()                 => get<ModelMetrics>('/model/metrics'),
  getFeatures: ()                => get<FeatureImportance[]>('/model/features'),
  getTrainStatus: ()             => get<TrainStatus>('/model/train-status'),
  startTraining: ()             => post<{ ok: boolean }>('/model/train', {}),

  // Background jobs (evaluate / harvest / backfill-news / backfill-topic)
  runJob: (job: JobName)        => post<{ ok: boolean; job: JobName }>('/model/run-job', { job }),
  getJobStatus: ()              => get<JobStatus>('/model/job-status'),
}
