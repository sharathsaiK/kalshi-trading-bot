import { useState } from 'react'
import { RefreshCw, Download, Play, CheckCircle, Clock } from 'lucide-react'
import { mockModelHistory, mockSpeakerStats } from '../data/mockData'

const Card = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`bg-[#111827] border border-[#1f2937] rounded-xl ${className}`}>{children}</div>
)

export default function Training() {
  const [training, setTraining] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentSeed, setCurrentSeed] = useState(0)

  const startTraining = () => {
    setTraining(true)
    setProgress(0)
    setCurrentSeed(1)
    let seed = 1
    const iv = setInterval(() => {
      seed++
      setCurrentSeed(seed)
      setProgress(Math.round((seed / 11) * 100))
      if (seed >= 11) {
        clearInterval(iv)
        setTimeout(() => { setTraining(false); setProgress(100) }, 800)
      }
    }, 600)
  }

  const totalRows = 1201
  const speakerRows = [
    { speaker: 'Trump',   rows: 784 },
    { speaker: 'Powell',  rows: 218 },
    { speaker: 'Vance',   rows: 103 },
    { speaker: 'Rubio',   rows: 73  },
    { speaker: 'Hegseth', rows: 37  },
    { speaker: 'Other',   rows: 16  },
  ]

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-4">
        {/* Current model */}
        <Card className="p-5">
          <h3 className="text-[11px] text-gray-500 uppercase tracking-wider mb-4">Current Model</h3>
          <div className="space-y-2.5 text-xs">
            {[
              { label: 'Version',        value: 'v1.4',        color: 'text-blue-400' },
              { label: 'Architecture',   value: '11-seed DART ensemble + LR + isotonic' },
              { label: 'Last trained',   value: '2026-05-29 14:22 UTC' },
              { label: 'Age',            value: '2 days ago',  color: 'text-yellow-400' },
              { label: 'Training rows',  value: '1,163 (pre-cutoff)' },
              { label: 'Features',       value: '23' },
              { label: 'Seeds',          value: '11 (42,7,13,99,2024,…)' },
              { label: 'Holdout cutoff', value: '2026-03-01' },
            ].map(({ label, value, color }) => (
              <div key={label} className="flex justify-between">
                <span className="text-gray-500">{label}</span>
                <span className={`font-medium ${color || 'text-gray-300'}`}>{value}</span>
              </div>
            ))}
          </div>

          <div className="mt-5 pt-4 border-t border-[#1f2937]">
            <h4 className="text-[10px] text-gray-600 uppercase tracking-wider mb-3">Holdout Performance</h4>
            <div className="grid grid-cols-3 gap-2 text-center">
              {[
                { label: 'Accuracy', value: '83.1%', color: 'text-emerald-400' },
                { label: 'AUC',      value: '0.808', color: 'text-blue-400' },
                { label: 'Brier',    value: '0.177', color: 'text-purple-400' },
              ].map(({ label, value, color }) => (
                <div key={label} className="bg-[#0c1426] rounded-lg py-2 px-1">
                  <div className={`text-base font-bold ${color}`}>{value}</div>
                  <div className="text-[10px] text-gray-600">{label}</div>
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* Training data */}
        <Card className="p-5">
          <h3 className="text-[11px] text-gray-500 uppercase tracking-wider mb-4">Training Data</h3>
          <div className="space-y-2 text-xs mb-5">
            {[
              { label: 'Total rows',        value: '1,240' },
              { label: 'Pre-cutoff rows',   value: '1,163' },
              { label: 'Holdout rows',      value: '77' },
              { label: 'Real rows',         value: '312',  color: 'text-emerald-400' },
              { label: 'Synthetic rows',    value: '851',  color: 'text-gray-500' },
              { label: 'New since retrain', value: '+38',  color: 'text-yellow-400' },
            ].map(({ label, value, color }) => (
              <div key={label} className="flex justify-between">
                <span className="text-gray-500">{label}</span>
                <span className={`font-medium ${color || 'text-gray-300'}`}>{value}</span>
              </div>
            ))}
          </div>

          <h4 className="text-[10px] text-gray-600 uppercase tracking-wider mb-3">By Speaker</h4>
          <div className="space-y-2.5">
            {speakerRows.map(s => (
              <div key={s.speaker}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-gray-400">{s.speaker}</span>
                  <span className="text-gray-500">{s.rows.toLocaleString()}</span>
                </div>
                <div className="h-1.5 bg-[#1a2640] rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500/60 rounded-full" style={{ width: `${(s.rows / totalRows) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Actions */}
        <Card className="p-5">
          <h3 className="text-[11px] text-gray-500 uppercase tracking-wider mb-4">Actions</h3>

          {training && (
            <div className="mb-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
              <div className="flex items-center justify-between text-xs mb-2">
                <span className="text-blue-400 font-medium">Training in progress...</span>
                <span className="text-gray-500">Seed {currentSeed}/11</span>
              </div>
              <div className="h-2 bg-[#1a2640] rounded-full overflow-hidden">
                <div className="h-full bg-blue-500 rounded-full transition-all duration-500" style={{ width: `${progress}%` }} />
              </div>
              <div className="text-[10px] text-gray-600 mt-1.5">fold 3/5 · {progress}% complete</div>
            </div>
          )}

          {progress === 100 && !training && (
            <div className="mb-4 p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg flex items-center gap-2 text-xs text-emerald-400">
              <CheckCircle size={13} /> Training complete — pending merge
            </div>
          )}

          <div className="space-y-2">
            <button
              onClick={startTraining}
              disabled={training}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-xs text-white rounded-lg font-medium transition-colors"
            >
              <RefreshCw size={13} className={training ? 'animate-spin' : ''} />
              {training ? 'Training...' : 'Retrain Model Now'}
            </button>
            <button
              disabled={progress !== 100}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-emerald-600/80 hover:bg-emerald-600 disabled:opacity-40 disabled:cursor-not-allowed text-xs text-white rounded-lg font-medium transition-colors"
            >
              <Download size={13} /> Merge Pending Model → Active
            </button>
            <button className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-[#0c1426] border border-[#1a2640] hover:border-blue-500/50 text-xs text-gray-400 hover:text-white rounded-lg font-medium transition-colors">
              <Play size={13} /> Run Holdout Evaluation
            </button>
            <button className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-[#0c1426] border border-[#1a2640] hover:border-blue-500/50 text-xs text-gray-400 hover:text-white rounded-lg font-medium transition-colors">
              <RefreshCw size={13} /> Harvest New Training Data
            </button>
            <button className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-[#0c1426] border border-[#1a2640] hover:border-blue-500/50 text-xs text-gray-400 hover:text-white rounded-lg font-medium transition-colors">
              <RefreshCw size={13} /> Backfill News Features
            </button>
            <button className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-[#0c1426] border border-[#1a2640] hover:border-blue-500/50 text-xs text-gray-400 hover:text-white rounded-lg font-medium transition-colors">
              <RefreshCw size={13} /> Backfill Topic Match
            </button>
          </div>
        </Card>
      </div>

      {/* Model history */}
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
            {mockModelHistory.map((m, i) => (
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
                    : <span className="text-gray-600">archived</span>
                  }
                </td>
                <td className="py-2.5 text-right">
                  {!m.active && (
                    <button className="text-[10px] text-gray-600 hover:text-blue-400 transition-colors">Rollback</button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
