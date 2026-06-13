import { useState } from 'react'
import { Save, RotateCcw, Info } from 'lucide-react'

const Card = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`bg-[#111827] border border-[#1f2937] rounded-xl ${className}`}>{children}</div>
)

const SectionTitle = ({ children }: { children: React.ReactNode }) => (
  <h3 className="text-[11px] text-gray-500 uppercase tracking-wider mb-4">{children}</h3>
)

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
        {hint && <Info size={11} className="text-gray-600 cursor-help" title={hint} />}
      </div>
      <div className="flex items-center gap-1">
        <input
          type="number" value={value} step={step} min={min} max={max}
          onChange={e => onChange(parseFloat(e.target.value) || 0)}
          className="w-20 bg-[#0c1426] border border-[#1a2640] text-xs text-white rounded-md px-2 py-1 text-right focus:outline-none focus:border-blue-500"
        />
        {unit && <span className="text-xs text-gray-500">{unit}</span>}
      </div>
    </div>
    <input
      type="range" min={min} max={max} step={step} value={value}
      onChange={e => onChange(parseFloat(e.target.value))}
      className="w-full"
    />
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
      {hint && <Info size={11} className="text-gray-600" title={hint} />}
    </div>
    <div className="flex items-center gap-1.5">
      <input
        value={value} onChange={e => onChange(e.target.value)}
        className="w-24 bg-[#0c1426] border border-[#1a2640] text-xs text-white rounded-md px-2 py-1.5 text-right focus:outline-none focus:border-blue-500"
      />
      {unit && <span className="text-xs text-gray-500 w-6">{unit}</span>}
    </div>
  </div>
)

const Toggle = ({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) => (
  <div className="flex items-center justify-between mb-3">
    <label className="text-xs text-gray-300">{label}</label>
    <button
      onClick={() => onChange(!value)}
      className={`w-9 h-5 rounded-full transition-colors relative ${value ? 'bg-blue-600' : 'bg-gray-700'}`}
    >
      <span className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-all shadow-sm ${value ? 'left-4' : 'left-0.5'}`} />
    </button>
  </div>
)

export default function Settings() {
  const [s, setS] = useState({
    yesMinEdge:    0.22,
    noMinEdge:     0.10,
    yesProbFloor:  0.72,
    noProbCeil:    0.30,
    confMaxMult:   3.0,
    bankroll:      1000,
    kellyFraction: 0.25,
    maxPositionPct: 0.10,
    baseRatePct:   0.05,
    nSamplesMin:   20,
    maxSpread:     0.10,
    minVolume:     100,
    minTimeClose:  30,
    noOddsCeil:    0.65,
    apiPing:       '0.4',
    newsPing:      '4',
    harvestPing:   '24',
    retrainRows:   '25',
    transcriptRefresh: '6',
    minTranscriptChars: '5000',
    autoRetrain:   true,
    liveMode:      false,
  })

  const set = (k: keyof typeof s) => (v: any) => setS(prev => ({ ...prev, [k]: v }))

  return (
    <div className="space-y-4">
      {/* Top bar */}
      <div className="flex items-center justify-between">
        <p className="text-xs text-gray-500">Changes apply on Save. Bot must be restarted for ping interval changes.</p>
        <div className="flex gap-2">
          <button className="flex items-center gap-1.5 px-3 py-1.5 bg-[#111827] border border-[#1f2937] text-xs text-gray-400 rounded-lg hover:text-white transition-colors">
            <RotateCcw size={12} /> Reset to defaults
          </button>
          <button className="flex items-center gap-1.5 px-4 py-1.5 bg-blue-600 hover:bg-blue-700 text-xs text-white rounded-lg font-medium transition-colors">
            <Save size={12} /> Save Changes
          </button>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {/* Col 1: Edge & prob gates */}
        <Card className="p-5">
          <SectionTitle>Edge & Probability Gates</SectionTitle>
          <Field label="YES min edge"     hint="Minimum EV required to log a YES bet"  value={s.yesMinEdge}    onChange={set('yesMinEdge')}    min={0} max={0.5} step={0.01} />
          <Field label="NO min edge"      hint="Minimum EV required to log a NO bet"   value={s.noMinEdge}     onChange={set('noMinEdge')}     min={0} max={0.5} step={0.01} />
          <Field label="YES prob floor"   hint="Skip YES bets below this model prob"   value={s.yesProbFloor}  onChange={set('yesProbFloor')}  min={0.5} max={1.0} step={0.01} />
          <Field label="NO prob ceiling"  hint="Skip NO bets above this model prob"    value={s.noProbCeil}    onChange={set('noProbCeil')}    min={0} max={0.5} step={0.01} />
          <Field label="Confidence mult"  hint="Max Kelly scaling multiplier"          value={s.confMaxMult}   onChange={set('confMaxMult')}   min={1} max={5} step={0.5} unit="×" />
          <Field label="NO odds ceiling"  hint="Skip NO bets when YES ask is above this" value={s.noOddsCeil}  onChange={set('noOddsCeil')}   min={0.3} max={0.9} step={0.05} />
        </Card>

        {/* Col 2: Kelly + liquidity */}
        <Card className="p-5">
          <SectionTitle>Kelly & Position Sizing</SectionTitle>
          <TextInput label="Bankroll"          unit="$"  value={s.bankroll}        onChange={v => set('bankroll')(parseFloat(v))} />
          <Field     label="Kelly fraction"    hint="0 = no bet, 1 = full Kelly"  value={s.kellyFraction}    onChange={set('kellyFraction')}   min={0} max={1} step={0.05} />
          <Field     label="Max position %"    hint="Max % of bankroll per trade"  value={s.maxPositionPct}   onChange={set('maxPositionPct')}  min={0.01} max={0.25} step={0.01} unit="%" />
          <Field     label="Base rate %"       hint="Baseline bet fraction"        value={s.baseRatePct}      onChange={set('baseRatePct')}     min={0.01} max={0.20} step={0.01} unit="%" />
          <TextInput label="n_samples min"     hint="Ignore speakers below this"  value={s.nSamplesMin}      onChange={v => set('nSamplesMin')(parseInt(v))} />

          <div className="mt-5 pt-4 border-t border-[#1f2937]">
            <SectionTitle>Liquidity Gates</SectionTitle>
            <Field     label="Max bid-ask spread" value={s.maxSpread}    onChange={set('maxSpread')}    min={0.02} max={0.25} step={0.01} />
            <TextInput label="Min volume ($)"    value={s.minVolume}    onChange={v => set('minVolume')(parseFloat(v))} unit="$" />
            <TextInput label="Min time to close" value={s.minTimeClose} onChange={v => set('minTimeClose')(parseInt(v))} unit="s" />
          </div>
        </Card>

        {/* Col 3: Pings + misc */}
        <Card className="p-5">
          <SectionTitle>Ping Intervals</SectionTitle>
          <TextInput label="Kalshi API ping"     value={s.apiPing}           onChange={set('apiPing')}           unit="s" />
          <TextInput label="News collection"     value={s.newsPing}          onChange={set('newsPing')}          unit="h" />
          <TextInput label="Data harvest"        value={s.harvestPing}       onChange={set('harvestPing')}       unit="h" />
          <TextInput label="Retrain trigger"     hint="Retrain after N new real rows" value={s.retrainRows} onChange={set('retrainRows')} unit="rows" />
          <TextInput label="Transcript refresh"  value={s.transcriptRefresh} onChange={set('transcriptRefresh')} unit="h" />

          <div className="mt-5 pt-4 border-t border-[#1f2937]">
            <SectionTitle>Misc</SectionTitle>
            <TextInput label="Min transcript chars" value={s.minTranscriptChars} onChange={set('minTranscriptChars')} />
            <Toggle    label="Auto-retrain on threshold" value={s.autoRetrain}  onChange={set('autoRetrain')} />
            <Toggle    label="Live mode"                  value={s.liveMode}    onChange={set('liveMode')} />
          </div>

          <div className="mt-5 pt-4 border-t border-[#1f2937]">
            <SectionTitle>Speaker Blocklist (YES bets)</SectionTitle>
            <div className="space-y-1.5">
              {['Hegseth'].map(sp => (
                <div key={sp} className="flex items-center justify-between bg-[#0c1426] border border-[#1a2640] rounded-lg px-3 py-2 text-xs">
                  <span className="text-gray-300">{sp}</span>
                  <button className="text-red-400 hover:text-red-300 transition-colors">✕</button>
                </div>
              ))}
              <button className="w-full text-xs text-gray-600 hover:text-gray-400 border border-dashed border-[#1a2640] rounded-lg py-2 transition-colors">
                + Add speaker
              </button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}
