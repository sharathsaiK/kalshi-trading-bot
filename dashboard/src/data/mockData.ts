export interface Trade {
  id: number
  ticker: string
  word: string
  speaker: string
  side: 'YES' | 'NO'
  prob: number
  odds: number
  ev: number
  contracts: number
  result: 'WIN' | 'LOSS' | 'OPEN'
  pnl: number
  date: string
  eventType: string
}

export interface PnLPoint {
  date: string
  pnl: number
}

export interface LiveMarket {
  ticker: string
  word: string
  speaker: string
  ourProb: number
  mktPrice: number
  ev: number
  call: string
  status: 'bet' | 'skip' | 'watch'
}

export interface ActivityItem {
  id: number
  time: string
  level: 'INFO' | 'WARN' | 'ERROR'
  source: string
  message: string
}

export interface SpeakerProfile {
  speaker: string
  word: string
  hitRateLifetime: number
  hitRateRecent: number
  momentum: number
  nSamples: number
  updatedAt: string
}

// ---- P&L history -------------------------------------------------------
const seed = (n: number) => {
  let s = n
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff }
}
const rng = seed(42)

export const mockPnLHistory: PnLPoint[] = (() => {
  const pts: PnLPoint[] = []
  let cum = 0
  const start = new Date('2026-03-01')
  for (let i = 0; i < 92; i++) {
    const d = new Date(start); d.setDate(start.getDate() + i)
    const daily = i < 5 ? 0 : parseFloat(((rng() * 3.5) - 0.4).toFixed(2))
    cum = parseFloat((cum + daily).toFixed(2))
    pts.push({ date: d.toISOString().split('T')[0], pnl: Math.max(0, cum) })
  }
  return pts
})()

// ---- Trades -----------------------------------------------------------
const speakerWords: Record<string, string[]> = {
  Trump:   ['Economy', 'Jobs', 'America', 'Wall', 'Border', 'China', 'Military', 'Trade'],
  Powell:  ['Rate', 'Inflation', 'Employment', 'Growth', 'Balance'],
  Vance:   ['Border', 'China', 'Family', 'Ohio', 'Manufacturing'],
  Rubio:   ['Cuba', 'Venezuela', 'China', 'NATO'],
  Hegseth: ['Military', 'Budget', 'Veterans'],
}
const speakers = Object.keys(speakerWords)
const eventTypes = ['sotu', 'press_conf', 'fomc', 'speech', 'debate']

export const mockTrades: Trade[] = (() => {
  const trades: Trade[] = []
  const start = new Date('2026-03-01')
  for (let i = 1; i <= 77; i++) {
    const speaker = speakers[Math.floor(rng() * speakers.length)]
    const words = speakerWords[speaker]
    const word = words[Math.floor(rng() * words.length)]
    const side: 'YES' | 'NO' = rng() > 0.43 ? 'YES' : 'NO'
    const prob = side === 'YES' ? 0.72 + rng() * 0.25 : 0.08 + rng() * 0.22
    const odds = side === 'YES' ? prob - 0.18 - rng() * 0.08 : (1 - prob) - 0.12 - rng() * 0.08
    const ev = parseFloat(Math.abs(side === 'YES' ? prob - odds : (1 - prob) - odds).toFixed(3))
    const contracts = Math.max(1, Math.floor(rng() * 12))
    const isOpen = i > 68
    const win = rng() < 0.831
    const result: 'WIN' | 'LOSS' | 'OPEN' = isOpen ? 'OPEN' : win ? 'WIN' : 'LOSS'
    const pnl = isOpen ? 0 : win
      ? parseFloat(((1 - odds) * contracts * 0.1).toFixed(2))
      : parseFloat((-odds * contracts * 0.1).toFixed(2))
    const d = new Date(start); d.setDate(start.getDate() + Math.floor((i / 77) * 87))
    const etype = eventTypes[Math.floor(rng() * eventTypes.length)]
    const month = d.toISOString().slice(2, 7).replace('-', '')
    trades.push({
      id: i, ticker: `KX${speaker.toUpperCase()}${etype.toUpperCase()}-${month}`,
      word, speaker, side,
      prob: parseFloat(prob.toFixed(3)), odds: parseFloat(Math.max(0.05, odds).toFixed(3)), ev, contracts,
      result, pnl, date: d.toISOString().split('T')[0], eventType: etype,
    })
  }
  return trades.reverse()
})()

// ---- Live markets -------------------------------------------------------
export interface MarketRow {
  ticker: string
  word: string
  speaker: string
  ourProb: number
  mktPrice: number
  yesAsk: number
  noAsk: number
  ev: number
  evSide: 'YES' | 'NO'
  contracts: number
  volume: number
  spread: number
  closeTime: string
  status: 'bet' | 'skip' | 'watch'
  skipReason?: string
}

export const mockLiveMarkets: LiveMarket[] = [
  { ticker: 'KXTRUMPSOTU-26APR14', word: 'Economy', speaker: 'Trump', ourProb: 0.84, mktPrice: 0.61, ev: 0.23, call: '>>> YES ×8', status: 'bet' },
  { ticker: 'KXTRUMPSOTU-26APR14', word: 'Jobs',    speaker: 'Trump', ourProb: 0.31, mktPrice: 0.55, ev: 0.14, call: '>>> NO ×4',  status: 'bet' },
  { ticker: 'KXTRUMPSOTU-26APR14', word: 'China',   speaker: 'Trump', ourProb: 0.62, mktPrice: 0.58, ev: 0.04, call: 'watch',       status: 'watch' },
  { ticker: 'KXVANCEINGRAHAM-26APR', word: 'Meme',  speaker: 'Vance', ourProb: 0.18, mktPrice: 0.30, ev: -0.12, call: 'no edge',   status: 'skip' },
  { ticker: 'KXPOWELLFED-26APR10',  word: 'Rate',   speaker: 'Powell', ourProb: 0.79, mktPrice: 0.55, ev: 0.24, call: '>>> YES ×9', status: 'bet' },
]

export const mockMarkets: MarketRow[] = [
  { ticker: 'KXTRUMPSOTU-26APR14', word: 'Tariff',    speaker: 'Donald Trump',  ourProb: 0.88, mktPrice: 0.63, yesAsk: 0.64, noAsk: 0.38, ev: 0.24, evSide: 'YES', contracts: 8,  volume: 4820, spread: 0.03, closeTime: '1h 14m', status: 'bet' },
  { ticker: 'KXTRUMPSOTU-26APR14', word: 'Economy',   speaker: 'Donald Trump',  ourProb: 0.84, mktPrice: 0.61, yesAsk: 0.62, noAsk: 0.40, ev: 0.22, evSide: 'YES', contracts: 7,  volume: 3910, spread: 0.04, closeTime: '1h 14m', status: 'bet' },
  { ticker: 'KXPOWELLFED-26APR10', word: 'Rate',      speaker: 'Jerome Powell', ourProb: 0.79, mktPrice: 0.55, yesAsk: 0.56, noAsk: 0.46, ev: 0.23, evSide: 'YES', contracts: 9,  volume: 6140, spread: 0.02, closeTime: '3h 40m', status: 'bet' },
  { ticker: 'KXPOWELLFED-26APR10', word: 'Inflation', speaker: 'Jerome Powell', ourProb: 0.91, mktPrice: 0.70, yesAsk: 0.71, noAsk: 0.31, ev: 0.20, evSide: 'YES', contracts: 5,  volume: 5200, spread: 0.03, closeTime: '3h 40m', status: 'bet' },
  { ticker: 'KXTRUMPSOTU-26APR14', word: 'Jobs',      speaker: 'Donald Trump',  ourProb: 0.28, mktPrice: 0.55, yesAsk: 0.56, noAsk: 0.46, ev: 0.18, evSide: 'NO',  contracts: 4,  volume: 2880, spread: 0.04, closeTime: '1h 14m', status: 'bet' },
  { ticker: 'KXTRUMPSOTU-26APR14', word: 'China',     speaker: 'Donald Trump',  ourProb: 0.62, mktPrice: 0.58, yesAsk: 0.59, noAsk: 0.43, ev: 0.04, evSide: 'YES', contracts: 0,  volume: 1920, spread: 0.05, closeTime: '1h 14m', status: 'watch' },
  { ticker: 'KXPOWELLFED-26APR10', word: 'Growth',    speaker: 'Jerome Powell', ourProb: 0.55, mktPrice: 0.52, yesAsk: 0.53, noAsk: 0.49, ev: 0.02, evSide: 'YES', contracts: 0,  volume: 980,  spread: 0.06, closeTime: '3h 40m', status: 'watch', skipReason: 'low volume' },
  { ticker: 'KXVANCEINGRAHAM-26APR', word: 'Border',  speaker: 'JD Vance',      ourProb: 0.44, mktPrice: 0.38, yesAsk: 0.39, noAsk: 0.63, ev: 0.03, evSide: 'NO',  contracts: 0,  volume: 540,  spread: 0.08, closeTime: '22m',    status: 'skip', skipReason: 'closing soon' },
  { ticker: 'KXVANCEINGRAHAM-26APR', word: 'Meme',    speaker: 'JD Vance',      ourProb: 0.18, mktPrice: 0.30, yesAsk: 0.31, noAsk: 0.71, ev: -0.12, evSide: 'NO', contracts: 0,  volume: 310,  spread: 0.12, closeTime: '22m',    status: 'skip', skipReason: 'no edge' },
  { ticker: 'KXRUBIOPRESS-26APR01', word: 'China',    speaker: 'Marco Rubio',   ourProb: 0.71, mktPrice: 0.58, yesAsk: 0.59, noAsk: 0.43, ev: 0.12, evSide: 'YES', contracts: 0,  volume: 220,  spread: 0.09, closeTime: '6h 02m', status: 'skip', skipReason: 'wide spread' },
]

// ---- Activity feed -------------------------------------------------------
export const mockActivityFeed: ActivityItem[] = [
  { id: 1,  time: '16:42:03', level: 'INFO', source: 'TRADE',   message: 'YES ×8 logged — KXTRUMPSOTU/Economy (EV +0.23, Kelly ×2.3)' },
  { id: 2,  time: '16:41:57', level: 'INFO', source: 'GATE',    message: 'skip: prob 0.68 < YES floor 0.72 — KXTRUMPSOTU/Jobs' },
  { id: 3,  time: '16:41:51', level: 'INFO', source: 'GATE',    message: 'skip: no edge — KXVANCEINGRAHAM/Meme (ev=-0.12)' },
  { id: 4,  time: '16:41:44', level: 'INFO', source: 'MODEL',   message: 'predict: Trump/Economy → 0.84 (lgb=0.812, cal=0.840)' },
  { id: 5,  time: '16:41:31', level: 'INFO', source: 'API',     message: 'Kalshi ping 0.38s — 5 open markets fetched' },
  { id: 6,  time: '16:40:12', level: 'INFO', source: 'TRADE',   message: 'NO ×4 logged — KXTRUMPSOTU/Jobs (EV +0.14)' },
  { id: 7,  time: '16:38:44', level: 'INFO', source: 'PROFILE', message: 'updated: Trump/sotu — 312 samples, hit_rate=0.74' },
  { id: 8,  time: '16:35:01', level: 'INFO', source: 'NEWS',    message: 'fetched 14 articles across 6 words (4.2s)' },
  { id: 9,  time: '16:33:10', level: 'WARN', source: 'API',     message: 'rate limit hit — backing off 2.0s' },
  { id: 10, time: '16:30:00', level: 'INFO', source: 'HARVEST', message: 'tick — 3 new training rows saved' },
]

export const newFeedPool = [
  { level: 'INFO' as const, source: 'API',     message: 'Kalshi ping 0.41s — markets healthy' },
  { level: 'INFO' as const, source: 'MODEL',   message: 'predict: Powell/Rate → 0.81 (lgb=0.798)' },
  { level: 'INFO' as const, source: 'GATE',    message: 'skip: wide spread (yes=0.14) — KXVANCEINGRAHAM/Ohio' },
  { level: 'WARN' as const, source: 'API',     message: 'response slow 1.8s — retrying' },
  { level: 'INFO' as const, source: 'TRADE',   message: 'YES ×9 logged — KXPOWELLFED/Rate (EV +0.24)' },
  { level: 'INFO' as const, source: 'PROFILE', message: 'updated: Powell/fomc — 88 samples, hit_rate=0.82' },
  { level: 'INFO' as const, source: 'GATE',    message: 'skip: kelly=0 — KXRUBIOPRESS/NATO' },
  { level: 'INFO' as const, source: 'HARVEST', message: 'settled: KXPOWELLFED-26APR10 — 6 rows saved' },
  { level: 'INFO' as const, source: 'NEWS',    message: 'fetched 9 articles for 4 words (3.1s)' },
  { level: 'ERROR' as const, source: 'TRANS',  message: 'fetch failed: KXHEGSETHCONF — timeout (25s)' },
]

// ---- Speaker profiles ---------------------------------------------------
export const mockSpeakerProfiles: SpeakerProfile[] = [
  { speaker: 'Trump',   word: 'Economy',       hitRateLifetime: 0.74, hitRateRecent: 0.81, momentum:  0.07, nSamples: 312, updatedAt: '2m ago' },
  { speaker: 'Trump',   word: 'Jobs',          hitRateLifetime: 0.61, hitRateRecent: 0.55, momentum: -0.06, nSamples: 298, updatedAt: '2m ago' },
  { speaker: 'Trump',   word: 'America',       hitRateLifetime: 0.89, hitRateRecent: 0.91, momentum:  0.02, nSamples: 301, updatedAt: '2m ago' },
  { speaker: 'Trump',   word: 'Wall',          hitRateLifetime: 0.48, hitRateRecent: 0.41, momentum: -0.07, nSamples: 278, updatedAt: '2m ago' },
  { speaker: 'Trump',   word: 'Border',        hitRateLifetime: 0.71, hitRateRecent: 0.74, momentum:  0.03, nSamples: 265, updatedAt: '2m ago' },
  { speaker: 'Trump',   word: 'China',         hitRateLifetime: 0.66, hitRateRecent: 0.69, momentum:  0.03, nSamples: 241, updatedAt: '2m ago' },
  { speaker: 'Powell',  word: 'Rate',          hitRateLifetime: 0.82, hitRateRecent: 0.77, momentum: -0.05, nSamples: 88,  updatedAt: '5d ago' },
  { speaker: 'Powell',  word: 'Inflation',     hitRateLifetime: 0.91, hitRateRecent: 0.94, momentum:  0.03, nSamples: 84,  updatedAt: '5d ago' },
  { speaker: 'Powell',  word: 'Employment',    hitRateLifetime: 0.78, hitRateRecent: 0.80, momentum:  0.02, nSamples: 76,  updatedAt: '5d ago' },
  { speaker: 'Vance',   word: 'Border',        hitRateLifetime: 0.55, hitRateRecent: 0.60, momentum:  0.05, nSamples: 32,  updatedAt: '12d ago' },
  { speaker: 'Rubio',   word: 'China',         hitRateLifetime: 0.68, hitRateRecent: 0.71, momentum:  0.03, nSamples: 28,  updatedAt: '15d ago' },
  { speaker: 'Hegseth', word: 'Military',      hitRateLifetime: 0.77, hitRateRecent: 0.70, momentum: -0.07, nSamples: 14,  updatedAt: '22d ago' },
]

// ---- Speaker stats -------------------------------------------------------
export const mockSpeakerStats = [
  { speaker: 'Trump',   bets: 64, wins: 56, pnl: 31.20, winRate: 87.5 },
  { speaker: 'Powell',  bets: 18, wins: 15, pnl: 10.50, winRate: 83.3 },
  { speaker: 'Vance',   bets:  8, wins:  6, pnl:  3.31, winRate: 75.0 },
  { speaker: 'Rubio',   bets:  4, wins:  3, pnl: -0.02, winRate: 75.0 },
  { speaker: 'Hegseth', bets:  2, wins:  1, pnl: -2.40, winRate: 50.0 },
]

// ---- Win rate over time --------------------------------------------------
export const mockWinRateHistory = [
  { date: 'Mar 1',  wr: 70 }, { date: 'Mar 8',  wr: 73 }, { date: 'Mar 15', wr: 76 },
  { date: 'Mar 22', wr: 79 }, { date: 'Apr 1',  wr: 80 }, { date: 'Apr 8',  wr: 82 },
  { date: 'Apr 15', wr: 81 }, { date: 'Apr 22', wr: 83 }, { date: 'May 1',  wr: 83 },
  { date: 'May 15', wr: 83 }, { date: 'Jun 1',  wr: 83 },
]

// ---- Calibration data ---------------------------------------------------
export const mockCalibration = [
  { prob: 0.10, actual: 0.09, perfect: 0.10 },
  { prob: 0.20, actual: 0.18, perfect: 0.20 },
  { prob: 0.30, actual: 0.29, perfect: 0.30 },
  { prob: 0.40, actual: 0.42, perfect: 0.40 },
  { prob: 0.50, actual: 0.48, perfect: 0.50 },
  { prob: 0.60, actual: 0.61, perfect: 0.60 },
  { prob: 0.70, actual: 0.72, perfect: 0.70 },
  { prob: 0.80, actual: 0.81, perfect: 0.80 },
  { prob: 0.90, actual: 0.88, perfect: 0.90 },
]

// ---- Model history -------------------------------------------------------
export const mockModelHistory = [
  { version: 'v1.4', date: '2026-05-29 14:22', auc: 0.808, brier: 0.1774, acc: 83.1, rows: 1163, active: true },
  { version: 'v1.3', date: '2026-05-04 09:11', auc: 0.791, brier: 0.1882, acc: 81.2, rows: 1041, active: false },
  { version: 'v1.2', date: '2026-04-18 16:45', auc: 0.774, brier: 0.1941, acc: 78.9, rows:  923, active: false },
  { version: 'v1.1', date: '2026-03-31 11:30', auc: 0.762, brier: 0.2010, acc: 77.4, rows:  840, active: false },
]

// ---- Logs ---------------------------------------------------------------
export const mockLogs: ActivityItem[] = [
  { id: 1,  time: '16:42:03', level: 'INFO',  source: 'TRADE',   message: 'YES ×8 logged — KXTRUMPSOTU/Economy (EV +0.23, Kelly ×2.3)' },
  { id: 2,  time: '16:41:57', level: 'INFO',  source: 'GATE',    message: 'skip: prob 0.68 < YES floor 0.72 — KXTRUMPSOTU/Jobs' },
  { id: 3,  time: '16:41:51', level: 'INFO',  source: 'GATE',    message: 'skip: no edge — KXVANCEINGRAHAM/Meme (ev_yes=-0.12, ev_no=-0.02)' },
  { id: 4,  time: '16:41:44', level: 'INFO',  source: 'MODEL',   message: 'predict_proba: Trump/Economy → 0.84 (lgb=0.812, lr=0.831, cal=0.840)' },
  { id: 5,  time: '16:41:31', level: 'INFO',  source: 'API',     message: 'Kalshi ping 0.38s — fetched 5 open markets for KXTRUMPSOTU' },
  { id: 6,  time: '16:40:12', level: 'INFO',  source: 'TRADE',   message: 'NO ×4 logged — KXTRUMPSOTU/Jobs (EV +0.14, Kelly ×1.4)' },
  { id: 7,  time: '16:38:44', level: 'INFO',  source: 'PROFILE', message: 'updated: Trump / sotu — 312 samples, hit_rate_lifetime=0.74' },
  { id: 8,  time: '16:35:01', level: 'INFO',  source: 'NEWS',    message: 'fetched 14 articles across 6 words (4.2s)' },
  { id: 9,  time: '16:33:10', level: 'WARN',  source: 'API',     message: 'Kalshi rate limit hit — backing off 2.0s' },
  { id: 10, time: '16:30:00', level: 'INFO',  source: 'HARVEST', message: 'harvest tick — 3 new training rows saved to training_data' },
  { id: 11, time: '16:22:41', level: 'INFO',  source: 'TRAIN',   message: 'retrain triggered (325 real rows, threshold 325)' },
  { id: 12, time: '16:22:49', level: 'INFO',  source: 'TRAIN',   message: 'seed 1/11 done — AUC 0.812' },
  { id: 13, time: '16:23:11', level: 'INFO',  source: 'TRAIN',   message: 'all seeds done — ensemble AUC 0.808, Brier 0.1774' },
  { id: 14, time: '16:23:12', level: 'INFO',  source: 'TRAIN',   message: 'model saved → kalshi_model.lgb (pending merge)' },
  { id: 15, time: '08:14:02', level: 'ERROR', source: 'TRANS',   message: 'transcript fetch failed: KXVANCEINGRAHAM — timeout (25s)' },
  { id: 16, time: '08:14:02', level: 'WARN',  source: 'TRANS',   message: 'skipping event KXVANCEINGRAHAM — no usable transcript' },
  { id: 17, time: '08:01:00', level: 'INFO',  source: 'BOT',     message: 'bot started — paper mode — bankroll $1,000.00' },
]

// ---- Upcoming events ----------------------------------------------------
export interface UpcomingEvent {
  ticker: string
  speaker: string
  type: string
  date: string
  daysOut: number
  markets: number
}

export const mockUpcomingEvents: UpcomingEvent[] = [
  { ticker: 'KXTRUMPPRESS-26JUN08', speaker: 'Trump',  type: 'press_conf', date: '2026-06-08', daysOut: 7,  markets: 8 },
  { ticker: 'KXPOWELLFED-26JUN11',  speaker: 'Powell', type: 'fomc',       date: '2026-06-11', daysOut: 10, markets: 5 },
  { ticker: 'KXRUBIOSTATE-26JUN15', speaker: 'Rubio',  type: 'speech',     date: '2026-06-15', daysOut: 14, markets: 4 },
  { ticker: 'KXVANCECNN-26JUN20',   speaker: 'Vance',  type: 'interview',  date: '2026-06-20', daysOut: 19, markets: 3 },
]

// ---- Transcripts (mock) -------------------------------------------------
export const mockTranscripts = [
  { id: 142, ticker: 'KXTRUMPSOTU-26APR14',      speaker: 'Donald Trump',     date: '2026-04-14', chars: 42318, source: 'YouTube', preview: 'My fellow Americans, tonight I stand before you to report that our nation has never been stronger...' },
  { id: 141, ticker: 'KXPOWELLFED-26APR10',      speaker: 'Jerome Powell',    date: '2026-04-10', chars: 28441, source: 'Fed.gov', preview: 'The Committee decided to maintain the target range for the federal funds rate at 4.25 to 4.5 percent...' },
  { id: 140, ticker: 'KXVANCEINGRAHAM-26APR05',  speaker: 'JD Vance',         date: '2026-04-05', chars: 18220, source: 'YouTube', preview: 'Thank you Laura. The situation at the border has improved dramatically since we took office...' },
  { id: 139, ticker: 'KXRUBIOPRESS-26APR01',     speaker: 'Marco Rubio',      date: '2026-04-01', chars: 14882, source: 'State.gov', preview: 'The United States will not tolerate further destabilization of the region by hostile actors...' },
  { id: 138, ticker: 'KXTRUMPSOTU-26MAR28',      speaker: 'Donald Trump',     date: '2026-03-28', chars: 38901, source: 'YouTube', preview: 'We have built the strongest economy in the history of the world. No country comes close...' },
  { id: 137, ticker: 'KXPOWELLFED-26MAR19',      speaker: 'Jerome Powell',    date: '2026-03-19', chars: 26114, source: 'Fed.gov', preview: 'Inflation has continued to move toward our 2 percent goal, though progress has been uneven...' },
  { id: 136, ticker: 'KXWARRENSENATE-26MAR10',   speaker: 'Elizabeth Warren',  date: '2026-03-10', chars: 9802,  source: 'C-SPAN',  preview: 'These banks are too big to fail and too big to jail and the American people deserve better...' },
  { id: 135, ticker: 'KXHEGSETHDOD-26MAR05',     speaker: 'Pete Hegseth',     date: '2026-03-05', chars: 11340, source: 'YouTube', preview: 'Our military is being rebuilt from the ground up. The era of woke Pentagon policy is over...' },
]

// ---- News articles (mock) ------------------------------------------------
export const mockNews = [
  { id: 1, speaker: 'Donald Trump',    word: 'Tariff',    title: 'Trump threatens 50% tariffs on EU imports starting June', source: 'Reuters',          date: '2026-06-05', relevance: 0.94, url: '#' },
  { id: 2, speaker: 'Donald Trump',    word: 'Economy',   title: 'US economy adds 180k jobs in May, beating expectations',  source: 'WSJ',              date: '2026-06-04', relevance: 0.81, url: '#' },
  { id: 3, speaker: 'Donald Trump',    word: 'China',     title: 'US-China trade talks stall over semiconductor controls',  source: 'FT',               date: '2026-06-03', relevance: 0.88, url: '#' },
  { id: 4, speaker: 'Donald Trump',    word: 'Border',    title: 'Border crossings hit 4-year low under new enforcement',   source: 'AP',               date: '2026-06-02', relevance: 0.76, url: '#' },
  { id: 5, speaker: 'Jerome Powell',   word: 'Rate',      title: 'Fed holds rates steady, signals one cut possible in Q3',  source: 'Bloomberg',        date: '2026-06-05', relevance: 0.97, url: '#' },
  { id: 6, speaker: 'Jerome Powell',   word: 'Inflation', title: 'Core PCE inflation ticks up to 2.4% in April',           source: 'WSJ',              date: '2026-06-03', relevance: 0.91, url: '#' },
  { id: 7, speaker: 'Jerome Powell',   word: 'Inflation', title: 'Fed officials split on timeline for rate cuts this year', source: 'Reuters',          date: '2026-06-01', relevance: 0.85, url: '#' },
  { id: 8, speaker: 'Elizabeth Warren', word: 'Bank',     title: 'Warren calls for breaking up JPMorgan in Senate hearing', source: 'The Guardian',    date: '2026-06-04', relevance: 0.89, url: '#' },
  { id: 9, speaker: 'Elizabeth Warren', word: 'Climate',  title: 'Warren introduces bill tying Fed mandate to climate risk', source: 'Politico',       date: '2026-06-02', relevance: 0.72, url: '#' },
  { id: 10, speaker: 'Marco Rubio',    word: 'China',     title: 'Rubio pushes allies to ban Huawei from 5G infrastructure', source: 'FT',             date: '2026-06-03', relevance: 0.86, url: '#' },
  { id: 11, speaker: 'JD Vance',       word: 'Border',    title: 'Vance touts record low border numbers in Fox interview',   source: 'Fox News',       date: '2026-06-04', relevance: 0.79, url: '#' },
  { id: 12, speaker: 'Pete Hegseth',   word: 'Military',  title: 'Pentagon budget proposal raises defense spending 8%',      source: 'Defense One',    date: '2026-06-01', relevance: 0.83, url: '#' },
]
