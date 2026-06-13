import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Live from './pages/Live'
import Markets from './pages/Markets'
import TradeLog from './pages/TradeLog'
import Analytics from './pages/Analytics'
import Model from './pages/Model'
import System from './pages/System'
import SpeakerProfiles from './pages/SpeakerProfiles'
import NewsTranscripts from './pages/NewsTranscripts'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/live" replace />} />
          <Route path="live" element={<Live />} />
          <Route path="markets" element={<Markets />} />
          <Route path="history" element={<TradeLog />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="model" element={<Model />} />
          <Route path="speakers" element={<SpeakerProfiles />} />
          <Route path="news" element={<NewsTranscripts />} />
          <Route path="system" element={<System />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
