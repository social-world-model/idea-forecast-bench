import React, { useState, useEffect, useCallback, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import About from './components/About';
import Navigation from './components/Navigation';
import Footnote from './components/Footnote';
import GeneratedIdeasList from './components/GeneratedIdeasList';
import RunPage from './pages/RunPage';
import HistoryPage from './pages/HistoryPage';
import RunDetailPage from './pages/RunDetailPage';
import './App.css';
import './pages/runs.css';
import type { Strategy } from './types';
import { API_BASE_URL, API_ENDPOINTS, REFRESH_INTERVAL, logger } from './config';

function App() {
  const initialFetchDone = useRef(false);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const fetchStrategies = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}${API_ENDPOINTS.STRATEGIES}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: Strategy[] = await res.json();
      setStrategies(data);
      setLastRefresh(new Date());
      logger.log(`Strategies loaded: ${data.length}`);
    } catch (err) {
      logger.error('fetchStrategies failed:', err);
    }
  }, []);

  useEffect(() => {
    if (initialFetchDone.current) return;
    initialFetchDone.current = true;
    fetchStrategies().finally(() => setIsLoading(false));
    const timer = setInterval(fetchStrategies, REFRESH_INTERVAL);
    return () => clearInterval(timer);
  }, [fetchStrategies]);

  return (
    <Router>
      <div className="App">
        <Navigation />
        <div className="main-content">
          <Routes>
            <Route
              path="/"
              element={
                <Dashboard
                  strategies={strategies}
                  lastRefresh={lastRefresh}
                  isLoading={isLoading}
                />
              }
            />
            <Route path="/about" element={<About />} />
            <Route path="/generated-ideas" element={<GeneratedIdeasList />} />
            <Route path="/runs" element={<RunPage />} />
            <Route path="/runs/history" element={<HistoryPage />} />
            <Route path="/runs/:runId" element={<RunDetailPage />} />
          </Routes>
        </div>
        <Footnote />
      </div>
    </Router>
  );
}

export default App;
