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

  // ── Fetch all strategies ────────────────────────────────────────────────────
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

  // ── Create strategy ─────────────────────────────────────────────────────────
  const createStrategy = useCallback(async (data: Partial<Strategy>): Promise<Strategy | null> => {
    try {
      const res = await fetch(`${API_BASE_URL}${API_ENDPOINTS.STRATEGIES}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const created: Strategy = await res.json();
      setStrategies((prev) => [created, ...prev]);
      return created;
    } catch (err) {
      logger.error('createStrategy failed:', err);
      return null;
    }
  }, []);

  // ── Delete strategy ─────────────────────────────────────────────────────────
  const deleteStrategy = useCallback(async (id: string) => {
    try {
      await fetch(`${API_BASE_URL}${API_ENDPOINTS.STRATEGY(id)}`, {
        method: 'DELETE',
      });
      setStrategies((prev) => prev.filter((s) => s.id !== id));
    } catch (err) {
      logger.error('deleteStrategy failed:', err);
    }
  }, []);

  // ── Poll until a job finishes ───────────────────────────────────────────────
  const pollStatus = useCallback((id: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(
          `${API_BASE_URL}${API_ENDPOINTS.STRATEGY_STATUS(id)}`
        );
        if (!res.ok) return;
        const st = await res.json();
        const finished =
          st.backtest_status !== 'running' && st.generation_status !== 'running';

        if (finished) {
          clearInterval(interval);
          // Re-fetch the full strategy record with results
          const sRes = await fetch(`${API_BASE_URL}${API_ENDPOINTS.STRATEGY(id)}`);
          if (sRes.ok) {
            const updated: Strategy = await sRes.json();
            setStrategies((prev) => prev.map((s) => (s.id === id ? updated : s)));
          }
        } else {
          setStrategies((prev) =>
            prev.map((s) =>
              s.id === id
                ? {
                    ...s,
                    backtest_status: st.backtest_status,
                    generation_status: st.generation_status,
                  }
                : s
            )
          );
        }
      } catch (_) {
        // ignore transient poll errors
      }
    }, 2000);
  }, []);

  // ── Trigger backtest ────────────────────────────────────────────────────────
  const runBacktest = useCallback(
    async (id: string) => {
      await fetch(`${API_BASE_URL}${API_ENDPOINTS.STRATEGY_BACKTEST(id)}`, {
        method: 'POST',
      });
      setStrategies((prev) =>
        prev.map((s) => (s.id === id ? { ...s, backtest_status: 'running' } : s))
      );
      pollStatus(id);
    },
    [pollStatus]
  );

  // ── Trigger generation ──────────────────────────────────────────────────────
  const runGeneration = useCallback(
    async (id: string, cutoffMonth?: string) => {
      await fetch(`${API_BASE_URL}${API_ENDPOINTS.STRATEGY_GENERATE(id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cutoff_month: cutoffMonth }),
      });
      setStrategies((prev) =>
        prev.map((s) =>
          s.id === id ? { ...s, generation_status: 'running' } : s
        )
      );
      pollStatus(id);
    },
    [pollStatus]
  );

  // ── Initial load + periodic refresh ────────────────────────────────────────
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
                  onRunBacktest={runBacktest}
                  onRunGeneration={runGeneration}
                  onCreateStrategy={createStrategy}
                  onDeleteStrategy={deleteStrategy}
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
