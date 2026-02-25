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
import type { ResearchIdea } from './types';
import { USE_MOCK_DATA, API_BASE_URL, API_ENDPOINTS, REFRESH_INTERVAL, logger } from './config';
import { fetchMockResearchIdeas } from './mockData';

function App() {
  const initialFetchComplete = useRef(false);

  const [researchIdeas, setResearchIdeas] = useState<ResearchIdea[]>([]);
  const [views, setViews] = useState<number>(0);
  const [isLoading, setIsLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const fetchResearchIdeas = useCallback(async () => {
    try {
      logger.log('Fetching research ideas...');
      let ideas: ResearchIdea[];

      if (USE_MOCK_DATA) {
        ideas = await fetchMockResearchIdeas();
      } else {
        const url = `${API_BASE_URL}${API_ENDPOINTS.RESEARCH_IDEAS}`;
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        ideas = await response.json();
      }

      setResearchIdeas(ideas);
      setLastRefresh(new Date());
    } catch (error) {
      logger.error('Research ideas fetch failed:', error);
      if (!USE_MOCK_DATA) {
        const ideas = await fetchMockResearchIdeas();
        setResearchIdeas(ideas);
        setLastRefresh(new Date());
      }
    }
  }, []);

  const fetchViews = useCallback(async () => {
    try {
      if (USE_MOCK_DATA) {
        setViews(Math.floor(Math.random() * 10000) + 1000);
        return;
      }

      const url = `${API_BASE_URL}${API_ENDPOINTS.VIEWS}`;
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        setViews(data.views);
      }

      await fetch(url, { method: 'POST' });

      const updatedResponse = await fetch(url);
      if (updatedResponse.ok) {
        const updatedData = await updatedResponse.json();
        setViews(updatedData.views);
      }
    } catch (error) {
      logger.error('Failed to fetch views:', error);
      setViews(Math.floor(Math.random() * 10000) + 1000);
    }
  }, []);

  const fetchAllDataInParallel = useCallback(async () => {
    try {
      await Promise.all([fetchResearchIdeas(), fetchViews()]);
    } finally {
      setIsLoading(false);
    }
  }, [fetchResearchIdeas, fetchViews]);

  useEffect(() => {
    if (initialFetchComplete.current) {
      return;
    }
    initialFetchComplete.current = true;

    fetchAllDataInParallel();

    const updateInterval = setInterval(async () => {
      await fetchResearchIdeas();
    }, REFRESH_INTERVAL);

    return () => {
      clearInterval(updateInterval);
    };
  }, [fetchAllDataInParallel, fetchResearchIdeas]);

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
                  researchIdeas={researchIdeas}
                  lastRefresh={lastRefresh}
                  views={views}
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
