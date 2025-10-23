import React, { useState, useEffect, useCallback, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import About from './components/About';
import Navigation from './components/Navigation';
import Footnote from './components/Footnote';
import './App.css';
import type { ResearchIdea } from './types';
import { USE_MOCK_DATA, API_BASE_URL, API_ENDPOINTS, REFRESH_INTERVAL, logger } from './config';
import { fetchMockResearchIdeas } from './mockData';

function App() {
  // Add a ref to track if the initial fetch has been triggered
  const initialFetchComplete = useRef(false);

  // Global data - research ideas
  const [researchIdeas, setResearchIdeas] = useState<ResearchIdea[]>([]);
  const [views, setViews] = useState<number>(0);
  const [isLoading, setIsLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  // Background data fetching functions
  const fetchResearchIdeas = useCallback(async () => {
    try {
      logger.log('🔄 Fetching research ideas...');
      
      let ideas: ResearchIdea[];
      
      if (USE_MOCK_DATA) {
        // Use mock data
        logger.log('📦 Using mock data');
        ideas = await fetchMockResearchIdeas();
      } else {
        // Use real API
        const url = `${API_BASE_URL}${API_ENDPOINTS.RESEARCH_IDEAS}`;
        logger.log('🌐 Fetching from API:', url);
        const response = await fetch(url);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        ideas = await response.json();
      }
      
      setResearchIdeas(ideas);
      setLastRefresh(new Date());
      logger.log(`✅ Research ideas updated: ${ideas.length} ideas`);
    } catch (error) {
      logger.error('❌ Research ideas fetch failed:', error);
      // If API fails and not using mock data, fallback to mock data
      if (!USE_MOCK_DATA) {
        logger.warn('⚠️ Falling back to mock data');
        try {
          const ideas = await fetchMockResearchIdeas();
          setResearchIdeas(ideas);
          setLastRefresh(new Date());
        } catch (mockError) {
          logger.error('❌ Mock data fetch also failed:', mockError);
        }
      }
    }
  }, []);

  const fetchViews = useCallback(async () => {
    try {
      if (USE_MOCK_DATA) {
        // Mock data mode: use random view count
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
      // Fallback to mock view count
      setViews(Math.floor(Math.random() * 10000) + 1000);
    }
  }, []);

  // Parallel data fetching function
  const fetchAllDataInParallel = useCallback(async () => {
    logger.log('🔄 Starting data fetch...', USE_MOCK_DATA ? '(MOCK MODE)' : '(API MODE)');
    const startTime = Date.now();

    try {
      // Execute all fetch functions in parallel
      await Promise.all([
        fetchResearchIdeas(),
        fetchViews()
      ]);

      const elapsed = Date.now() - startTime;
      logger.log(`✅ Fetch completed in ${elapsed}ms`);
    } catch (error) {
      logger.error('❌ Fetch failed:', error);
    } finally {
      setIsLoading(false);
    }
  }, [fetchResearchIdeas, fetchViews]);

  // Unified background data management
  useEffect(() => {
    // This effect should only run ONCE during the app's entire lifecycle
    if (initialFetchComplete.current) {
      logger.log('↩️ Initial fetch already completed, skipping effect.');
      return;
    }
    initialFetchComplete.current = true;

    logger.log('🚀 Starting data management...', `Mode: ${USE_MOCK_DATA ? 'MOCK' : 'API'}`);

    // Fetch all data immediately on app start in parallel
    fetchAllDataInParallel();

    // Update research ideas periodically
    const updateInterval = setInterval(async () => {
      logger.log(`🔄 Periodic update (${REFRESH_INTERVAL / 60000} min)...`);
      await fetchResearchIdeas();
    }, REFRESH_INTERVAL);

    logger.log(`⏰ Update interval set: ${REFRESH_INTERVAL / 60000} minutes`);

    // Cleanup all intervals on unmount
    return () => {
      clearInterval(updateInterval);
      logger.log('🛑 All intervals cleared');
    };
  }, [fetchAllDataInParallel, fetchResearchIdeas]);

  return (
    <Router>
      <div className="App">
        <Navigation />
        <div className="main-content">
          <Routes>
            <Route path="/" element={
              <Dashboard
                researchIdeas={researchIdeas}
                lastRefresh={lastRefresh}
                views={views}
                isLoading={isLoading}
              />
            } />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>
        <Footnote />
      </div>
    </Router>
  );
}

export default App;
