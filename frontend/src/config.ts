/**
 * Frontend config
 */

export const USE_MOCK_DATA = false;

export const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export const REFRESH_INTERVAL = parseInt(process.env.REACT_APP_REFRESH_INTERVAL || '300000', 10);

export const DEBUG_MODE = process.env.NODE_ENV === 'development';

export const API_ENDPOINTS = {
  RESEARCH_IDEAS: '/api/research-ideas',
  VIEWS: '/api/views',
  RUNS_START: '/api/runs/start',
  RUNS_LIST: '/api/runs/list',
  RUNS_DETAIL: '/api/runs',
  RUNS_REPORT: '/api/runs/report',
};

export const logger = {
  log: (...args: any[]) => {
    if (DEBUG_MODE) {
      console.log('[App]', ...args);
    }
  },
  error: (...args: any[]) => {
    console.error('[App Error]', ...args);
  },
  warn: (...args: any[]) => {
    if (DEBUG_MODE) {
      console.warn('[App Warning]', ...args);
    }
  },
};
