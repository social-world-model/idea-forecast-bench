/**
 * Frontend configuration
 */

export const USE_MOCK_DATA = false;

export const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export const REFRESH_INTERVAL = parseInt(
  process.env.REACT_APP_REFRESH_INTERVAL || '30000',
  10
);

export const DEBUG_MODE = process.env.NODE_ENV === 'development';

export const API_ENDPOINTS = {
  // ── Backtest / Strategy API ──────────────────────────────────
  STRATEGIES: '/api/strategies',
  STRATEGY: (id: string) => `/api/strategies/${id}`,
  STRATEGY_BACKTEST: (id: string) => `/api/strategies/${id}/backtest`,
  STRATEGY_GENERATE: (id: string) => `/api/strategies/${id}/generate`,
  STRATEGY_STATUS: (id: string) => `/api/strategies/${id}/status`,
  HEALTH: '/api/health',

  // ── Legacy Run API (kept for RunPage / HistoryPage) ──────────
  GENERATE_IDEAS: '/api/generate-ideas',
  RESEARCH_IDEAS: '/api/research-ideas',
  VIEWS: '/api/views',
  RUNS_LIST: '/api/runs/list',
  RUNS_DETAIL: '/api/runs',
  RUNS_REPORT: '/api/runs/report',
};

export const logger = {
  log: (...args: unknown[]) => {
    if (DEBUG_MODE) console.log('[App]', ...args);
  },
  error: (...args: unknown[]) => console.error('[App Error]', ...args),
  warn: (...args: unknown[]) => {
    if (DEBUG_MODE) console.warn('[App Warning]', ...args);
  },
};
