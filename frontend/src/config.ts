export const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || '';

export const REFRESH_INTERVAL = parseInt(
  process.env.REACT_APP_REFRESH_INTERVAL || '30000',
  10
);

export const DEBUG_MODE = process.env.NODE_ENV === 'development';

export const API_ENDPOINTS = {
  STRATEGIES: '/api/strategies',
  STRATEGY: (id: string) => `/api/strategies/${id}`,
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
