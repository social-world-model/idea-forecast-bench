/**
 * Frontend configuration
 */

// API base URL
export const API_BASE_URL = "http://localhost:5000";

// Data refresh interval (ms)
export const REFRESH_INTERVAL = parseInt(
  process.env.REACT_APP_REFRESH_INTERVAL || "30000"  // 30s default
);

// Debug logging
export const DEBUG_MODE = process.env.NODE_ENV === "development";

// API endpoints
export const API_ENDPOINTS = {
  STRATEGIES: "/api/strategies",
  STRATEGY: (id: string) => `/api/strategies/${id}`,
  STRATEGY_BACKTEST: (id: string) => `/api/strategies/${id}/backtest`,
  STRATEGY_GENERATE: (id: string) => `/api/strategies/${id}/generate`,
  STRATEGY_STATUS: (id: string) => `/api/strategies/${id}/status`,
  HEALTH: "/api/health",
};

/**
 * Logger utility
 */
export const logger = {
  log: (...args: any[]) => {
    if (DEBUG_MODE) console.log("[App]", ...args);
  },
  error: (...args: any[]) => console.error("[App Error]", ...args),
  warn: (...args: any[]) => {
    if (DEBUG_MODE) console.warn("[App Warning]", ...args);
  },
};
