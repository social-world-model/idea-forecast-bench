/**
 * 前端配置文件
 */

// 是否使用模拟数据（开发环境可以设置为true来测试前端）
export const USE_MOCK_DATA = false;

// API基础URL
export const API_BASE_URL = 'http://localhost:5000';

// 数据刷新间隔（毫秒）
export const REFRESH_INTERVAL = parseInt(process.env.REACT_APP_REFRESH_INTERVAL || '300000'); // 默认5分钟

// 是否启用调试日志
export const DEBUG_MODE = process.env.NODE_ENV === 'development';

// API端点
export const API_ENDPOINTS = {
  RESEARCH_IDEAS: '/api/research-ideas',
  VIEWS: '/api/views',
};

/**
 * 日志工具
 */
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

