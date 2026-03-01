// ── Legacy types (used by Run pages) ─────────────────────────────────────────

export interface ResearchIdea {
  id: string;
  title: string;
  description: string;
  author: string;
  institution?: string;
  tags: string[];
  upvotes: number;
  created_at: string;
  updated_at?: string;
  url?: string;
  citations?: number;
  impact_score?: number;
}

export type RunStatus = 'pending' | 'running' | 'success' | 'failed';

export interface RunRecord {
  run_id: string;
  status: RunStatus;
  keywords: string[];
  n: number;
  created_at: string;
  updated_at: string;
  started_at?: string;
  finished_at?: string;
  duration_seconds?: number;
  error?: string;
  output_path?: string;
  ideas_count: number;
  report?: {
    run_id: string;
    keywords: string[];
    n: number;
    ideas_count: number;
    average_score: number;
    average_novelty: number;
    average_feasibility: number;
    generated_at: string;
    model: string;
  };
  ideas?: Array<Record<string, unknown>>;
}

export interface RunsReport {
  summary: {
    total_runs: number;
    running_runs: number;
    successful_runs: number;
    failed_runs: number;
    success_rate: number;
    average_duration_seconds: number;
    average_ideas_per_run: number;
  };
  keyword_frequency: Record<string, number>;
  score_trend: Array<{
    run_id: string;
    timestamp: string;
    average_score: number;
    ideas_count: number;
  }>;
  comparison: Array<{
    run_id: string;
    average_score: number;
    average_novelty: number;
    average_feasibility: number;
    ideas_count: number;
    keywords: string[];
  }>;
  generated_at: string;
}

// ── Backtest engine types (mirrors src/backtest/models.py) ────────────────────

export type StrategyStatus = 'pending' | 'running' | 'done' | 'failed';

/** Mirrors IdeaPrediction dataclass */
export interface IdeaPrediction {
  rank: number;
  title: string;
  rationale: string;
  key_terms: string[];
  confidence: number;
}

/** Mirrors EvaluationResult dataclass */
export interface EvaluationResult {
  hit_at_k: number;
  recall_at_k: number;
  precision_at_k: number;
  mrr: number;
  novelty: number;
  diversity: number;
  matched_prediction_ranks: number[];
  matched_terms: string[];
}

/** Mirrors BacktestWindowResult dataclass */
export interface WindowResult {
  cutoff_month: string;
  future_end_month: string;
  train_papers: number;
  future_papers: number;
  predictions: IdeaPrediction[];
  evaluation: EvaluationResult;
}

/** Summary averaged over all windows */
export interface BacktestSummary {
  windows: number;
  avg_hit_at_k: number;
  avg_recall_at_k: number;
  avg_precision_at_k: number;
  avg_mrr: number;
  avg_novelty: number;
  avg_diversity: number;
}

export interface BacktestResult {
  summary: BacktestSummary;
  windows: WindowResult[];
}

/** Strategy hyper-parameters (varies by strategy_name) */
export interface StrategyParams {
  recent_months?: number;
  min_keyword_freq?: number;
  model_id?: string;
  prompt_id?: string;
  prompt_version?: string;
  temperature?: number | null;
  [key: string]: unknown;
}

/** BacktestConfig fields + data path */
export interface StrategyConfig {
  top_k: number;
  horizon_months: number;
  min_train_papers: number;
  start_month: string;  // "YYYY-MM"
  end_month: string;    // "YYYY-MM"
  data_dir: string;
}

/** Generation result: predictions at a single cutoff */
export interface GenerationResult {
  cutoff_month: string;
  predictions: IdeaPrediction[];
}

export interface Strategy {
  id: string;
  name: string;
  strategy_name: string;      // "keyword_trend"
  params: StrategyParams;
  config: StrategyConfig;
  created_at: string;
  backtest_status: StrategyStatus;
  generation_status: StrategyStatus;
  backtest_result?: BacktestResult | null;
  generation?: GenerationResult | null;
}
