// ── Backtest engine types (mirrors idea_forecast_bench/models.py) ─────────────────

export type StrategyStatus = 'pending' | 'running' | 'done' | 'failed';

/** Mirrors IdeaPrediction dataclass */
export interface IdeaPrediction {
  rank: number;
  title: string;
  rationale: string;
  approach: string;
  score: number;
  confidence?: number | null;
  key_terms?: string[];
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
  matched_paper_ids: string[];
}

/** Mirrors BacktestWindowResult dataclass */
export interface WindowResult {
  cutoff_month: string;
  cutoff_date: string;
  future_end_month: string;
  future_end_date: string;
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
  model_name?: string;
  predictor_config?: string;
  similarity_config?: string;
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
  cutoff_date: string;
  cutoff_month: string;
  predictions: IdeaPrediction[];
}

export interface TopicRun {
  topic_id: string;
  topic_name: string;
  matched_paper_count: number;
  generation_status: StrategyStatus;
  backtest_status: StrategyStatus;
  generation?: GenerationResult | null;
  backtest_result?: BacktestResult | null;
  generation_error?: string | null;
  backtest_error?: string | null;
}

export interface DailyEvaluation {
  evaluated_at: string;
  prediction_cutoff_date: string;
  prediction_cutoff_month: string;
  new_papers_count: number;
  prediction_count: number;
  hit_at_k: number;
  recall_at_k: number;
  precision_at_k: number;
  mrr: number;
  novelty: number;
  diversity: number;
  matched_prediction_ranks: number[];
  matched_paper_ids: string[];
}

export interface Strategy {
  id: string;
  name: string;
  strategy_name: string;      // "topic_trend"
  params: StrategyParams;
  config: StrategyConfig;
  created_at: string;
  backtest_status: StrategyStatus;
  generation_status: StrategyStatus;
  backtest_result?: BacktestResult | null;
  generation?: GenerationResult | null;
  topic_runs: TopicRun[];
  leaderboard_score?: number | null;
  daily_evaluation?: DailyEvaluation | null;
  last_daily_run_at?: string | null;
  last_generation_cutoff_month?: string | null;
}
