// Centralized shared types — aligned with src/backtest/* Python models

// ── Job status ───────────────────────────────────────────────────────────────
export type StrategyStatus = "pending" | "running" | "done" | "failed";

// ── Mirrors src/backtest/models.py ───────────────────────────────────────────

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

/** Summary averaged over all windows (from runner._summarize_windows) */
export interface BacktestSummary {
  windows: number;
  avg_hit_at_k: number;
  avg_recall_at_k: number;
  avg_precision_at_k: number;
  avg_mrr: number;
  avg_novelty: number;
  avg_diversity: number;
}

/** Full backtest result { summary, windows } */
export interface BacktestResult {
  summary: BacktestSummary;
  windows: WindowResult[];
}

// ── Strategy ─────────────────────────────────────────────────────────────────

/** Mirrors BacktestConfig + data_dir */
export interface StrategyConfig {
  top_k: number;
  horizon_months: number;
  min_train_papers: number;
  start_month: string;   // "YYYY-MM"
  end_month: string;     // "YYYY-MM"
  data_dir: string;
}

/** Strategy-specific hyper-parameters (KeywordTrendStrategy) */
export interface StrategyParams {
  recent_months: number;
  min_keyword_freq: number;
}

/** Generation result: predictions at a single cutoff */
export interface GenerationResult {
  cutoff_month: string;
  predictions: IdeaPrediction[];
}

export interface Strategy {
  id: string;
  name: string;
  strategy_name: string;     // "keyword_trend" (IdeaStrategy.name)
  params: StrategyParams;
  config: StrategyConfig;
  created_at: string;
  backtest_status: StrategyStatus;
  generation_status: StrategyStatus;
  backtest_result?: BacktestResult | null;
  generation?: GenerationResult | null;
}
