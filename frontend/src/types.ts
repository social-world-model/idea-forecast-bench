// Shared frontend types

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
