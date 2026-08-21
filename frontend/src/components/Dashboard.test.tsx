import React from 'react';
import { render, screen } from '@testing-library/react';
import Dashboard from './Dashboard';
import type { Strategy } from '../types';

jest.mock('react-router-dom', () => ({
  useNavigate: () => () => undefined,
}));

function makeStrategy(): Strategy {
  return {
    id: 'strategy-1',
    name: 'Topic Strategy',
    strategy_name: 'topic_trend',
    params: {},
    config: {
      top_k: 5,
      horizon_months: 3,
      min_train_papers: 2,
      start_month: '2024-01',
      end_month: '2024-06',
      data_dir: '',
    },
    created_at: '2026-01-01T00:00:00Z',
    backtest_status: 'done',
    generation_status: 'done',
    backtest_result: null,
    generation: null,
    topic_runs: [
      {
        topic_id: 'optimizer',
        topic_name: 'Optimizer',
        matched_paper_count: 2,
        generation_status: 'done',
        backtest_status: 'done',
        generation: {
          cutoff_date: '2024-06-01',
          cutoff_month: '2024-06',
          predictions: [
            {
              rank: 1,
              title: 'Adaptive AdamW schedules',
              rationale: 'Track optimizer drift.',
              approach: 'Benchmark schedule changes.',
              score: 0.8,
              confidence: 0.8,
              key_terms: ['adamw'],
            },
          ],
        },
        backtest_result: {
          summary: {
            windows: 1,
            avg_hit_at_k: 0.6,
            avg_recall_at_k: 0.6,
            avg_precision_at_k: 0.6,
            avg_mrr: 0.5,
            avg_novelty: 0.4,
            avg_diversity: 0.3,
          },
          windows: [
            {
              cutoff_month: '2024-04',
              cutoff_date: '2024-04-01',
              future_end_month: '2024-07',
              future_end_date: '2024-07-31',
              train_papers: 5,
              future_papers: 2,
              predictions: [
                {
                  rank: 1,
                  title: 'Adaptive AdamW schedules',
                  rationale: 'Track optimizer drift.',
                  approach: 'Benchmark schedule changes.',
                  score: 0.8,
                  confidence: 0.8,
                  key_terms: ['adamw'],
                },
              ],
              evaluation: {
                hit_at_k: 1,
                recall_at_k: 1,
                precision_at_k: 1,
                mrr: 1,
                novelty: 0.5,
                diversity: 0.5,
                matched_prediction_ranks: [1],
                matched_paper_ids: ['p1'],
              },
            },
          ],
        },
      },
    ],
    leaderboard_score: null,
    daily_evaluation: null,
    last_daily_run_at: null,
    last_generation_cutoff_month: null,
  };
}

test('Dashboard renders topic sections from topic runs', () => {
  render(<Dashboard strategies={[makeStrategy()]} lastRefresh={new Date()} />);

  expect(screen.getAllByText('Optimizer').length).toBeGreaterThan(0);
  expect(screen.getByText('papers=2')).toBeInTheDocument();
  expect(screen.getByText(/Train up to/i)).toBeInTheDocument();
});
