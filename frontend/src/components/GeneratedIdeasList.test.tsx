import { flattenGeneratedIdeas } from './GeneratedIdeasList';
import type { Strategy } from '../types';

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
        matched_paper_count: 3,
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
        backtest_result: null,
      },
    ],
    leaderboard_score: null,
    daily_evaluation: null,
    last_daily_run_at: null,
    last_generation_cutoff_month: null,
  };
}

test('flattenGeneratedIdeas reads topic runs', () => {
  const items = flattenGeneratedIdeas([makeStrategy()]);

  expect(items).toHaveLength(1);
  expect(items[0].topicName).toBe('Optimizer');
  expect(items[0].prediction.title).toBe('Adaptive AdamW schedules');
});
