import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Dashboard.css';
import type {
  Strategy,
  TopicRun,
  IdeaPrediction,
  BacktestSummary,
  WindowResult,
} from '../types';

export type DashboardProps = {
  strategies: Strategy[];
  lastRefresh?: Date;
  isLoading?: boolean;
};

function relativeTime(d?: Date): string {
  if (!d) return '';
  const mins = Math.floor((Date.now() - d.getTime()) / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  return hrs < 24 ? `${hrs}h ago` : `${Math.floor(hrs / 24)}d ago`;
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function fmtIso(raw?: string | null): string {
  if (!raw) return '-';
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) return '-';
  return parsed.toLocaleString();
}

function aggregateTopicSummary(topicRuns: TopicRun[]): BacktestSummary | null {
  const summaries = topicRuns
    .map((topicRun) => topicRun.backtest_result?.summary ?? null)
    .filter((summary): summary is BacktestSummary => Boolean(summary) && summary.windows > 0);

  if (summaries.length === 0) {
    return null;
  }

  const totalWindows = summaries.reduce((sum, summary) => sum + summary.windows, 0);
  const weightedAverage = (
    selector: (summary: BacktestSummary) => number,
  ): number => (
    totalWindows > 0
      ? summaries.reduce((sum, summary) => sum + (selector(summary) * summary.windows), 0) / totalWindows
      : 0
  );

  return {
    windows: totalWindows,
    avg_hit_at_k: Number(weightedAverage((summary) => summary.avg_hit_at_k).toFixed(4)),
    avg_recall_at_k: Number(weightedAverage((summary) => summary.avg_recall_at_k).toFixed(4)),
    avg_precision_at_k: Number(weightedAverage((summary) => summary.avg_precision_at_k).toFixed(4)),
    avg_mrr: Number(weightedAverage((summary) => summary.avg_mrr).toFixed(4)),
    avg_novelty: Number(weightedAverage((summary) => summary.avg_novelty).toFixed(4)),
    avg_diversity: Number(weightedAverage((summary) => summary.avg_diversity).toFixed(4)),
  };
}

function strategyBacktestSummary(strategy: Strategy): BacktestSummary | null {
  return aggregateTopicSummary(strategy.topic_runs ?? []) ?? strategy.backtest_result?.summary ?? null;
}

function activeTopicRuns(strategy: Strategy): TopicRun[] {
  return (strategy.topic_runs ?? []).filter((topicRun) => (
    topicRun.matched_paper_count > 0
    || topicRun.generation
    || topicRun.backtest_result
    || topicRun.generation_status !== 'pending'
    || topicRun.backtest_status !== 'pending'
  ));
}

const MetricBar: React.FC<{ value: number; color?: string }> = ({
  value,
  color = '#9c9ef8',
}) => (
  <div className="metric-bar-wrap">
    <div
      className="metric-bar-fill"
      style={{ width: `${Math.min(value * 100, 100)}%`, background: color }}
    />
  </div>
);

const StrategyRow: React.FC<{
  strategy: Strategy;
  rank: number;
  isSelected: boolean;
  onSelect: () => void;
}> = ({ strategy, rank, isSelected, onSelect }) => {
  const summary = strategyBacktestSummary(strategy);
  const dailyEval = strategy.daily_evaluation ?? null;
  const topicCount = activeTopicRuns(strategy).length;
  const leaderboardScore =
    strategy.leaderboard_score ??
    (summary ? summary.avg_hit_at_k : null);
  const displayHit = dailyEval ? dailyEval.hit_at_k : (summary ? summary.avg_hit_at_k : null);
  const displayMrr = dailyEval ? dailyEval.mrr : (summary ? summary.avg_mrr : null);

  const recentMonths =
    typeof strategy.params?.recent_months === 'number'
      ? strategy.params.recent_months
      : null;
  const minFreq =
    typeof strategy.params?.min_keyword_freq === 'number'
      ? strategy.params.min_keyword_freq
      : null;
  const modelName =
    typeof strategy.params?.model_name === 'string'
      ? strategy.params.model_name
      : null;
  const predictorConfig =
    typeof strategy.params?.predictor_config === 'string'
      ? strategy.params.predictor_config
      : null;
  const similarityConfig =
    typeof strategy.params?.similarity_config === 'string'
      ? strategy.params.similarity_config
      : null;
  const temperature =
    typeof strategy.params?.temperature === 'number'
      ? strategy.params.temperature
      : null;

  return (
    <tr className={`strategy-row ${isSelected ? 'selected' : ''}`} onClick={onSelect}>
      <td className="col-rank">
        <span className={`rank-badge ${rank <= 3 ? `top-${rank}` : ''}`}>
          #{rank}
        </span>
      </td>
      <td className="col-name">
        <div className="strategy-name">{strategy.name}</div>
        <div className="strategy-meta">
          <span className="meta-chip">{strategy.strategy_name}</span>
          {recentMonths !== null && (
            <span className="meta-chip">recent={recentMonths}m</span>
          )}
          {minFreq !== null && (
            <span className="meta-chip">min_freq={minFreq}</span>
          )}
          {modelName && <span className="meta-chip">model={modelName}</span>}
          {predictorConfig && <span className="meta-chip">predictor={predictorConfig}</span>}
          {similarityConfig && <span className="meta-chip">similarity={similarityConfig}</span>}
          {temperature !== null && (
            <span className="meta-chip">temp={temperature}</span>
          )}
          {topicCount > 0 && (
            <span className="meta-chip topic-chip">{topicCount} topics</span>
          )}
        </div>
      </td>
      <td className="col-window">
        <div className="window-val">
          {strategy.config.start_month} → {strategy.config.end_month}
        </div>
        <div className="window-label">
          horizon {strategy.config.horizon_months}m · top-{strategy.config.top_k}
        </div>
      </td>
      <td className="col-metric">
        {leaderboardScore !== null ? (
          <>
            <div className="metric-value accent">{pct(leaderboardScore)}</div>
            <MetricBar value={leaderboardScore} color="#9c9ef8" />
          </>
        ) : (
          <span className="metric-na">—</span>
        )}
      </td>
      <td className="col-metric">
        {displayHit !== null ? (
          <>
            <div className="metric-value">{pct(displayHit)}</div>
            <MetricBar value={displayHit} color="#6ee7b7" />
          </>
        ) : (
          <span className="metric-na">—</span>
        )}
      </td>
      <td className="col-metric">
        {displayMrr !== null ? (
          <>
            <div className="metric-value">{displayMrr.toFixed(3)}</div>
            <MetricBar value={displayMrr} color="#fbbf24" />
          </>
        ) : (
          <span className="metric-na">—</span>
        )}
      </td>
      <td className="col-ideas">
        <span className="ideas-count">{summary ? summary.windows : '—'}</span>
      </td>
      <td className="col-metric">
        <div className="strategy-meta">
          <span className="meta-chip">
            source: {dailyEval ? 'daily' : topicCount > 0 ? 'topic_runs' : summary ? 'legacy' : 'pending'}
          </span>
          <span className="meta-chip">daily_at: {fmtIso(strategy.last_daily_run_at)}</span>
          <span className="meta-chip">bt: {strategy.backtest_status}</span>
          <span className="meta-chip">gen: {strategy.generation_status}</span>
        </div>
      </td>
    </tr>
  );
};

const PredictionCard: React.FC<{ pred: IdeaPrediction; matchedRanks?: number[] }> = ({
  pred,
  matchedRanks = [],
}) => {
  const [expanded, setExpanded] = useState(false);
  const isMatched = matchedRanks.includes(pred.rank);

  return (
    <div
      className={`idea-card ${expanded ? 'expanded' : ''} ${isMatched ? 'matched' : ''}`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="idea-rank">
        <span className={`rank-badge ${pred.rank <= 3 ? `top-${pred.rank}` : ''}`}>
          #{pred.rank}
        </span>
      </div>
      <div className="idea-body">
        <div className="idea-header">
          <h3 className="idea-title">{pred.title}</h3>
          <div className="idea-scores">
            <span className="score-badge confidence" title="Score">
              {(pred.score * 100).toFixed(0)}%
            </span>
            <span className="score-badge confidence" title="Confidence">
              {(((pred.confidence ?? pred.score) || 0) * 100).toFixed(0)}%
            </span>
            {isMatched && (
              <span className="score-badge matched-badge" title="Matched future paper">
                Hit
              </span>
            )}
          </div>
        </div>

        {pred.key_terms && pred.key_terms.length > 0 && (
          <div className="idea-tags-row">
            {pred.key_terms.map((t, i) => (
              <span key={i} className="tag">
                {t}
              </span>
            ))}
          </div>
        )}

        {expanded && (
          <div className="idea-expanded">
            <p className="idea-desc">{pred.rationale}</p>
            <p className="idea-desc">{pred.approach}</p>
          </div>
        )}
      </div>
    </div>
  );
};

const WindowDetail: React.FC<{ window: WindowResult; index: number }> = ({
  window: w,
  index,
}) => {
  const [open, setOpen] = useState(index === 0);
  const e = w.evaluation;

  return (
    <div className="window-card">
      <div className="window-header" onClick={() => setOpen(!open)}>
        <div className="window-title">
          <span className="window-cutoff">
            Train up to <strong>{w.cutoff_month}</strong>
          </span>
          <span className="window-future">
            → Eval {w.cutoff_month} - {w.future_end_month}
          </span>
          <span className="window-counts">
            {w.train_papers} train / {w.future_papers} future
          </span>
        </div>
        <div className="window-eval-chips">
          <span className="eval-chip hit">Hit {pct(e.hit_at_k)}</span>
          <span className="eval-chip recall">Recall {pct(e.recall_at_k)}</span>
          <span className="eval-chip mrr">MRR {e.mrr.toFixed(3)}</span>
        </div>
        <span className="window-toggle">{open ? '▲' : '▼'}</span>
      </div>

      {open && (
        <div className="window-body">
          {e.matched_paper_ids.length > 0 && (
            <div className="matched-terms">
              <span className="matched-terms-label">Matched papers: </span>
              {e.matched_paper_ids.map((t, i) => (
                <span key={i} className="tag match-tag">
                  {t}
                </span>
              ))}
            </div>
          )}
          <div className="predictions-grid">
            {w.predictions.map((pred) => (
              <PredictionCard
                key={pred.rank}
                pred={pred}
                matchedRanks={e.matched_prediction_ranks}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const TopicBacktestSection: React.FC<{ topicRun: TopicRun }> = ({ topicRun }) => {
  const summary = topicRun.backtest_result?.summary ?? null;
  const windows = topicRun.backtest_result?.windows ?? [];

  return (
    <div className="topic-section">
      <div className="topic-section-header">
        <div>
          <h3 className="topic-section-title">{topicRun.topic_name}</h3>
          <div className="topic-section-meta">
            <span className="meta-chip">papers={topicRun.matched_paper_count}</span>
            <span className="meta-chip">status={topicRun.backtest_status}</span>
            {summary && <span className="meta-chip">windows={summary.windows}</span>}
          </div>
        </div>
      </div>

      {!topicRun.backtest_result ? (
        <div className="topic-empty">
          {topicRun.backtest_status === 'running'
            ? 'Backtest in progress for this topic.'
            : topicRun.matched_paper_count === 0
            ? 'No matched papers for this topic.'
            : topicRun.backtest_status === 'failed'
            ? topicRun.backtest_error || 'Backtest failed for this topic.'
            : 'No backtest results yet for this topic.'}
        </div>
      ) : windows.length === 0 ? (
        <div className="topic-empty">No valid backtest windows for this topic.</div>
      ) : (
        <div className="windows-list">
          {windows.map((window, index) => (
            <WindowDetail key={`${topicRun.topic_id}-${window.cutoff_date}`} window={window} index={index} />
          ))}
        </div>
      )}
    </div>
  );
};

const TopicGenerationSection: React.FC<{ topicRun: TopicRun }> = ({ topicRun }) => (
  <div className="topic-section">
    <div className="topic-section-header">
      <div>
        <h3 className="topic-section-title">{topicRun.topic_name}</h3>
        <div className="topic-section-meta">
          <span className="meta-chip">papers={topicRun.matched_paper_count}</span>
          <span className="meta-chip">status={topicRun.generation_status}</span>
          {topicRun.generation && (
            <span className="meta-chip">cutoff={topicRun.generation.cutoff_month}</span>
          )}
        </div>
      </div>
    </div>

    {!topicRun.generation ? (
      <div className="topic-empty">
        {topicRun.generation_status === 'running'
          ? 'Generation in progress for this topic.'
          : topicRun.matched_paper_count === 0
          ? 'No matched papers for this topic.'
          : topicRun.generation_status === 'failed'
          ? topicRun.generation_error || 'Generation failed for this topic.'
          : 'No generated ideas yet for this topic.'}
      </div>
    ) : (
      <div className="predictions-grid">
        {topicRun.generation.predictions.map((pred) => (
          <PredictionCard key={`${topicRun.topic_id}-${pred.rank}`} pred={pred} />
        ))}
      </div>
    )}
  </div>
);

const Dashboard: React.FC<DashboardProps> = ({
  strategies,
  lastRefresh,
  isLoading = false,
}) => {
  const navigate = useNavigate();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detailTab, setDetailTab] = useState<'windows' | 'generation'>('windows');

  const effectiveId = selectedId ?? (strategies[0]?.id ?? null);
  const selected = strategies.find((strategy) => strategy.id === effectiveId);
  const selectedSummary = selected ? strategyBacktestSummary(selected) : null;
  const selectedTopicRuns = selected ? activeTopicRuns(selected) : [];

  if (isLoading) {
    return (
      <div className="dashboard-container">
        <div className="loading-state">
          <div className="loading-spinner" />
          <p>Loading strategies...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <div className="title-container">
          <h1 className="dashboard-title">IdeaForecastBench</h1>
        </div>
        <p className="dashboard-subtitle">
          Benchmark for research idea prediction strategies - backtest on historical
          papers, generate forward-looking ideas, and compare strategy performance.
          {' '}
          <button className="about-link" onClick={() => navigate('/about')}>
            Learn more →
          </button>
        </p>
        <div className="last-updated">Last updated: {relativeTime(lastRefresh)}</div>
      </div>

      <section className="section">
        <div className="section-header">
          <h2 className="section-title">Strategy Leaderboard</h2>
        </div>

        {strategies.length === 0 ? (
          <div className="empty-state">No strategies available.</div>
        ) : (
          <div className="table-wrap">
            <table className="strategy-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Strategy</th>
                  <th>Window</th>
                  <th>Score</th>
                  <th>Hit@K</th>
                  <th>MRR</th>
                  <th>Windows</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {strategies.map((strategy, index) => (
                  <StrategyRow
                    key={strategy.id}
                    strategy={strategy}
                    rank={index + 1}
                    isSelected={strategy.id === effectiveId}
                    onSelect={() => setSelectedId(strategy.id)}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {selected && (
        <section className="section">
          <div className="section-header">
            <h2 className="section-title">
              Detail
              <span className="section-subtitle">- {selected.name}</span>
            </h2>
            <div className="tab-bar">
              <button
                className={`tab-btn ${detailTab === 'windows' ? 'active' : ''}`}
                onClick={() => setDetailTab('windows')}
              >
                Backtest Windows
              </button>
              <button
                className={`tab-btn ${detailTab === 'generation' ? 'active' : ''}`}
                onClick={() => setDetailTab('generation')}
              >
                Generation
              </button>
            </div>
          </div>

          {selectedSummary && (
            <div className="metrics-summary">
              {[
                { label: 'Hit@K', value: pct(selectedSummary.avg_hit_at_k), color: '#9c9ef8' },
                { label: 'Recall@K', value: pct(selectedSummary.avg_recall_at_k), color: '#6ee7b7' },
                { label: 'Precision@K', value: pct(selectedSummary.avg_precision_at_k), color: '#fbbf24' },
                { label: 'MRR', value: selectedSummary.avg_mrr.toFixed(3), color: '#f9a8d4' },
                { label: 'Novelty', value: pct(selectedSummary.avg_novelty), color: '#a5b4fc' },
                { label: 'Diversity', value: pct(selectedSummary.avg_diversity), color: '#86efac' },
                { label: 'Windows', value: String(selectedSummary.windows), color: 'rgba(255,255,255,0.6)' },
              ].map((item) => (
                <div key={item.label} className="metric-card">
                  <div className="metric-card-value" style={{ color: item.color }}>
                    {item.value}
                  </div>
                  <div className="metric-card-label">{item.label}</div>
                </div>
              ))}
            </div>
          )}

          {selected.daily_evaluation && (
            <div className="metrics-summary">
              {[
                { label: 'Daily Hit@K', value: pct(selected.daily_evaluation.hit_at_k), color: '#9c9ef8' },
                { label: 'Daily Recall@K', value: pct(selected.daily_evaluation.recall_at_k), color: '#6ee7b7' },
                { label: 'Daily Precision@K', value: pct(selected.daily_evaluation.precision_at_k), color: '#fbbf24' },
                { label: 'Daily MRR', value: selected.daily_evaluation.mrr.toFixed(3), color: '#f9a8d4' },
                { label: 'New Papers', value: String(selected.daily_evaluation.new_papers_count), color: '#a5b4fc' },
                { label: 'Eval At', value: fmtIso(selected.daily_evaluation.evaluated_at), color: '#86efac' },
              ].map((item) => (
                <div key={item.label} className="metric-card">
                  <div className="metric-card-value" style={{ color: item.color }}>
                    {item.value}
                  </div>
                  <div className="metric-card-label">{item.label}</div>
                </div>
              ))}
            </div>
          )}

          {detailTab === 'windows' && (
            <>
              {selectedTopicRuns.length > 0 ? (
                <div className="topic-sections">
                  {selectedTopicRuns.map((topicRun) => (
                    <TopicBacktestSection key={topicRun.topic_id} topicRun={topicRun} />
                  ))}
                </div>
              ) : !selected.backtest_result ? (
                <div className="empty-state">
                  {selected.backtest_status === 'running'
                    ? 'Backtest in progress...'
                    : selected.backtest_status === 'failed'
                    ? 'Backtest failed.'
                    : 'No backtest results yet.'}
                </div>
              ) : selected.backtest_result.windows.length === 0 ? (
                <div className="empty-state">No valid windows found.</div>
              ) : (
                <div className="windows-list">
                  {selected.backtest_result.windows.map((window, index) => (
                    <WindowDetail key={index} window={window} index={index} />
                  ))}
                </div>
              )}
            </>
          )}

          {detailTab === 'generation' && (
            <>
              {selectedTopicRuns.length > 0 ? (
                <div className="topic-sections">
                  {selectedTopicRuns.map((topicRun) => (
                    <TopicGenerationSection key={topicRun.topic_id} topicRun={topicRun} />
                  ))}
                </div>
              ) : !selected.generation ? (
                <div className="empty-state">
                  {selected.generation_status === 'running'
                    ? 'Generation in progress...'
                    : 'No ideas generated yet.'}
                </div>
              ) : (
                <>
                  <div className="generation-header">
                    <span className="generation-cutoff">
                      Ideas generated at cutoff:{' '}
                      <strong>{selected.generation.cutoff_month}</strong>
                    </span>
                  </div>
                  <div className="predictions-grid">
                    {selected.generation.predictions.map((pred) => (
                      <PredictionCard key={pred.rank} pred={pred} />
                    ))}
                  </div>
                </>
              )}
            </>
          )}
        </section>
      )}
    </div>
  );
};

export default Dashboard;
