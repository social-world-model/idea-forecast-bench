import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Dashboard.css';
import type {
  Strategy,
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
  const summary: BacktestSummary | undefined | null =
    strategy.backtest_result?.summary;
  const dailyEval = strategy.daily_evaluation ?? null;
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
  const modelId =
    typeof strategy.params?.model_id === 'string'
      ? strategy.params.model_id
      : null;
  const promptId =
    typeof strategy.params?.prompt_id === 'string'
      ? strategy.params.prompt_id
      : null;
  const promptVersion =
    typeof strategy.params?.prompt_version === 'string'
      ? strategy.params.prompt_version
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
          {modelId && <span className="meta-chip">model={modelId}</span>}
          {promptId && (
            <span className="meta-chip">
              prompt={promptId}
              {promptVersion ? `@${promptVersion}` : ''}
            </span>
          )}
          {temperature !== null && (
            <span className="meta-chip">temp={temperature}</span>
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
            source: {dailyEval ? 'daily' : summary ? 'backtest' : 'pending'}
          </span>
          <span className="meta-chip">daily_at: {fmtIso(strategy.last_daily_run_at)}</span>
          <span className="meta-chip">bt: {strategy.backtest_status}</span>
          <span className="meta-chip">gen: {strategy.generation_status}</span>
        </div>
      </td>
    </tr>
  );
};

const PredictionCard: React.FC<{ pred: IdeaPrediction; matchedTerms?: string[] }> = ({
  pred,
  matchedTerms = [],
}) => {
  const [expanded, setExpanded] = useState(false);
  const isMatched = pred.key_terms.some((t) =>
    matchedTerms.includes(t.toLowerCase())
  );

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
            <span className="score-badge confidence" title="Confidence">
              {(pred.confidence * 100).toFixed(0)}%
            </span>
            {isMatched && (
              <span className="score-badge matched-badge" title="Matched future paper">
                ✓ Hit
              </span>
            )}
          </div>
        </div>

        <div className="idea-tags-row">
          {pred.key_terms.map((t, i) => (
            <span
              key={i}
              className={`tag ${matchedTerms.includes(t.toLowerCase()) ? 'match-tag' : ''}`}
            >
              {t}
            </span>
          ))}
        </div>

        {expanded && (
          <div className="idea-expanded">
            <p className="idea-desc">{pred.rationale}</p>
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
          {e.matched_terms.length > 0 && (
            <div className="matched-terms">
              <span className="matched-terms-label">Matched terms: </span>
              {e.matched_terms.map((t, i) => (
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
                matchedTerms={e.matched_terms}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const Dashboard: React.FC<DashboardProps> = ({
  strategies,
  lastRefresh,
  isLoading = false,
}) => {
  const navigate = useNavigate();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detailTab, setDetailTab] = useState<'windows' | 'generation'>('windows');

  const effectiveId = selectedId ?? (strategies[0]?.id ?? null);
  const selected = strategies.find((s) => s.id === effectiveId);

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
          <h1 className="dashboard-title">Live Idea Bench</h1>
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
          <h2 className="section-title">📊 Strategy Leaderboard</h2>
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
                  <th>Score ↓</th>
                  <th>Hit@K</th>
                  <th>MRR</th>
                  <th>Windows</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {strategies.map((s, i) => (
                  <StrategyRow
                    key={s.id}
                    strategy={s}
                    rank={i + 1}
                    isSelected={s.id === effectiveId}
                    onSelect={() => setSelectedId(s.id)}
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
              🔍 Detail
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

          {selected.backtest_result?.summary && (
            <div className="metrics-summary">
              {[
                { label: 'Hit@K', value: pct(selected.backtest_result.summary.avg_hit_at_k), color: '#9c9ef8' },
                { label: 'Recall@K', value: pct(selected.backtest_result.summary.avg_recall_at_k), color: '#6ee7b7' },
                { label: 'Precision@K', value: pct(selected.backtest_result.summary.avg_precision_at_k), color: '#fbbf24' },
                { label: 'MRR', value: selected.backtest_result.summary.avg_mrr.toFixed(3), color: '#f9a8d4' },
                { label: 'Novelty', value: pct(selected.backtest_result.summary.avg_novelty), color: '#a5b4fc' },
                { label: 'Diversity', value: pct(selected.backtest_result.summary.avg_diversity), color: '#86efac' },
                { label: 'Windows', value: String(selected.backtest_result.summary.windows), color: 'rgba(255,255,255,0.6)' },
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
              {!selected.backtest_result ? (
                <div className="empty-state">
                  {selected.backtest_status === 'running'
                    ? 'Backtest in progress...'
                    : selected.backtest_status === 'failed'
                    ? 'Backtest failed.'
                    : 'No backtest results yet.'}
                </div>
              ) : selected.backtest_result.windows.length === 0 ? (
                <div className="empty-state">
                  No valid windows found.
                </div>
              ) : (
                <div className="windows-list">
                  {selected.backtest_result.windows.map((w, i) => (
                    <WindowDetail key={i} window={w} index={i} />
                  ))}
                </div>
              )}
            </>
          )}

          {detailTab === 'generation' && (
            <>
              {!selected.generation ? (
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
