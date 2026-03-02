import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Dashboard.css";
import type {
  Strategy,
  IdeaPrediction,
  BacktestSummary,
  WindowResult,
} from "../types";

// ── Props ────────────────────────────────────────────────────────────────────
export type DashboardProps = {
  strategies: Strategy[];
  lastRefresh?: Date;
  isLoading?: boolean;
  onRunBacktest: (id: string) => void;
  onRunGeneration: (id: string, cutoffMonth?: string) => void;
  onCreateStrategy: (data: Partial<Strategy>) => Promise<Strategy | null>;
  onDeleteStrategy: (id: string) => void;
};

// ── Helpers ──────────────────────────────────────────────────────────────────
function relativeTime(d?: Date): string {
  if (!d) return "";
  const mins = Math.floor((Date.now() - d.getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  return hrs < 24 ? `${hrs}h ago` : `${Math.floor(hrs / 24)}d ago`;
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

// ── Status Pill ───────────────────────────────────────────────────────────────
const StatusPill: React.FC<{ status: string }> = ({ status }) => (
  <span className={`status-pill status-${status}`}>
    {status === "running" && <span className="status-spinner" />}
    {status}
  </span>
);

// ── Metric Bar ────────────────────────────────────────────────────────────────
const MetricBar: React.FC<{ value: number; color?: string }> = ({
  value,
  color = "#9c9ef8",
}) => (
  <div className="metric-bar-wrap">
    <div
      className="metric-bar-fill"
      style={{ width: `${Math.min(value * 100, 100)}%`, background: color }}
    />
  </div>
);

// ── Strategy Table Row ────────────────────────────────────────────────────────
const StrategyRow: React.FC<{
  strategy: Strategy;
  rank: number;
  isSelected: boolean;
  onSelect: () => void;
  onBacktest: () => void;
  onGenerate: () => void;
  onDelete: () => void;
}> = ({ strategy, rank, isSelected, onSelect, onBacktest, onGenerate, onDelete }) => {
  const summary: BacktestSummary | undefined | null =
    strategy.backtest_result?.summary;
  const btRunning = strategy.backtest_status === "running";
  const genRunning = strategy.generation_status === "running";

  const recentMonths =
    typeof strategy.params?.recent_months === "number" ? strategy.params.recent_months : null;
  const minFreq =
    typeof strategy.params?.min_keyword_freq === "number" ? strategy.params.min_keyword_freq : null;
  const modelId =
    typeof strategy.params?.model_id === "string" ? strategy.params.model_id : null;
  const promptId =
    typeof strategy.params?.prompt_id === "string" ? strategy.params.prompt_id : null;
  const promptVersion =
    typeof strategy.params?.prompt_version === "string" ? strategy.params.prompt_version : null;
  const temperature =
    typeof strategy.params?.temperature === "number" ? strategy.params.temperature : null;

  return (
    <tr
      className={`strategy-row ${isSelected ? "selected" : ""}`}
      onClick={onSelect}
    >
      {/* Rank */}
      <td className="col-rank">
        <span className={`rank-badge ${rank <= 3 ? `top-${rank}` : ""}`}>
          #{rank}
        </span>
      </td>

      {/* Name + meta */}
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
              prompt={promptId}{promptVersion ? `@${promptVersion}` : ""}
            </span>
          )}
          {temperature !== null && (
            <span className="meta-chip">temp={temperature}</span>
          )}
        </div>
      </td>

      {/* Time window */}
      <td className="col-window">
        <div className="window-val">
          {strategy.config.start_month} → {strategy.config.end_month}
        </div>
        <div className="window-label">
          horizon {strategy.config.horizon_months}m · top-{strategy.config.top_k}
        </div>
      </td>

      {/* Hit@K */}
      <td className="col-metric">
        {summary ? (
          <>
            <div className="metric-value accent">
              {pct(summary.avg_hit_at_k)}
            </div>
            <MetricBar value={summary.avg_hit_at_k} color="#9c9ef8" />
          </>
        ) : (
          <StatusPill status={strategy.backtest_status} />
        )}
      </td>

      {/* Recall@K */}
      <td className="col-metric">
        {summary ? (
          <>
            <div className="metric-value">{pct(summary.avg_recall_at_k)}</div>
            <MetricBar value={summary.avg_recall_at_k} color="#6ee7b7" />
          </>
        ) : (
          <span className="metric-na">—</span>
        )}
      </td>

      {/* MRR */}
      <td className="col-metric">
        {summary ? (
          <>
            <div className="metric-value">{summary.avg_mrr.toFixed(3)}</div>
            <MetricBar value={summary.avg_mrr} color="#fbbf24" />
          </>
        ) : (
          <span className="metric-na">—</span>
        )}
      </td>

      {/* Windows */}
      <td className="col-ideas">
        <span className="ideas-count">
          {summary ? summary.windows : "—"}
        </span>
      </td>

      {/* Actions */}
      <td className="col-actions" onClick={(e) => e.stopPropagation()}>
        <button
          className="action-btn backtest"
          onClick={onBacktest}
          disabled={btRunning}
          title="Run Backtest"
        >
          {btRunning ? "…" : "⚡ Backtest"}
        </button>
        <button
          className="action-btn generate"
          onClick={onGenerate}
          disabled={genRunning}
          title="Generate Ideas at end_month"
        >
          {genRunning ? "…" : "✨ Generate"}
        </button>
        <button
          className="action-btn delete"
          onClick={onDelete}
          title="Delete"
        >
          ✕
        </button>
      </td>
    </tr>
  );
};

// ── Prediction Card ───────────────────────────────────────────────────────────
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
      className={`idea-card ${expanded ? "expanded" : ""} ${isMatched ? "matched" : ""}`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="idea-rank">
        <span className={`rank-badge ${pred.rank <= 3 ? `top-${pred.rank}` : ""}`}>
          #{pred.rank}
        </span>
      </div>
      <div className="idea-body">
        <div className="idea-header">
          <h3 className="idea-title">{pred.title}</h3>
          <div className="idea-scores">
            <span
              className="score-badge confidence"
              title="Confidence"
            >
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
              className={`tag ${matchedTerms.includes(t.toLowerCase()) ? "match-tag" : ""}`}
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

// ── Window Detail ─────────────────────────────────────────────────────────────
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
            → Eval {w.cutoff_month} – {w.future_end_month}
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
        <span className="window-toggle">{open ? "▲" : "▼"}</span>
      </div>

      {open && (
        <div className="window-body">
          {e.matched_terms.length > 0 && (
            <div className="matched-terms">
              <span className="matched-terms-label">Matched terms: </span>
              {e.matched_terms.map((t, i) => (
                <span key={i} className="tag match-tag">{t}</span>
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

// ── New Strategy Form ─────────────────────────────────────────────────────────
const NewStrategyForm: React.FC<{
  onCreate: (data: Partial<Strategy>) => Promise<Strategy | null>;
  onClose: () => void;
}> = ({ onCreate, onClose }) => {
  const [strategyName, setStrategyName] = useState<"keyword_trend" | "prompt_llm">(
    "keyword_trend"
  );
  const [name, setName] = useState("");
  const [recentMonths, setRecentMonths] = useState(3);
  const [minFreq, setMinFreq] = useState(2);
  const [modelId, setModelId] = useState("gpt-4o-mini");
  const [promptId, setPromptId] = useState("llm_baseline");
  const [promptVersion, setPromptVersion] = useState("v1");
  const [temperature, setTemperature] = useState<number>(0.7);
  const [topK, setTopK] = useState(5);
  const [horizonMonths, setHorizonMonths] = useState(3);
  const [startMonth, setStartMonth] = useState("2024-01");
  const [endMonth, setEndMonth] = useState("2024-12");
  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);

    const defaultName =
      strategyName === "prompt_llm"
        ? `Prompt LLM · ${startMonth} → ${endMonth}`
        : `Keyword Trend · ${startMonth} → ${endMonth}`;

    const params =
      strategyName === "prompt_llm"
        ? {
            model_id: modelId,
            prompt_id: promptId,
            prompt_version: promptVersion,
            temperature,
          }
        : { recent_months: recentMonths, min_keyword_freq: minFreq };

    await onCreate({
      name: name || defaultName,
      strategy_name: strategyName,
      params,
      config: {
        top_k: topK,
        horizon_months: horizonMonths,
        min_train_papers: 3,
        start_month: startMonth,
        end_month: endMonth,
        data_dir: "",  // backend uses DEFAULT_DATA_DIR
      },
    });
    setSaving(false);
    onClose();
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>New Strategy</h2>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <form className="strategy-form" onSubmit={handleSubmit}>
          <label>
            Strategy Type
            <select
              value={strategyName}
              onChange={(e) => setStrategyName(e.target.value as "keyword_trend" | "prompt_llm")}
            >
              <option value="keyword_trend">keyword_trend (heuristic)</option>
              <option value="prompt_llm">prompt_llm (LLM)</option>
            </select>
          </label>

          <label>
            Name (optional)
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Keyword Trend · 2024 H1"
            />
          </label>

          {strategyName === "keyword_trend" ? (
            <div className="form-row">
              <label>
                Recent Months
                <input
                  type="number" min={1} max={24}
                  value={recentMonths}
                  onChange={(e) => setRecentMonths(+e.target.value)}
                />
              </label>
              <label>
                Min Keyword Freq
                <input
                  type="number" min={1} max={20}
                  value={minFreq}
                  onChange={(e) => setMinFreq(+e.target.value)}
                />
              </label>
            </div>
          ) : (
            <>
              <div className="form-row">
                <label>
                  Model ID
                  <input
                    value={modelId}
                    onChange={(e) => setModelId(e.target.value)}
                    placeholder="gpt-4o-mini"
                  />
                </label>
                <label>
                  Temperature
                  <input
                    type="number"
                    min={0}
                    max={2}
                    step={0.1}
                    value={temperature}
                    onChange={(e) => setTemperature(+e.target.value)}
                  />
                </label>
              </div>
              <div className="form-row">
                <label>
                  Prompt ID
                  <input
                    value={promptId}
                    onChange={(e) => setPromptId(e.target.value)}
                    placeholder="llm_baseline"
                  />
                </label>
                <label>
                  Prompt Version
                  <input
                    value={promptVersion}
                    onChange={(e) => setPromptVersion(e.target.value)}
                    placeholder="v1"
                  />
                </label>
              </div>
            </>
          )}

          <div className="form-row">
            <label>
              Top-K Ideas
              <input
                type="number" min={1} max={20}
                value={topK}
                onChange={(e) => setTopK(+e.target.value)}
              />
            </label>
            <label>
              Horizon Months
              <input
                type="number" min={1} max={12}
                value={horizonMonths}
                onChange={(e) => setHorizonMonths(+e.target.value)}
              />
            </label>
          </div>

          <div className="form-row">
            <label>
              Start Month
              <input
                value={startMonth}
                onChange={(e) => setStartMonth(e.target.value)}
                placeholder="2024-01"
                pattern="\d{4}-\d{2}"
              />
            </label>
            <label>
              End Month
              <input
                value={endMonth}
                onChange={(e) => setEndMonth(e.target.value)}
                placeholder="2024-12"
                pattern="\d{4}-\d{2}"
              />
            </label>
          </div>

          <div className="form-actions">
            <button type="button" className="btn-secondary" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn-primary" disabled={saving}>
              {saving ? "Creating…" : "Create Strategy"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// ── Dashboard ─────────────────────────────────────────────────────────────────
const Dashboard: React.FC<DashboardProps> = ({
  strategies,
  lastRefresh,
  isLoading = false,
  onRunBacktest,
  onRunGeneration,
  onCreateStrategy,
  onDeleteStrategy,
}) => {
  const navigate = useNavigate();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [detailTab, setDetailTab] = useState<"windows" | "generation">("windows");

  const effectiveId = selectedId ?? (strategies[0]?.id ?? null);
  const selected = strategies.find((s) => s.id === effectiveId);

  if (isLoading) {
    return (
      <div className="dashboard-container">
        <div className="loading-state">
          <div className="loading-spinner" />
          <p>Loading strategies…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      {/* Header */}
      <div className="dashboard-header">
        <div className="title-container">
          <h1 className="dashboard-title">Live Idea Bench</h1>
        </div>
        <p className="dashboard-subtitle">
          Benchmark for research idea prediction strategies — backtest on historical
          papers, generate forward-looking ideas, and compare strategy performance.
          {" "}<button className="about-link" onClick={() => navigate("/about")}>
            Learn more →
          </button>
        </p>
        <div className="last-updated">Last updated: {relativeTime(lastRefresh)}</div>
      </div>

      {/* Strategy Leaderboard */}
      <section className="section">
        <div className="section-header">
          <h2 className="section-title">📊 Strategy Leaderboard</h2>
          <button className="btn-primary" onClick={() => setShowForm(true)}>
            + New Strategy
          </button>
        </div>

        {strategies.length === 0 ? (
          <div className="empty-state">No strategies yet. Create one to get started.</div>
        ) : (
          <div className="table-wrap">
            <table className="strategy-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Strategy</th>
                  <th>Window</th>
                  <th>Hit@K ↓</th>
                  <th>Recall@K</th>
                  <th>MRR</th>
                  <th>Windows</th>
                  <th>Actions</th>
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
                    onBacktest={() => onRunBacktest(s.id)}
                    onGenerate={() => onRunGeneration(s.id, s.config.end_month)}
                    onDelete={() => onDeleteStrategy(s.id)}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Detail Panel */}
      {selected && (
        <section className="section">
          <div className="section-header">
            <h2 className="section-title">
              🔍 Detail
              <span className="section-subtitle">— {selected.name}</span>
            </h2>
            <div className="tab-bar">
              <button
                className={`tab-btn ${detailTab === "windows" ? "active" : ""}`}
                onClick={() => setDetailTab("windows")}
              >
                Backtest Windows
              </button>
              <button
                className={`tab-btn ${detailTab === "generation" ? "active" : ""}`}
                onClick={() => setDetailTab("generation")}
              >
                Generation
              </button>
            </div>
          </div>

          {/* Summary metrics bar */}
          {selected.backtest_result?.summary && (
            <div className="metrics-summary">
              {[
                { label: "Hit@K", value: pct(selected.backtest_result.summary.avg_hit_at_k), color: "#9c9ef8" },
                { label: "Recall@K", value: pct(selected.backtest_result.summary.avg_recall_at_k), color: "#6ee7b7" },
                { label: "Precision@K", value: pct(selected.backtest_result.summary.avg_precision_at_k), color: "#fbbf24" },
                { label: "MRR", value: selected.backtest_result.summary.avg_mrr.toFixed(3), color: "#f9a8d4" },
                { label: "Novelty", value: pct(selected.backtest_result.summary.avg_novelty), color: "#a5b4fc" },
                { label: "Diversity", value: pct(selected.backtest_result.summary.avg_diversity), color: "#86efac" },
                { label: "Windows", value: String(selected.backtest_result.summary.windows), color: "rgba(255,255,255,0.6)" },
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

          {/* Backtest Windows tab */}
          {detailTab === "windows" && (
            <>
              {!selected.backtest_result ? (
                <div className="empty-state">
                  {selected.backtest_status === "running"
                    ? "Backtest in progress…"
                    : selected.backtest_status === "failed"
                    ? "Backtest failed. Try running again."
                    : "No backtest results yet. Click ⚡ Backtest to run."}
                </div>
              ) : selected.backtest_result.windows.length === 0 ? (
                <div className="empty-state">
                  No valid windows found. Try relaxing min_train_papers or extending the time range.
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

          {/* Generation tab */}
          {detailTab === "generation" && (
            <>
              {!selected.generation ? (
                <div className="empty-state">
                  {selected.generation_status === "running"
                    ? "Generation in progress…"
                    : "No ideas generated yet. Click ✨ Generate to produce forward-looking ideas at end_month."}
                </div>
              ) : (
                <>
                  <div className="generation-header">
                    <span className="generation-cutoff">
                      Ideas generated at cutoff:{" "}
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

      {/* New Strategy Modal */}
      {showForm && (
        <NewStrategyForm
          onCreate={onCreateStrategy}
          onClose={() => setShowForm(false)}
        />
      )}
    </div>
  );
};

export default Dashboard;
