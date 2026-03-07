import React, { useEffect, useMemo, useState } from 'react';
import './GeneratedIdeasList.css';
import { API_BASE_URL, API_ENDPOINTS } from '../config';
import type { IdeaPrediction, Strategy } from '../types';

interface GeneratedIdeaItem {
  id: string;
  strategyId: string;
  strategyName: string;
  strategyType: string;
  cutoffDate: string;
  cutoffMonth: string;
  rank: number;
  prediction: IdeaPrediction;
}

const GeneratedIdeasList: React.FC = () => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ideas = useMemo<GeneratedIdeaItem[]>(() => {
    return strategies
      .flatMap((strategy) => {
        if (!strategy.generation) {
          return [];
        }

        return strategy.generation.predictions.map((prediction) => ({
          id: `${strategy.id}-${strategy.generation?.cutoff_date}-${prediction.rank}`,
          strategyId: strategy.id,
          strategyName: strategy.name,
          strategyType: strategy.strategy_name,
          cutoffDate: strategy.generation?.cutoff_date ?? '',
          cutoffMonth: strategy.generation?.cutoff_month ?? '',
          rank: prediction.rank,
          prediction,
        }));
      })
      .sort((left, right) => {
        const cutoffCmp = right.cutoffDate.localeCompare(left.cutoffDate);
        if (cutoffCmp !== 0) {
          return cutoffCmp;
        }
        const strategyCmp = left.strategyName.localeCompare(right.strategyName);
        if (strategyCmp !== 0) {
          return strategyCmp;
        }
        return left.rank - right.rank;
      });
  }, [strategies]);

  const fetchIdeas = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.STRATEGIES}`);
      if (!response.ok) {
        throw new Error('Failed to fetch strategies');
      }

      const data: Strategy[] = await response.json();
      setStrategies(data);
    } catch (err: unknown) {
      console.error('Error fetching strategy generations:', err);
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchIdeas();
  }, []);

  return (
    <div className="generated-ideas-container">
      <div className="generated-ideas-heading">
        <div>
          <h1 className="generated-ideas-title">Generated Ideas</h1>
          <p className="generated-ideas-subtitle">
            Latest predictions from each strategy&apos;s persisted generation snapshot.
          </p>
        </div>
        <button className="refresh-button" onClick={fetchIdeas} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {error && <div className="error-container">Error: {error}</div>}

      {!loading && ideas.length === 0 && !error && (
        <div className="empty-container">No strategy generations are available yet.</div>
      )}

      <div className="ideas-grid">
        {ideas.map((idea) => (
          <article key={idea.id} className="idea-card">
            <div className="idea-header">
              <div>
                <div className="idea-meta-row">
                  <span className="badge">#{idea.rank}</span>
                  <span className="badge">{idea.strategyType}</span>
                  <span className="badge">cutoff {idea.cutoffMonth}</span>
                </div>
                <h2 className="idea-title">{idea.prediction.title}</h2>
                <p className="idea-source">{idea.strategyName}</p>
              </div>
              <div className="idea-badges">
                <span className="badge confidence-badge">
                  score {(idea.prediction.score * 100).toFixed(0)}%
                </span>
                {idea.prediction.confidence !== undefined && idea.prediction.confidence !== null && (
                  <span className="badge confidence-badge">
                    {(idea.prediction.confidence * 100).toFixed(0)}% confidence
                  </span>
                )}
              </div>
            </div>

            <div className="idea-content">
              <div>
                <div className="section-title">Rationale</div>
                <p className="section-text">{idea.prediction.rationale || 'No rationale provided.'}</p>
              </div>

              <div>
                <div className="section-title">Approach</div>
                <p className="section-text">{idea.prediction.approach || 'No approach provided.'}</p>
              </div>

              <div>
                <div className="section-title">Signals</div>
                <div className="idea-terms">
                  {idea.prediction.key_terms && idea.prediction.key_terms.length > 0 ? (
                    idea.prediction.key_terms.map((term) => (
                      <span key={term} className="term-chip">
                        {term}
                      </span>
                    ))
                  ) : (
                    <>
                      <span className="term-chip">score {(idea.prediction.score * 100).toFixed(0)}%</span>
                      {idea.prediction.confidence !== undefined && idea.prediction.confidence !== null && (
                        <span className="term-chip">confidence {(idea.prediction.confidence * 100).toFixed(0)}%</span>
                      )}
                    </>
                  )}
                </div>
              </div>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
};

export default GeneratedIdeasList;
