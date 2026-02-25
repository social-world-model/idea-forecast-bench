import React, { useEffect, useMemo, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { fetchRunDetail } from '../api/runsApi';
import type { RunRecord } from '../types';
import { downloadCsv, downloadJson } from '../utils/export';

const RunDetailPage: React.FC = () => {
  const { runId = '' } = useParams();
  const [run, setRun] = useState<RunRecord | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchRunDetail(runId, true);
        setRun(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load run detail');
      }
    };
    load();
  }, [runId]);

  const ideaRows = useMemo(() => {
    const ideas = (run?.ideas || []) as Array<Record<string, unknown>>;
    return ideas.map((idea) => [
      String(idea.Title || ''),
      String(idea.Score || ''),
      String(idea.Novelty || ''),
      String(idea.Feasibility || ''),
      String(idea.source_url || ''),
    ]);
  }, [run]);

  const exportIdeas = () => {
    downloadCsv(['Title', 'Score', 'Novelty', 'Feasibility', 'source_url'], ideaRows, `${runId}-ideas.csv`);
  };

  if (error) {
    return <div className="runs-page"><div className="error-box">{error}</div></div>;
  }

  if (!run) {
    return <div className="runs-page">Loading run detail...</div>;
  }

  return (
    <div className="runs-page">
      <h1>Run Detail</h1>
      <p className="muted">Run ID: {run.run_id}</p>

      <div className="row-actions">
        <Link to="/runs/history">Back to History</Link>
        <button onClick={() => downloadJson(run, `${run.run_id}.json`)}>Export Run JSON</button>
        <button onClick={exportIdeas} disabled={!ideaRows.length}>Export Ideas CSV</button>
      </div>

      <div className="panel">
        <h2>Status</h2>
        <p><strong>Status:</strong> <span className={`status ${run.status}`}>{run.status}</span></p>
        <p><strong>Created:</strong> {new Date(run.created_at).toLocaleString()}</p>
        <p><strong>Duration:</strong> {run.duration_seconds ?? '-'}s</p>
        <p><strong>Ideas:</strong> {run.ideas_count}</p>
        <p><strong>Keywords:</strong> {run.keywords.join(', ')}</p>
        {run.error && <pre className="error-pre">{run.error}</pre>}
      </div>

      {run.report && (
        <div className="kpi-grid">
          <div className="kpi-card"><span>Avg Score</span><strong>{run.report.average_score}</strong></div>
          <div className="kpi-card"><span>Avg Novelty</span><strong>{run.report.average_novelty}</strong></div>
          <div className="kpi-card"><span>Avg Feasibility</span><strong>{run.report.average_feasibility}</strong></div>
          <div className="kpi-card"><span>Model</span><strong>{run.report.model}</strong></div>
        </div>
      )}

      <div className="panel">
        <h2>Generated Ideas</h2>
        {!ideaRows.length ? (
          <p className="muted">No ideas available.</p>
        ) : (
          <table className="runs-table">
            <thead>
              <tr>
                <th>Title</th>
                <th>Score</th>
                <th>Novelty</th>
                <th>Feasibility</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {(run.ideas as Array<Record<string, unknown>>).map((idea, idx) => (
                <tr key={`${run.run_id}-${idx}`}>
                  <td>{String(idea.Title || '')}</td>
                  <td>{String(idea.Score || '')}</td>
                  <td>{String(idea.Novelty || '')}</td>
                  <td>{String(idea.Feasibility || '')}</td>
                  <td>
                    {idea.source_url ? (
                      <a href={String(idea.source_url)} target="_blank" rel="noreferrer">Open</a>
                    ) : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default RunDetailPage;
