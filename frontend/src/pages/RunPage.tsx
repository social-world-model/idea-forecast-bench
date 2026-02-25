import React, { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchRuns, startRun } from '../api/runsApi';
import type { RunRecord } from '../types';

const RunPage: React.FC = () => {
  const navigate = useNavigate();
  const [keywordsInput, setKeywordsInput] = useState('Diffusion Language Model, Multimodal Visual Reasoning');
  const [nInput, setNInput] = useState(5);
  const [activeRun, setActiveRun] = useState<RunRecord | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const parsedKeywords = useMemo(
    () => keywordsInput.split(',').map((item) => item.trim()).filter(Boolean),
    [keywordsInput]
  );

  const handleStart = async () => {
    setLoading(true);
    setError('');
    try {
      const run = await startRun(parsedKeywords, nInput);
      setActiveRun(run);
      const refresh = window.setInterval(async () => {
        const runs = await fetchRuns(20);
        const latest = runs.find((item) => item.run_id === run.run_id);
        if (latest) {
          setActiveRun(latest);
          if (latest.status === 'success' || latest.status === 'failed') {
            window.clearInterval(refresh);
          }
        }
      }, 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start run');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="runs-page">
      <h1>Run</h1>
      <p className="muted">Start a new generation run and monitor its status.</p>

      <div className="panel">
        <label>
          Keywords (comma separated)
          <textarea value={keywordsInput} onChange={(e) => setKeywordsInput(e.target.value)} rows={4} />
        </label>
        <label>
          Papers to fetch
          <input
            type="number"
            min={1}
            value={nInput}
            onChange={(e) => setNInput(Number(e.target.value) || 1)}
          />
        </label>
        <button onClick={handleStart} disabled={loading || parsedKeywords.length === 0}>
          {loading ? 'Starting...' : 'Start Run'}
        </button>
      </div>

      {error && <div className="error-box">{error}</div>}

      {activeRun && (
        <div className="panel">
          <h2>Active Run</h2>
          <p><strong>ID:</strong> {activeRun.run_id}</p>
          <p><strong>Status:</strong> <span className={`status ${activeRun.status}`}>{activeRun.status}</span></p>
          <p><strong>Ideas:</strong> {activeRun.ideas_count}</p>
          {activeRun.error && <pre className="error-pre">{activeRun.error}</pre>}
          <div className="row-actions">
            <button onClick={() => navigate(`/runs/${activeRun.run_id}`)}>Open Detail</button>
            <button onClick={() => navigate('/runs/history')}>Open History</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default RunPage;
