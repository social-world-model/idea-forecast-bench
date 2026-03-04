import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchRuns } from '../api/runsApi';
import type { RunRecord } from '../types';

const RunPage: React.FC = () => {
  const navigate = useNavigate();
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    const loadRuns = async () => {
      try {
        const data = await fetchRuns(20);
        if (mounted) {
          setRuns(data);
          setError('');
        }
      } catch (err) {
        if (mounted) {
          setError(err instanceof Error ? err.message : 'Failed to load runs');
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    loadRuns();
    const timer = window.setInterval(loadRuns, 10000);

    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, []);

  const activeRun = useMemo(() => {
    return runs.find((run) => run.status === 'running' || run.status === 'pending') ?? runs[0] ?? null;
  }, [runs]);

  return (
    <div className="runs-page">
      <h1>Run Status</h1>
      <p className="muted">Read-only run monitor. This page does not start new runs.</p>

      {error && <div className="error-box">{error}</div>}

      {loading && <div className="panel">Loading runs...</div>}

      {activeRun && !loading && (
        <div className="panel">
          <h2>Latest Run</h2>
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

      {!loading && runs.length === 0 && !error && (
        <div className="panel">
          <p className="muted">No runs available.</p>
        </div>
      )}
    </div>
  );
};

export default RunPage;
