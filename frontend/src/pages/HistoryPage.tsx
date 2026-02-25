import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { fetchRuns, fetchRunsReport } from '../api/runsApi';
import RunComparisonChart from '../components/runs/RunComparisonChart';
import ScoreTrendChart from '../components/runs/ScoreTrendChart';
import type { RunRecord, RunsReport } from '../types';
import { downloadCsv, downloadJson } from '../utils/export';

const HistoryPage: React.FC = () => {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [report, setReport] = useState<RunsReport | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const load = async () => {
      try {
        const [runsData, reportData] = await Promise.all([fetchRuns(100), fetchRunsReport()]);
        setRuns(runsData);
        setReport(reportData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load history');
      }
    };
    load();
  }, []);

  const exportHistoryCsv = () => {
    downloadCsv(
      ['run_id', 'status', 'created_at', 'duration_seconds', 'ideas_count', 'keywords'],
      runs.map((run) => [
        run.run_id,
        run.status,
        run.created_at,
        run.duration_seconds ?? '',
        run.ideas_count,
        run.keywords.join('|'),
      ]),
      'run-history.csv'
    );
  };

  return (
    <div className="runs-page">
      <h1>History</h1>
      <p className="muted">Review previous runs and compare output quality.</p>

      {error && <div className="error-box">{error}</div>}

      <div className="row-actions">
        <button onClick={exportHistoryCsv}>Export CSV</button>
        {report && <button onClick={() => downloadJson(report, 'run-report.json')}>Export Report JSON</button>}
      </div>

      {report && (
        <>
          <div className="kpi-grid">
            <div className="kpi-card"><span>Total</span><strong>{report.summary.total_runs}</strong></div>
            <div className="kpi-card"><span>Success Rate</span><strong>{Math.round(report.summary.success_rate * 100)}%</strong></div>
            <div className="kpi-card"><span>Avg Duration</span><strong>{report.summary.average_duration_seconds}s</strong></div>
            <div className="kpi-card"><span>Avg Ideas/Run</span><strong>{report.summary.average_ideas_per_run}</strong></div>
          </div>
          <div className="chart-grid">
            <ScoreTrendChart points={report.score_trend} />
            <RunComparisonChart runs={report.comparison} />
          </div>
        </>
      )}

      <div className="panel">
        <h2>Runs</h2>
        <table className="runs-table">
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Status</th>
              <th>Created</th>
              <th>Ideas</th>
              <th>Score</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.run_id}>
                <td>{run.run_id.slice(0, 8)}</td>
                <td><span className={`status ${run.status}`}>{run.status}</span></td>
                <td>{new Date(run.created_at).toLocaleString()}</td>
                <td>{run.ideas_count}</td>
                <td>{run.report?.average_score ?? '-'}</td>
                <td><Link to={`/runs/${run.run_id}`}>Detail</Link></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default HistoryPage;
