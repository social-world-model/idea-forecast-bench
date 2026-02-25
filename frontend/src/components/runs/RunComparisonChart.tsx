import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

type ComparisonItem = {
  run_id: string;
  average_score: number;
  average_novelty: number;
  average_feasibility: number;
};

interface RunComparisonChartProps {
  runs: ComparisonItem[];
}

const RunComparisonChart: React.FC<RunComparisonChartProps> = ({ runs }) => {
  ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

  if (!runs.length) {
    return <div className="chart-empty">No run comparison available yet.</div>;
  }

  const labels = runs.map((run) => run.run_id.slice(0, 8));
  const data = {
    labels,
    datasets: [
      {
        label: 'Score',
        data: runs.map((run) => run.average_score),
        backgroundColor: 'rgba(56, 189, 248, 0.8)',
      },
      {
        label: 'Novelty',
        data: runs.map((run) => run.average_novelty),
        backgroundColor: 'rgba(45, 212, 191, 0.8)',
      },
      {
        label: 'Feasibility',
        data: runs.map((run) => run.average_feasibility),
        backgroundColor: 'rgba(251, 191, 36, 0.8)',
      },
    ],
  };

  return (
    <div className="chart-card">
      <h3>Top Run Comparison</h3>
      <Bar
        data={data}
        options={{
          responsive: true,
          scales: {
            y: { beginAtZero: true, suggestedMax: 10 },
          },
        }}
      />
    </div>
  );
};

export default RunComparisonChart;
