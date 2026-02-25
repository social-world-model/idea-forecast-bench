import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

type ScoreTrendPoint = {
  run_id: string;
  timestamp: string;
  average_score: number;
};

interface ScoreTrendChartProps {
  points: ScoreTrendPoint[];
}

const ScoreTrendChart: React.FC<ScoreTrendChartProps> = ({ points }) => {
  if (!points.length) {
    return <div className="chart-empty">No trend data yet.</div>;
  }

  const labels = points
    .slice()
    .reverse()
    .map((point) => new Date(point.timestamp).toLocaleString());
  const scores = points
    .slice()
    .reverse()
    .map((point) => Number(point.average_score || 0));

  const data = {
    labels,
    datasets: [
      {
        label: 'Average Score',
        data: scores,
        borderColor: '#60a5fa',
        backgroundColor: 'rgba(96, 165, 250, 0.25)',
        borderWidth: 2,
        tension: 0.25,
      },
    ],
  };

  return (
    <div className="chart-card">
      <h3>Score Trend</h3>
      <Line
        data={data}
        options={{
          responsive: true,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true },
          },
        }}
      />
    </div>
  );
};

export default ScoreTrendChart;
