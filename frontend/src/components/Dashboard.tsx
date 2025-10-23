import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Dashboard.css";
import type { ResearchIdea } from "../types";

// ------- Types -------
export type DashboardProps = {
  researchIdeas: ResearchIdea[];
  lastRefresh?: Date | string;
  views?: number;
  isLoading?: boolean;
};

// ------- Helpers -------
function relativeTime(dateLike?: Date | string) {
  if (!dateLike) return "";
  const date = typeof dateLike === "string" ? new Date(dateLike) : dateLike;
  const diffMs = Date.now() - date.getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min${mins === 1 ? "" : "s"} ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs} hour${hrs === 1 ? "" : "s"} ago`;
  const days = Math.floor(hrs / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

// ------- Research Idea Card -------
const ResearchIdeaCard: React.FC<{
  idea: ResearchIdea;
  rank: number;
}> = ({ idea, rank }) => {
  const [isExpanded, setIsExpanded] = React.useState(false);

  const handleClick = (e: React.MouseEvent) => {
    // Don't toggle if clicking on a link
    if ((e.target as HTMLElement).tagName === 'A') {
      return;
    }
    setIsExpanded(!isExpanded);
  };

  return (
    <div 
      className={`research-idea-card ${isExpanded ? 'expanded' : ''}`} 
      onClick={handleClick}
    >
      <div className="idea-rank">
        <span className={`rank-badge ${rank <= 3 ? `top-${rank}` : ''}`}>
          #{rank}
        </span>
      </div>
      <div className="idea-content">
        <div className="idea-header">
          <h3 className="idea-title">{idea.title}</h3>
          <div className="idea-score">
            {idea.impact_score !== undefined ? idea.impact_score.toFixed(1) : idea.upvotes}
          </div>
        </div>
        
        <div className="idea-meta-row">
          {idea.tags && idea.tags.length > 0 && (
            <div className="idea-tags">
              {idea.tags.map((tag, idx) => (
                <span key={idx} className="tag">{tag}</span>
              ))}
            </div>
          )}
          <div className="idea-date">
            {relativeTime(idea.created_at)}
          </div>
        </div>

        {isExpanded && (
          <div className="idea-expanded-content">
            <p className="idea-description">{idea.description}</p>
            <a 
              href={idea.url || '#'} 
              target="_blank" 
              rel="noopener noreferrer"
              className="idea-link"
              onClick={(e) => e.stopPropagation()}
            >
              View Full Proposal →
            </a>
              </div>
        )}
      </div>
    </div>
  );
};

// ------- Main Dashboard Component -------
const Dashboard: React.FC<DashboardProps> = ({
  researchIdeas = [],
  lastRefresh = new Date(),
  views = 0,
  isLoading = false
}) => {
  const navigate = useNavigate();
  const [showAll, setShowAll] = useState(false);
  const [sortBy, setSortBy] = useState<'score' | 'recent'>('score');

  // Format number for display
  const formatNumber = (num: number): string => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1).replace('.0', '') + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1).replace('.0', '') + 'K';
    } else {
      return num.toLocaleString();
    }
  };

  // Sort based on selected criteria
  const sortedIdeas = [...researchIdeas].sort((a, b) => {
    if (sortBy === 'score') {
      const scoreA = a.impact_score !== undefined ? a.impact_score : a.upvotes;
      const scoreB = b.impact_score !== undefined ? b.impact_score : b.upvotes;
      return scoreB - scoreA;
    } else {
      // Sort by recent (created_at)
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    }
  });

  const displayedIdeas = showAll ? sortedIdeas : sortedIdeas.slice(0, 10);

  if (isLoading) {
    return (
      <div className="dashboard-container">
        <div className="loading-state">
          <div className="loading-spinner"></div>
          <p>Loading research ideas...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <div className="title-container">
          <h1 className="dashboard-title">
            Live Idea Benchmark
          </h1>
          <div className="views-badge-small">
            Views: {formatNumber(views)}
          </div>
        </div>
        <p className="dashboard-subtitle">
          A real-time leaderboard for the most impactful research ideas and innovations. Learn more at{" "}
          <button
            className="about-link"
            onClick={() => navigate('/about')}
          >
            About
          </button>.
        </p>

        <div className="info-box">
          We aggregate research ideas from around the world, ranked by their impact score. 
          Each idea is evaluated based on multiple factors to ensure quality and originality.
        </div>

        <div className="last-updated">
          Last updated: {relativeTime(lastRefresh)}
        </div>
      </div>

      <div className="sort-controls">
        <span className="sort-label">Sort by:</span>
        <select 
          className="sort-select"
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as 'score' | 'recent')}
        >
          <option value="score">Score</option>
          <option value="recent">Recent</option>
        </select>
      </div>

      <div className="research-ideas-list">
        {displayedIdeas.length === 0 ? (
          <div className="empty-state">
            <p>No research ideas available</p>
          </div>
        ) : (
          displayedIdeas.map((idea, idx) => (
            <ResearchIdeaCard
              key={idea.id}
              idea={idea}
              rank={idx + 1}
            />
          ))
        )}
      </div>

      {researchIdeas.length > 10 && (
        <div className="view-all-container">
          <button
            className="view-all-btn"
            onClick={() => setShowAll(!showAll)}
          >
            {showAll ? 'Show Less' : `View All (${researchIdeas.length})`}
          </button>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
