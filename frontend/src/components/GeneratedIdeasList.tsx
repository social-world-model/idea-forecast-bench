import React, { useEffect, useState } from 'react';
import './GeneratedIdeasList.css';
import { API_BASE_URL, API_ENDPOINTS } from '../config';

interface GeneratedIdea {
    id: string;
    Title: string;
    Background?: string;
    Method?: string;
    Experiment?: any;
    ComparisonTable?: string;
    source_paper?: string;
    source_url?: string;
    NoveltyScore?: number;
    FeasibilityScore?: number;
    ImpactScore?: number;
    [key: string]: any;
}

const GeneratedIdeasList: React.FC = () => {
    const [ideas, setIdeas] = useState<GeneratedIdea[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchIdeas = async () => {
        setLoading(true);
        setError(null);
        try {
            // Read-only fetch: backend returns cached generated ideas.
            const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.GENERATE_IDEAS}`);

            if (!response.ok) {
                throw new Error('Failed to fetch ideas');
            }

            const data = await response.json();
            setIdeas(data);
        } catch (err: any) {
            console.error("Error fetching ideas:", err);
            setError(err.message || 'An error occurred');
        } finally {
            setLoading(false);
        }
    };

    // Auto-fetch on mount
    useEffect(() => {
        fetchIdeas();
    }, []);

    return (
        <div className="generated-ideas-container">
            <h1 className="generated-ideas-title">Generated Research Ideas (ICLR)</h1>

            <div className="controls-container" style={{ marginBottom: '20px', textAlign: 'center' }}>
                <button
                    onClick={fetchIdeas}
                    disabled={loading}
                    style={{ padding: '10px 20px', cursor: loading ? 'not-allowed' : 'pointer' }}
                >
                    {loading ? 'Refreshing...' : 'Refresh Ideas'}
                </button>
            </div>

            {error && <div className="error-container">Error: {error}</div>}

            {ideas.length === 0 && !loading && !error && (
                <div className="empty-container">No cached generated ideas are available yet.</div>
            )}

            <div className="ideas-grid">
                {ideas.map((idea) => (
                    <div key={idea.id} className="idea-card">
                        <div className="idea-header">
                            <div>
                                <h2 className="idea-title">{idea.Title}</h2>
                                {idea.source_paper && (
                                    <p className="idea-source">
                                        Based on: <a href={idea.source_url || '#'} target="_blank" rel="noopener noreferrer">{idea.source_paper}</a>
                                    </p>
                                )}
                            </div>
                            <div className="idea-badges">
                                {idea.NoveltyScore && <span className="badge">Novelty: {idea.NoveltyScore}</span>}
                                {idea.FeasibilityScore && <span className="badge">Feasibility: {idea.FeasibilityScore}</span>}
                                {idea.ImpactScore && <span className="badge">Impact: {idea.ImpactScore}</span>}
                            </div>
                        </div>

                        <div className="idea-content">
                            {idea.Background && (
                                <div>
                                    <div className="section-title">Background</div>
                                    <p className="section-text">{idea.Background}</p>
                                </div>
                            )}
                            {idea.Method && (
                                <div>
                                    <div className="section-title">Proposed Method</div>
                                    <p className="section-text">{idea.Method}</p>
                                </div>
                            )}
                            {idea.ComparisonTable && (
                                <div>
                                    <div className="section-title">Comparison</div>
                                    <div className="comparison-table">
                                        {idea.ComparisonTable}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default GeneratedIdeasList;
