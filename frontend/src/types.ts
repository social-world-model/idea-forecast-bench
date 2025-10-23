// Centralized shared types to avoid duplication across components

export interface ResearchIdea {
  id: string;
  title: string;
  description: string;
  author: string;
  institution?: string;
  tags: string[];
  upvotes: number;
  created_at: string;
  updated_at?: string;
  url?: string;
  citations?: number;
  impact_score?: number;
}
