import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import openai

from src.configs import Config
from src.topic_extraction import read_file_content
from utils.data import MatchResult, clean_paper_content

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

class SimilarityEngine(ABC):
    @abstractmethod
    def compute_similarity(self, idea: str, context: str) -> MatchResult:
        """
        Compute similarity between a research idea and a document context.
        Returns a MatchResult object.
        """
        pass

class LLMSimilarityEngine(SimilarityEngine):
    def __init__(self, client: openai.OpenAI, config: Config):
        self.client = client
        self.config = config
        self.engine_name = f"LLM ({config.model_name})"

    def compute_similarity(self, idea: str, context: str) -> MatchResult:
        if not self.config.prompt_template or not self.config.prompt_template.similarity_prompt:
            return MatchResult(score=0.0, reasoning="Prompt template not loaded", engine_name=self.engine_name)

        max_chars = self.config.max_context_chars
        if len(context) > max_chars:
            half = max_chars // 2
            context = context[:half] + "\n... [TRUNCATED] ...\n" + context[-half:]

        sim_prompt = self.config.prompt_template.similarity_prompt
        system_content = sim_prompt.system_prompt
        user_content = sim_prompt.user_prompt_template.format(idea=idea, context=context)

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=self.config.temperature,
            )
            
            content = response.choices[0].message.content.strip()
            score = 0.0
            reasoning = ""
            
            score_match = re.search(r"Score:\s*([0-1](\.\d+)?)", content)
            if score_match:
                score = float(score_match.group(1))
            
            reasoning_match = re.search(r"Reasoning:\s*(.*)", content, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                
            return MatchResult(
                score=max(0.0, min(1.0, score)),
                reasoning=reasoning,
                engine_name=self.engine_name
            )
        except Exception as e:
            print(f"Error computing LLM similarity: {e}")
            return MatchResult(score=0.0, reasoning=f"Error: {str(e)}", engine_name=self.engine_name)

class SentenceTransformerSimilarityEngine(SimilarityEngine):
    def __init__(self, config: Config):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Required dependencies (sentence-transformers, scikit-learn) not installed.")
        
        self.config = config.embedding
        self.engine_name = f"Embedding ({self.config.model_name})"
        self.model = SentenceTransformer(self.config.model_name)

    def _extract_sections(self, context: str) -> Dict[str, str]:
        sections = {
            'abstract': '',
            'introduction': '',
            'method': '',
            'conclusion': '',
            'full': context
        }
        
        # Section extraction patterns
        patterns = {
            'abstract': r'#\s*Abstract\s*\n(.*?)(?=\n#|\Z)',
            'introduction': r'#\s*(?:\d+\s*)?Introduction\s*\n(.*?)(?=\n#|\Z)',
            'method': r'#\s*(?:\d+\s*)?(?:Method|Methodology|Approach|Proposed|Technique)\s*\n(.*?)(?=\n#|\Z)',
            'conclusion': r'#\s*(?:\d+\s*)?(?:Conclusion|Discussion|Summary)\s*\n(.*?)(?=\n#|\Z)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, context, re.DOTALL | re.IGNORECASE)
            if match:
                sections[key] = match.group(1).strip()
        
        return sections

    def _encode(self, text: str) -> np.ndarray:
        if not text or len(text.strip()) == 0:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        # Basic preprocessing: remove excessive whitespace and truncate
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > self.config.max_context_chars:
            text = text[:self.config.max_context_chars]
            
        return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    def compute_similarity(self, idea: str, context: str) -> MatchResult:
        try:
            idea_embedding = self._encode(idea)
            
            if self.config.use_section_weights:
                sections = self._extract_sections(context)
                weights = {
                    'abstract': self.config.abstract_weight,
                    'introduction': self.config.introduction_weight,
                    'method': self.config.method_weight,
                    'conclusion': self.config.conclusion_weight
                }
                
                scores = {}
                for section, weight in weights.items():
                    if sections[section]:
                        section_embedding = self._encode(sections[section])
                        sim = cosine_similarity(idea_embedding.reshape(1, -1), section_embedding.reshape(1, -1))[0, 0]
                        scores[section] = float(sim)
                    else:
                        scores[section] = 0.0
                
                total_weight = sum(weights[k] for k in scores if scores[k] > 0)
                if total_weight > 0:
                    final_score = sum(scores[k] * weights[k] for k in scores) / total_weight
                else:
                    # Fallback to full content
                    full_embedding = self._encode(context)
                    final_score = cosine_similarity(idea_embedding.reshape(1, -1), full_embedding.reshape(1, -1))[0, 0]
                
                reasoning = ", ".join([f"{k}: {v:.3f}" for k, v in scores.items() if v > 0])
                reasoning = f"Weighted section similarity: {reasoning}"
            else:
                full_embedding = self._encode(context)
                final_score = cosine_similarity(idea_embedding.reshape(1, -1), full_embedding.reshape(1, -1))[0, 0]
                reasoning = "Cosine similarity of full document embedding"

            return MatchResult(
                score=max(0.0, min(1.0, float(final_score))),
                reasoning=reasoning,
                engine_name=self.engine_name
            )
        except Exception as e:
            return MatchResult(score=0.0, reasoning=f"Error: {str(e)}", engine_name=self.engine_name)

class ResearchMatcher:
    def __init__(self, engine: SimilarityEngine):
        self.engine = engine

    def match_idea_to_file(self, idea: str, file_path: Union[str, Path]) -> MatchResult:
        content = read_file_content(str(file_path))
        if not content:
            return MatchResult(score=0.0, reasoning="Empty or missing file", engine_name=self.engine.engine_name)
            
        content = clean_paper_content(content)
        return self.engine.compute_similarity(idea, content)

def example_usage():
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    prompt_dir = project_root / "prompt"
    
    try:
        config = Config.load_config(str(config_path), str(prompt_dir))
    except Exception as e:
        print(f"Failed to load config: {e}")
        return

    # Test LLM Engine
    api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
    if api_key:
        client = openai.OpenAI(api_key=api_key)
        llm_engine = LLMSimilarityEngine(client=client, config=config)
        matcher = ResearchMatcher(engine=llm_engine)
        print("\n--- Testing LLM Engine ---")
        # ... logic ...
    
    # Test Embedding Engine
    if DEPENDENCIES_AVAILABLE:
        embed_engine = SentenceTransformerSimilarityEngine(config=config)
        matcher = ResearchMatcher(engine=embed_engine)
        print("\n--- Testing Embedding Engine ---")
        # ... logic ...
