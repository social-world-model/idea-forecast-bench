import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml
from pydantic import BaseModel

from src.types import SimilarityPrompt

T = TypeVar("T", bound=BaseModel)

class PromptTemplate(BaseModel):
    similarity_prompt: SimilarityPrompt

class EmbeddingConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    max_context_chars: int = 50000
    use_section_weights: bool = True
    abstract_weight: float = 0.4
    introduction_weight: float = 0.2
    method_weight: float = 0.2
    conclusion_weight: float = 0.2

class Config(BaseModel):
    model_name: str
    max_context_chars: int
    temperature: float
    openai_api_key: Optional[str] = None
    embedding: EmbeddingConfig = EmbeddingConfig()
    prompt_template: Optional[PromptTemplate] = None

    @classmethod
    def load_config(cls, config_path: str, prompt_dir: str) -> "Config":
        """
        Loads the main config and the prompt templates.
        """
        # Load main hyperparameters
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        # Handle nesting if embedding config is present in yaml
        embedding_data = config_data.pop("embedding", {})
        config = cls(**config_data)
        if embedding_data:
            config.embedding = EmbeddingConfig(**embedding_data)
        
        # Load prompt templates
        config.prompt_template = PromptTemplate(
            similarity_prompt=cls._load_yaml_file(
                os.path.join(prompt_dir, "similarity.yaml"), SimilarityPrompt
            )
        )
        return config

    @staticmethod
    def _load_yaml_file(file_path: str, model_class: Type[T]) -> T:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file '{file_path}' does not exist.")
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
            return model_class(**data)
