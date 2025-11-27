from typing import Dict

from pydantic import BaseModel


class ThinkerPrompt(BaseModel):
    idea_system_prompt: str
    evaluation_system_prompt: str
    idea_evaluation_prompt: str
    modify_idea_prompt: str
    merge_ideas_prompt: str
    query_prompt: str
    rethink_query_prompt: str
    novelty_query_prompt: str
    novelty_system_prompt: str
    idea_first_prompt: str
    idea_reflection_prompt: str
    novelty_prompt: str
    experiment_plan_prompt: str
    non_experiment_plan_prompt: str


class SafetyPrompt(BaseModel):
    risk_assessment_system_prompt: str
    attack_detection_system_prompt: str
    ethical_defense_system_prompt: str
    ethical_defense_prompt: str
