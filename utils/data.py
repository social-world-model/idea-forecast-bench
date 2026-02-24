import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class SimilarityPrompt(BaseModel):
    system_prompt: str
    user_prompt_template: str

@dataclass
class MatchResult:
    """Encapsulates the result of a similarity match."""
    score: float
    reasoning: Optional[str] = None
    engine_name: str = "Unknown"

def clean_paper_content(content: str) -> str:
    """
    Removes the references section and everything after it from the paper content.
    """
    # Common patterns for references header in markdown
    ref_patterns = [
        r"^#+\s*REFERENCES\s*$",
        r"^#+\s*References\s*$",
        r"^REFERENCES\s*$",
        r"^References\s*$"
    ]
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        strip_line = line.strip()
        for pattern in ref_patterns:
            if re.match(pattern, strip_line):
                return "\n".join(lines[:i])
    
    return content
