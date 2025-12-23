import re
import openai
from typing import List

def extract_abstract(text: str) -> str:
    """
    Extract the Abstract section from markdown text.
    Looks for headers like # Abstract, ## ABSTRACT, etc.
    """
    # Try to find a header that contains "Abstract" (case-insensitive)
    abstract_pattern = re.compile(r'^(#+)\s*Abstract', re.IGNORECASE | re.MULTILINE)
    match = abstract_pattern.search(text)
    
    if not match:
        fallback_pattern = re.compile(r'^Abstract\s*\n', re.IGNORECASE | re.MULTILINE)
        match = fallback_pattern.search(text)
        if not match:
            return text[:3000]
    
    start_pos = match.start()
    
    if match.group(0).startswith('#'):
        header_level = len(match.group(1))
        remaining_text = text[match.end():]
        next_header_pattern = re.compile(rf'^#{{1,{header_level}}}\s+', re.MULTILINE)
        next_match = next_header_pattern.search(remaining_text)
        
        if next_match:
            return text[start_pos : match.end() + next_match.start()].strip()
    
    return text[start_pos : start_pos + 5000].strip()


async def extract_keywords(
    text: str,
    client: openai.AsyncOpenAI,
    model: str,
) -> List[str]:
    prompt = (
        "Extract 5–10 domain-specific keywords from the content below.\n"
        "Return ONLY a comma-separated list.\n\n"
        f"Content:\n{text}"
    )

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You extract keywords from technical documents."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        out = (resp.choices[0].message.content or "").strip()
        return [k.strip() for k in out.split(",") if k.strip()]
    except Exception as e:
        print(f"LLM error: {e}")
        return []


async def predict_future_ideas(
    abstracts: List[str],
    client: openai.AsyncOpenAI,
    model: str,
    domain_context: str = "",
    num_ideas: int = 5
) -> str:
    """
    Generate future research ideas based on a list of abstracts.
    """
    context_text = "\n\n---\n\n".join(abstracts[:20])
    
    prompt = f"""
You are a senior research scientist. Based on the following abstracts from {domain_context}, 
predict {num_ideas} specific "next-generation" research ideas that would likely emerge in the following months.

Abstracts:
{context_text}

For each idea, provide:
1. Title
2. Rationale (why this follows from the provided papers)
3. Potential implementation approach
"""

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant specialized in trend forecasting."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content or "No ideas generated."
    except Exception as e:
        return f"Prediction error: {e}"

