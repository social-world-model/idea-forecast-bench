import os
import sys
import json
from typing import List, Dict, Any
from pathlib import Path
import openreview

# Add project root to path to allow imports if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import Config
from backend import config
from backend import llm_utils, output_contract, prompt_registry


_PROMPT_ID = "llm_baseline"
_PROMPT_VERSION = "v1"
_JSON_DECODER = json.JSONDecoder()


def _build_user_message(paper: Dict[str, Any]) -> str:
    title = str(paper.get("title") or "").strip()
    abstract = str(paper.get("abstract") or "").strip()
    return (
        "Generate exactly one research idea for this paper context. "
        "Return JSON object only.\n"
        f"Paper title: {title}\n"
        f"Paper abstract: {abstract}"
    )


def _extract_json_object(response_text: str) -> Dict[str, Any] | None:
    text = response_text.strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = _JSON_DECODER.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    return None

def fetch_papers_from_openreview(keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetches papers from OpenReview based on keywords and config settings.
    """
    print(f"Fetching papers from OpenReview (Venue: {config.VENUE_ID})...")
    
    try:
        client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
        
        # Fetch accepted papers from the venue
        # Note: 'content.venueid' might vary slightly, but usually it's the venue ID.
        # For ICLR 2024/2025, we look for submissions.
        submissions = client.get_all_notes(content={'venueid': config.VENUE_ID}, details='directReplies')
        
        print(f"Total submissions found in venue: {len(submissions)}")
        
        filtered_papers = []
        
        for note in submissions:
            if len(filtered_papers) >= limit:
                break
                
            title = note.content.get('title', {}).get('value', '')
            abstract = note.content.get('abstract', {}).get('value', '')
            
            paper_keywords = note.content.get('keywords', {}).get('value', [])
            
            # Simple keyword matching
            # Check if ANY of our search keywords is in the paper's keywords list (case-insensitive)
            match = False
            for search_keyword in keywords:
                # Check if search_keyword is contained in any of the paper_keywords
                # We use 'in' to allow partial matches (e.g. "LLM" matches "LLM Agents")
                # or exact match? User said "check keyword, if there is a match". 
                # Usually exact match or substring match on the keyword tag is better.
                # Let's do substring match against the list of paper keywords.
                for pk in paper_keywords:
                    if search_keyword.lower() in pk.lower():
                        match = True
                        break
                if match:
                    break
            
            if match:
                paper_info = {
                    "title": title,
                    "abstract": abstract,
                    "url": f"https://openreview.net/forum?id={note.id}",
                    "id": note.id
                }
                filtered_papers.append(paper_info)
                
        print(f"Filtered down to {len(filtered_papers)} papers matching keywords.")
        return filtered_papers
        
    except Exception as e:
        print(f"Error fetching from OpenReview: {e}")
        return []

def generate_ideas(keywords: List[str] | None = None, n: int = 5) -> List[Dict[str, Any]]:
    """
    Orchestrates the workflow: Fetch Papers (OpenReview) -> Generate Ideas.
    Returns a list of generated ideas.
    """
    # Use keywords from config if not provided
    if not keywords:
        keywords = [str(item) for item in config.KEYWORDS]
    else:
        keywords = [str(item) for item in keywords]
        
    print(f"Using keywords: {keywords}")
    
    papers = fetch_papers_from_openreview(keywords, limit=n)

    if not papers:
        print("No papers found from OpenReview.")
        return []

    print(f"Found {len(papers)} papers. Generating ideas...")
    
    generated_ideas: List[Dict[str, Any]] = []

    try:
        policy = prompt_registry.get_prompt_policy(_PROMPT_ID, _PROMPT_VERSION)
        model_id = str(config.MODEL or policy.get("model_id") or "gpt-4o-mini")
        client, resolved_model = llm_utils.create_client(model_id)
        system_message = str(policy.get("template") or "").strip()
        temperature = float(policy.get("temperature", 0.7))
        if not system_message:
            print(
                f"Prompt template is empty for {_PROMPT_ID}@{_PROMPT_VERSION}."
            )
            return []
    except Exception as e:
        print(f"Failed to initialize prompt-only generator: {e}")
        return []

    for i, paper in enumerate(papers):
        print(f"Generating idea for paper {i+1}/{len(papers)}: {paper['title']}")

        try:
            raw_text, _ = llm_utils.get_response_from_llm(
                msg=_build_user_message(paper),
                client=client,
                model=resolved_model,
                system_message=system_message,
                temperature=temperature,
            )

            raw_idea = _extract_json_object(raw_text)
            if raw_idea is None:
                print(f"Skipping malformed JSON output for {paper['title']}")
                continue

            try:
                idea = output_contract.normalize_idea(raw_idea)
            except ValueError as exc:
                print(
                    f"Skipping invalid normalized output for {paper['title']}: {exc}"
                )
                continue

            idea["source_paper"] = paper["title"]
            idea["source_url"] = paper["url"]
            idea["id"] = f"idea_{i}_{os.urandom(4).hex()}"

            generated_ideas.append(idea)
            print(f"Generated idea: {idea.get('Title', 'Untitled')}")

        except Exception as e:
            print(f"Error generating idea for {paper['title']}: {e}")

    return generated_ideas

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate ideas from ICLR papers (CLI)")
    parser.add_argument("--keywords", nargs="+", default=None, help="Keywords to search for (default: use config.py)")
    parser.add_argument("--n", type=int, default=config.NUM_PAPERS_TO_FETCH, help="Number of papers to use")
    parser.add_argument("--output", default="backend/generated_ideas.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    keywords = args.keywords
    if not keywords:
        keywords = config.KEYWORDS
        
    print(f"Generating ideas for keywords: {keywords}")
    try:
        ideas = generate_ideas(keywords, args.n)
        
        if not ideas:
            print("No ideas generated.")
            return

        # Save to file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(ideas, f, indent=2)
            
        print(f"Successfully saved {len(ideas)} ideas to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
