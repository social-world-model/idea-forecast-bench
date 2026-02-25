import os
import sys
import json
import time
from typing import List, Dict, Any
from pathlib import Path
import openreview

# Add project root to path to allow imports if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import Thinker
from backend.tiny_scientist.thinker import Thinker
# Import Config
from backend import config

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

def generate_ideas(keywords: List[str] = None, n: int = 5) -> List[Dict[str, Any]]:
    """
    Orchestrates the workflow: Fetch Papers (OpenReview) -> Generate Ideas.
    Returns a list of generated ideas.
    """
    # Use keywords from config if not provided
    if not keywords:
        keywords = config.KEYWORDS
        
    print(f"Using keywords: {keywords}")
    
    papers = fetch_papers_from_openreview(keywords, limit=n)

    if not papers:
        print("No papers found from OpenReview.")
        return []

    print(f"Found {len(papers)} papers. Generating ideas...")
    
    generated_ideas = []
    
    # Initialize Thinker
    # Note: This requires GOOGLE_API_KEY or other LLM API key to be set in environment
    try:
        thinker = Thinker(
            tools=[], 
            iter_num=1,
            search_papers=False, 
            model=config.MODEL,
            generate_exp_plan=False,
        )
    except Exception as e:
        print(f"Failed to initialize Thinker: {e}")
        return []
    
    for i, paper in enumerate(papers):
        print(f"Generating idea for paper {i+1}/{len(papers)}: {paper['title']}")
        
        intent = paper['abstract']
        
        try:
            # Use thinker.run to generate idea with full pipeline (refinement, experiment plan, etc.)
            # run returns a list of dicts (if num_ideas > 1) or a single dict/list (if num_ideas=1)
            # We are generating 1 idea per paper here.
            idea_result = thinker.run(
                intent, 
                num_ideas=1, 
                check_novelty=False, 
            )
            
            if not idea_result:
                print(f"No idea generated for {paper['title']}")
                continue
                
            # Handle potential list return type
            if isinstance(idea_result, list):
                if not idea_result:
                    continue
                idea = idea_result[0]
            else:
                idea = idea_result

            # Add metadata
            idea['source_paper'] = paper['title']
            idea['source_url'] = paper['url']
            idea['id'] = f"idea_{i}_{os.urandom(4).hex()}"
            
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
