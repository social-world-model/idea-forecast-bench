import argparse
import os
import sys
from pathlib import Path

import openai

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.configs import Config
from src.matching import (LLMSimilarityEngine, ResearchMatcher,
                          SentenceTransformerSimilarityEngine)
from utils.data import MatchResult


def parse_args():
    parser = argparse.ArgumentParser(description="Test Research Idea matching against academic documents.")
    parser.add_argument(
        "--idea", 
        type=str, 
        default="Using large language models to automate the extraction of research topics from markdown files.",
        help="The research idea description to test."
    )
    parser.add_argument(
        "--file", 
        type=str, 
        help="Path to a specific .md file in the md_mineru folder. If not provided, tests the first file found."
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["llm", "embedding"],
        default="llm",
        help="The similarity engine to use (default: llm)."
    )
    return parser.parse_args()

def print_result(result: MatchResult, file_name: str):
    """Elegant console output for the match result."""
    border = "=" * 60
    print(f"\n{border}")
    print(f" MATCHING RESULT for: {file_name}")
    print(f"{border}")
    print(f" Engine:    {result.engine_name}")
    print(f" Score:     {result.score:.4f}")
    print(f"\n Reasoning:")
    print(f" {result.reasoning}")
    print(f"{border}\n")

def main():
    args = parse_args()
    
    # 1. Setup paths
    # Point to the actual config location
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    prompt_dir = PROJECT_ROOT / "prompt"
    md_dir = PROJECT_ROOT / "md_mineru"
    
    # 2. Load Configuration
    try:
        config = Config.load_config(str(config_path), str(prompt_dir))
    except Exception as e:
        print(f"[-] Error loading configuration: {e}")
        return

    # 3. Initialize Engine
    if args.engine == "llm":
        api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[-] Error: OPENAI_API_KEY not found in config or environment.")
            return
        client = openai.OpenAI(api_key=api_key)
        engine = LLMSimilarityEngine(client=client, config=config)
    else:
        try:
            engine = SentenceTransformerSimilarityEngine(config=config)
        except ImportError as e:
            print(f"[-] Error: {e}")
            return

    # 4. Resolve target file
    if args.file:
        target_file = Path(args.file)
        if not target_file.is_absolute():
            target_file = md_dir / args.file
    else:
        # Pick the first .md file in md_mineru (recursively find one)
        md_files = list(md_dir.glob("**/*.md"))
        if not md_files:
            print(f"[-] Error: No markdown files found in {md_dir}")
            return
        target_file = md_files[0]

    if not target_file.exists():
        print(f"[-] Error: File not found at {target_file}")
        return

    # 5. Execute Matching
    print(f"[*] Comparing idea with: {target_file.name}")
    print(f"[*] Engine: {args.engine}")
    print(f"[*] Idea: {args.idea[:100]}...")
    
    matcher = ResearchMatcher(engine=engine)
    result = matcher.match_idea_to_file(args.idea, target_file)
    
    # 6. Output Result
    print_result(result, target_file.name)

if __name__ == "__main__":
    main()
