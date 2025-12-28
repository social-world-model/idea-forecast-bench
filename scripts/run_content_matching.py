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
        default="Graph Representation Learning",
        help="The research idea description to test."
    )
    parser.add_argument(
        "--file", 
        type=str, 
        help="Path to a specific .md file in the md_mineru folder. If not provided, tests the first file found."
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to a folder containing .md files. If provided, matches the idea against all files in this folder."
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

    # 4. Resolve target files
    target_files = []
    if args.folder:
        folder_path = Path(args.folder)
        if not folder_path.is_absolute():
            folder_path = PROJECT_ROOT / args.folder
        
        if not folder_path.exists():
            print(f"[-] Error: Folder not found at {folder_path}")
            return
        
        target_files = list(folder_path.glob("**/*.md"))
        if not target_files:
            print(f"[-] Error: No markdown files found in folder {folder_path}")
            return
        print(f"[*] Found {len(target_files)} files in folder: {folder_path}")
    elif args.file:
        target_file = Path(args.file)
        if not target_file.is_absolute():
            target_file = md_dir / args.file
        
        if not target_file.exists():
            print(f"[-] Error: File not found at {target_file}")
            return
        target_files = [target_file]
    else:
        # Pick the first .md file in md_mineru (recursively find one)
        md_files = list(md_dir.glob("**/*.md"))
        if not md_files:
            print(f"[-] Error: No markdown files found in {md_dir}")
            return
        target_files = [md_files[0]]

    # 5. Execute Matching
    print(f"[*] Idea: {args.idea}")
    print(f"[*] Engine: {args.engine}")
    
    matcher = ResearchMatcher(engine=engine)
    
    for i, target_file in enumerate(target_files):
        print(f"[*] [{i+1}/{len(target_files)}] Comparing with: {target_file.relative_to(PROJECT_ROOT) if target_file.is_relative_to(PROJECT_ROOT) else target_file.name}")
        
        result = matcher.match_idea_to_file(args.idea, target_file)
        
        # 6. Output Result
        print_result(result, target_file.name)

if __name__ == "__main__":
    main()
