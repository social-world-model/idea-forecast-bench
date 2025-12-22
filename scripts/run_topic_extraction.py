import os
import sys
import openai
from pathlib import Path

# Add the project root to sys.path to allow importing from src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.topic_extraction import get_markdown_files, read_file_content, extract_domain_keywords

def main():
    # Define the path to the md_mineru directory relative to the project root
    md_dir = Path("/mnt/disk1_from_server2/haofeiy2/live-idea-bench/md_mineru")
    
    print(f"Searching for markdown files in: {md_dir}")
    
    if not md_dir.exists():
        print(f"Directory not found: {md_dir}")
        return

    md_files = get_markdown_files(str(md_dir))
    print(f"Found {len(md_files)} markdown files.")
    
    # Initialize OpenAI client
    # Assumes OPENAI_API_KEY is in environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-key'")
        return
        
    client = openai.OpenAI(api_key=api_key)

    # Process files
    results = {}
    for i, file_path in enumerate(md_files):
        print(f"\n[{i+1}/{len(md_files)}] Processing: {file_path}")
        content = read_file_content(file_path)
        
        if not content:
            print("  Skipping empty file.")
            continue
            
        # Extract keywords
        keywords = extract_domain_keywords(content, client)
        
        print(f"  Keywords: {keywords}")
        results[file_path] = keywords
        
    # Optional: Save results to a file (e.g., JSON)
    # import json
    # with open(project_root / "keywords_results.json", 'w', encoding='utf-8') as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)
    # print(f"\nResults saved to {project_root / 'keywords_results.json'}")

if __name__ == "__main__":
    main()

