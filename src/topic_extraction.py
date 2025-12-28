import argparse
import glob
import os
from pathlib import Path
from typing import List

import openai

# Note: You need to install the openai library: pip install openai
# Ensure your OPENAI_API_KEY is set in your environment variables.

def get_markdown_files(root_dir: str) -> List[str]:
    """
    Recursively find all markdown files in the specified directory.
    
    Args:
        root_dir: The root directory to search in.
        
    Returns:
        A list of file paths to markdown files.
    """
    # Construct the search pattern for .md files
    pattern = os.path.join(root_dir, "**", "*.md")
    # specific structure for mineru output might be .md or inside folders
    files = glob.glob(pattern, recursive=True)
    return files

def read_file_content(file_path: str) -> str:
    """
    Read the content of a file.
    
    Args:
        file_path: The path to the file.
        
    Returns:
        The content of the file as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def extract_domain_keywords(text: str, client: openai.OpenAI, model: str = "gpt-4o", max_chars: int = 10000) -> List[str]:
    """
    Extract domain keywords from text using an LLM.
    
    Args:
        text: The text to analyze.
        client: The OpenAI client instance.
        model: The model to use (default: gpt-4o).
        max_chars: Maximum number of characters to process (default: 10000).
        
    Returns:
        A list of extracted keywords.
    """
    # Truncate text if it's too long to avoid token limits (rudimentary truncation)
    # Adjust this limit based on the model's context window
    if len(text) > max_chars:
        text = text[:max_chars] + "...(truncated)"

    prompt = f"""
    You are a domain expert. Please read the following academic paper or technical document content and extract 5-10 key domain-specific keywords.
    
    Return ONLY the keywords as a comma-separated list.
    
    Content:
    {text}
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for extracting keywords from technical documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        
        content = response.choices[0].message.content.strip()
        # Split by comma and clean up whitespace
        keywords = [k.strip() for k in content.split(',')]
        return keywords
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Extract domain keywords from markdown files.")
    parser.add_argument("--max_files", type=int, default=10, help="Maximum number of files to process (0 for no limit).")
    parser.add_argument("--max_chars", type=int, default=10000, help="Maximum characters per file to send to LLM.")
    args = parser.parse_args()

    # Define the path to the md_mineru directory relative to this script
    # This script is in src/, so we go up one level to find md_mineru
    base_dir = Path(__file__).parent.parent
    md_dir = base_dir / "md_mineru"
    
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
        return
        
    client = openai.OpenAI(api_key=api_key)

    # Process files
    max_files = args.max_files # Set a maximum limit for processed files
    processed_count = 0
    
    for file_path in md_files:
        if max_files > 0 and processed_count >= max_files:
            print(f"\nReached maximum file limit of {max_files}. Stopping.")
            break
            
        print(f"\nProcessing: {file_path}")
        content = read_file_content(file_path)
        
        if not content:
            continue
            
        # Extract keywords
        keywords = extract_domain_keywords(content, client, max_chars=args.max_chars)
        
        print(f"Keywords: {keywords}")
        
        processed_count += 1
        
        # Optional: Break after a few files for testing purposes
        # break 

if __name__ == "__main__":
    main()

