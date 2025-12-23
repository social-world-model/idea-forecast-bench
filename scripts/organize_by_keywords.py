import json
import os
import shutil
import sys
from pathlib import Path
import argparse

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils import load_json, group_by_keywords

def main():
    parser = argparse.ArgumentParser(description="Group papers by keywords and optionally organize files.")
    parser.add_argument("--input", type=str, default="../keywords_results.json", help="Path to the keywords results JSON file.")
    parser.add_argument("--output", type=str, default="keyword_subsets.json", help="Path to the output grouped JSON file.")
    parser.add_argument("--min_papers", type=int, default=1, help="Minimum number of papers a keyword must have to be processed.")
    parser.add_argument("--organize_dir", type=str, help="If provided, will also create a directory structure with symlinks/copies.")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink", help="Mode for --organize_dir: symlink or copy.")
    parser.add_argument("--target_keywords", type=str, help="Comma-separated list or path to a file containing target category names.")
    parser.add_argument("--fuzzy_threshold", type=float, default=0.6, help="Fuzzy matching threshold (0.0 to 1.0).")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (Path(__file__).parent / input_path).resolve()
        
    output_json = Path(args.output)
    if not output_json.is_absolute():
        output_json = (Path(__file__).parent / output_json).resolve()

    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        return

    # Load target keywords if provided
    targets = None
    if args.target_keywords:
        if os.path.exists(args.target_keywords):
            with open(args.target_keywords, 'r') as f:
                targets = [line.strip() for line in f if line.strip()]
        else:
            targets = [t.strip() for t in args.target_keywords.split(",")]
        print(f"Loaded {len(targets)} target categories for fuzzy matching.")

    print(f"Loading results from {input_path}...")
    results = load_json(input_path)
    if not results:
        return

    # Use the centralized grouping function from src.utils
    final_keyword_map = group_by_keywords(
        results, 
        target_categories=targets, 
        fuzzy_threshold=args.fuzzy_threshold,
        min_papers=args.min_papers
    )
    
    print(f"Found {len(final_keyword_map)} resulting categories.")

    # Always save the JSON file
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_keyword_map, f, indent=2, ensure_ascii=False)
    print(f"Grouped JSON saved to {output_json}")

    # Optionally create directory structure
    if args.organize_dir:
        output_base = Path(args.organize_dir)
        if not output_base.is_absolute():
            output_base = (Path(__file__).parent / output_base).resolve()
            
        print(f"Organizing files in {args.mode} mode in {output_base}...")
        processed_count = 0
        for keyword, files in final_keyword_map.items():
            # Clean for filesystem safety
            fs_safe_name = "".join([c if c.isalnum() or c in (" ", "-", "_") else "_" for c in keyword]).strip()
            fs_safe_name = fs_safe_name[:50].strip()
            
            keyword_dir = output_base / fs_safe_name
            keyword_dir.mkdir(parents=True, exist_ok=True)

            for src_path_str in files:
                src_path = Path(src_path_str)
                if not src_path.exists():
                    continue

                dest_path = keyword_dir / src_path.name
                try:
                    if args.mode == "symlink":
                        if dest_path.exists(): dest_path.unlink()
                        dest_path.symlink_to(src_path.resolve())
                    elif args.mode == "copy":
                        shutil.copy2(src_path, dest_path)
                except Exception as e:
                    print(f"  Error processing {src_path.name} for category '{keyword}': {e}")

            processed_count += 1
            if processed_count % 100 == 0:
                print(f"  Processed {processed_count}/{len(final_keyword_map)} categories...")
        print(f"Successfully organized subsets in: {output_base}")

if __name__ == "__main__":
    main()
