import asyncio
import os
import sys
import argparse
import json
from pathlib import Path
import openai

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils import load_json, read_text
from src.prompting import extract_abstract, predict_future_ideas

async def main():
    parser = argparse.ArgumentParser(description="Predict research ideas based on grouped keyword subsets.")
    parser.add_argument("--input", type=str, default="data/keyword_subsets.json", help="Path to keyword_subsets.json.")
    parser.add_argument("--output", type=str, default="data/predictions_results.json", help="Path to save prediction results.")
    parser.add_argument("--keywords", type=str, help="Comma-separated keywords to filter for.")
    parser.add_argument("--start", type=str, default="1701", help="Start YYMM (e.g., 2301).")
    parser.add_argument("--end", type=str, default="2001", help="End YYMM (e.g., 2401).")
    parser.add_argument("--n_ideas", type=int, default=5, help="Number of ideas to generate.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use.")
    parser.add_argument("--iterative", action="store_true", help="Run iteratively in 3-month steps from start to end.")
    args = parser.parse_args()

    # Helper to increment YYMM by months
    def add_months(yymm: str, months: int) -> str:
        yy = int(yymm[:2])
        mm = int(yymm[2:])
        total_months = yy * 12 + (mm - 1) + months
        new_yy = total_months // 12
        new_mm = (total_months % 12) + 1
        return f"{new_yy:02d}{new_mm:02d}"

    # Generate time windows (3 months each)
    windows = []
    if args.iterative:
        current_start = args.start
        while current_start < args.end:
            current_end = add_months(current_start, 2)
            if current_end > args.end:
                current_end = args.end
            windows.append((current_start, current_end))
            current_start = add_months(current_end, 1)
    else:
        windows = [(args.start, args.end)]

    # 1. Load the grouped results
    results_path = Path(args.input)
    if not results_path.is_absolute():
        results_path = project_root / results_path
    
    keyword_groups = load_json(results_path)
    if not keyword_groups:
        print(f"No data found in {results_path}")
        return

    # Prepare output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing predictions if they exist
    all_predictions = []
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                all_predictions = json.load(f)
        except:
            all_predictions = []

    print(f"Loaded {len(keyword_groups)} keywords from {results_path}")

    # 2. Handle target keywords
    target_keywords = []
    if args.keywords:
        target_keywords = [k.strip().lower() for k in args.keywords.split(",")]
    else:
        top_k = max(keyword_groups, key=lambda k: len(keyword_groups[k]))
        target_keywords = [top_k]
    
    # 3. Initialize LLM Client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set.")
        return
    client = openai.AsyncOpenAI(api_key=api_key)

    # 4. Iterate over windows and keywords
    for win_start, win_end in windows:
        print(f"\n" + "#"*60)
        print(f"TIME WINDOW: {win_start} to {win_end}")
        print("#"*60)

        for keyword in target_keywords:
            actual_keyword = None
            if keyword in keyword_groups:
                actual_keyword = keyword
            else:
                for k in keyword_groups.keys():
                    if k.lower() == keyword.lower():
                        actual_keyword = k
                        break
            
            if not actual_keyword:
                print(f"Keyword '{keyword}' not found in subsets.")
                continue
            
            file_paths = keyword_groups[actual_keyword]
            
            # 5. Filter by date within the keyword's file list for this window
            filtered_paths = []
            for p_str in file_paths:
                p = Path(p_str)
                filename = p.name
                if filename[:4].isdigit():
                    date_prefix = filename[:4]
                    if win_start <= date_prefix <= win_end:
                        filtered_paths.append(p)
            
            print(f"\nDomain: '{actual_keyword}'")
            print(f"  - Papers in window {win_start}-{win_end}: {len(filtered_paths)}")

            if not filtered_paths:
                print(f"  - Skipping: no papers found in this window.")
                continue

            # 6. Extract abstracts
            abstracts = []
            for p in filtered_paths:
                full_text = read_text(p)
                abstract = extract_abstract(full_text)
                if abstract:
                    abstracts.append(abstract)

            if not abstracts:
                print(f"  - Skipping: no abstracts could be extracted.")
                continue

            # 7. Predict
            prediction_text = await predict_future_ideas(
                abstracts=abstracts,
                client=client,
                model=args.model,
                domain_context=f"{actual_keyword} (Period: {win_start}-{win_end})",
                num_ideas=args.n_ideas
            )

            # 8. Save result with metadata
            result_entry = {
                "keyword": actual_keyword,
                "start_date": win_start,
                "end_date": win_end,
                "num_papers": len(filtered_paths),
                "model": args.model,
                "prediction": prediction_text,
                "source_files": [str(p) for p in filtered_paths]
            }
            
            all_predictions.append(result_entry)
            
            # Save after each prediction to avoid data loss
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_predictions, f, indent=2, ensure_ascii=False)

            print(f"  - Prediction saved for {actual_keyword} ({win_start}-{win_end})")

    print(f"\nDone. All predictions saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
