import json
import argparse
from pathlib import Path
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description="Analyze keyword distribution from subsets JSON.")
    parser.add_argument("--input", type=str, default="../data/keyword_subsets.json", help="Path to keyword_subsets.json")
    parser.add_argument("--min_range", type=int, default=50, help="Lower bound for specific interest.")
    parser.add_argument("--max_range", type=int, default=100, help="Upper bound for specific interest.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        # Try relative to scripts directory
        input_path = Path(__file__).parent / args.input

    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    print(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Calculate counts
    counts = {k: len(v) for k, v in data.items()}
    total_keywords = len(counts)
    
    if total_keywords == 0:
        print("No keywords found.")
        return

    # 1. Overall Distribution Summary
    bins = [
        (1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 200), (201, float('inf'))
    ]
    distribution = Counter()
    for count in counts.values():
        for b in bins:
            if b[0] <= count <= b[1]:
                distribution[b] += 1
                break

    print("\n" + "="*40)
    print("KEYWORD DISTRIBUTION SUMMARY")
    print("="*40)
    print(f"{'Paper Count Range':<20} | {'Num Keywords':<15}")
    print("-" * 40)
    for b in bins:
        range_str = f"{b[0]}-{b[1] if b[1] != float('inf') else '+'}"
        print(f"{range_str:<20} | {distribution[b]:<15}")

    # 2. Specific Interest: 50-100 papers
    interest_group = [(k, c) for k, c in counts.items() if args.min_range <= c <= args.max_range]
    # Sort by count descending
    interest_group.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "="*40)
    print(f"KEYWORDS WITH {args.min_range}-{args.max_range} PAPERS ({len(interest_group)} total)")
    print("="*40)
    if not interest_group:
        print("None found in this range.")
    else:
        print(f"{'Keyword':<30} | {'Count':<10}")
        print("-" * 40)
        # Show top 50 if too many
        for k, c in interest_group[:50]:
            print(f"{k[:30]:<30} | {c:<10}")
        
        if len(interest_group) > 50:
            print(f"... and {len(interest_group) - 50} more.")

    # 3. Quick Stats
    avg_papers = sum(counts.values()) / total_keywords
    max_k = max(counts, key=counts.get)
    print("\n" + "="*40)
    print("QUICK STATS")
    print("="*40)
    print(f"Total Unique Keywords: {total_keywords}")
    print(f"Average Papers/Keyword: {avg_papers:.2f}")
    print(f"Keyword with Most Papers: '{max_k}' ({counts[max_k]} papers)")

if __name__ == "__main__":
    main()

