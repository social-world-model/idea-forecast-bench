import json
from pathlib import Path
from typing import Dict, List

def find_markdown_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.md"))


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Read error: {path} ({e})")
        return ""


def truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(truncated)"


def load_json(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Load error: {path} ({e})")
        return {}


def save_json(path: Path, obj: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def filter_by_arxiv_date(results: Dict[str, List[str]], start_yymm: str, end_yymm: str) -> Dict[str, List[str]]:
    """
    Filter results based on arXiv date prefix in the filename (e.g., '2303').
    """
    filtered = {}
    for file_path, keywords in results.items():
        filename = Path(file_path).name
        # arXiv IDs usually start with YYMM. (e.g., 2303.12345)
        if filename[:4].isdigit():
            date_prefix = filename[:4]
            if start_yymm <= date_prefix <= end_yymm:
                filtered[file_path] = keywords
    return filtered


def group_by_keywords(
    results: Dict[str, List[str]], 
    target_categories: List[str] = None,
    fuzzy_threshold: float = 0.6,
    min_papers: int = 1
) -> Dict[str, List[str]]:
    """
    Invert the mapping from {file: [keywords]} to {keyword: [files]}.
    Supports fuzzy matching against target_categories and length filtering.
    """
    import difflib
    from collections import defaultdict
    
    keyword_map = defaultdict(set)
    targets_lower = [t.lower() for t in target_categories] if target_categories else []
    
    for file_path, keywords in results.items():
        for k in keywords:
            clean_k = k.strip()
            # Skip if empty or too long to be a real keyword (e.g., LLM hallucination)
            if not clean_k or len(clean_k) > 100:
                continue
            
            matched_k = clean_k
            
            if target_categories:
                matches = difflib.get_close_matches(clean_k.lower(), targets_lower, n=1, cutoff=fuzzy_threshold)
                if matches:
                    idx = targets_lower.index(matches[0])
                    matched_k = target_categories[idx]
                else:
                    continue # Skip if not matching any target
            
            # Basic cleanup for key normalization
            norm_k = matched_k.strip().lower() if not target_categories else matched_k
            if norm_k:
                keyword_map[norm_k].add(file_path)
                
    # Filter by min_papers and convert sets to sorted lists
    return {k: sorted(list(v)) for k, v in keyword_map.items() if len(v) >= min_papers}

