import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from live_idea_bench.config import load_runtime_config, load_similarity_config
from live_idea_bench.papers import clean_paper_content, read_file_content
from live_idea_bench.similarity import compute_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Test research-idea similarity against markdown papers.")
    parser.add_argument("--idea", type=str, default="Graph Representation Learning")
    parser.add_argument("--file", type=str, help="Path to a specific markdown file.")
    parser.add_argument("--folder", type=str, help="Path to a folder containing markdown files.")
    parser.add_argument(
        "--engine",
        type=str,
        choices=["hybrid", "embedding", "llm"],
        default="hybrid",
        help="Similarity engine to use.",
    )
    return parser.parse_args()


def print_result(score: float, reasoning: str | None, engine_name: str, file_name: str) -> None:
    border = "=" * 60
    print(f"\n{border}")
    print(f" MATCHING RESULT for: {file_name}")
    print(f"{border}")
    print(f" Engine:    {engine_name}")
    print(f" Score:     {score:.4f}")
    print(f"\n Reasoning:")
    print(f" {reasoning or '-'}")
    print(f"{border}\n")


def main():
    args = parse_args()
    config = load_runtime_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    similarity = load_similarity_config("similarity.yaml")
    similarity.engine = args.engine

    md_dir = PROJECT_ROOT / "md_mineru"
    target_files = []
    if args.folder:
        folder_path = Path(args.folder)
        if not folder_path.is_absolute():
            folder_path = PROJECT_ROOT / args.folder
        if not folder_path.exists():
            print(f"[-] Error: Folder not found at {folder_path}")
            return
        target_files = list(folder_path.glob("**/*.md"))
    elif args.file:
        target_file = Path(args.file)
        if not target_file.is_absolute():
            target_file = md_dir / args.file
        if not target_file.exists():
            print(f"[-] Error: File not found at {target_file}")
            return
        target_files = [target_file]
    else:
        target_files = list(md_dir.glob("**/*.md"))[:1]

    if not target_files:
        print("[-] No markdown files found.")
        return

    print(f"[*] Idea: {args.idea}")
    print(f"[*] Engine: {args.engine}")
    for idx, target_file in enumerate(target_files, start=1):
        print(f"[*] [{idx}/{len(target_files)}] Comparing with: {target_file}")
        content = clean_paper_content(read_file_content(target_file))
        result = compute_similarity(args.idea, content, similarity, config, model_name=config.model_name)
        print_result(result.score, result.reasoning, result.engine_name, target_file.name)


if __name__ == "__main__":
    main()
