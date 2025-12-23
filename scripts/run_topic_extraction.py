import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import openai

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils import find_markdown_files, read_text
from src.prompting import extract_keywords, extract_abstract

MD_DIR = Path("/mnt/disk1_from_server2/haofeiy2/live-idea-bench/md_mineru")
OUT_FILE = project_root / "keywords_results.json"
CONCURRENCY_LIMIT = 20


def load_results(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tf:
        json.dump(obj, tf, indent=2, ensure_ascii=False)
        tmp_name = tf.name
    Path(tmp_name).replace(path)


async def process_file(
    idx: int,
    total: int,
    file_path: Path,
    client: openai.AsyncOpenAI,
    results: dict,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        full_content = read_text(file_path)
        if not full_content:
            return

        intro_content = extract_abstract(full_content)

        if not intro_content:
            return

        keywords = await extract_keywords(intro_content, client, model="gpt-4o-mini")
        if not keywords:
            return

        results[str(file_path)] = keywords
        atomic_write_json(OUT_FILE, results)
        print(f"[{idx}/{total}] {file_path} -> {keywords}")


async def main() -> None:
    if not MD_DIR.exists():
        print(f"Directory not found: {MD_DIR}")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set. Example:")
        print("  export OPENAI_API_KEY='your-key'")
        return

    client = openai.AsyncOpenAI(api_key=api_key)

    md_files = find_markdown_files(MD_DIR)
    print(f"Found {len(md_files)} markdown files under {MD_DIR}")

    results = load_results(OUT_FILE)
    print(f"Loaded {len(results)} existing results from {OUT_FILE}")

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []

    for idx, file_path in enumerate(md_files, start=1):
        if str(file_path) in results:
            continue
        tasks.append(process_file(idx, len(md_files), file_path, client, results, semaphore))

    if tasks:
        await asyncio.gather(*tasks)

    print(f"Done. Saved to {OUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
