from __future__ import annotations

import json
from pathlib import Path


from backend.services.daily_pipeline import PipelineAlreadyRunningError, run_daily_pipeline  # noqa: E402


def main() -> int:
    try:
        report = run_daily_pipeline()
    except PipelineAlreadyRunningError as exc:
        print(f"Daily pipeline skipped: {exc}")
        return 2
    except Exception as exc:
        print(f"Daily pipeline failed: {exc}")
        return 1

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
