from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import openai

from forecaster.config import HindsightConfig
from forecaster.hindsight.extractor import parse_innovation
from forecaster.hindsight.prompt import build_hindsight_prompt
from forecaster.hindsight.topic_sampling import TOPIC_HINDSIGHT_HORIZON_MONTHS
from forecaster.models import Innovation
from idea_forecast_bench.backtest import split_train_future_by_cutoff

logger = logging.getLogger(__name__)

# `Final` (with no explicit annotation) keeps the literal type, which is what
# `openai.resources.Batches.create` requires for its `endpoint` argument.
_BATCH_ENDPOINT: Final = "/v1/chat/completions"
_CUSTOM_ID_MAX_LEN = 64
_CUSTOM_ID_SEP = "|"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchTarget:
    """One extraction target prepared for batch submission."""

    topic_id: str
    episode_id: str
    future_paper_id: str
    cutoff_date: str
    custom_id: str


@dataclass
class BatchState:
    """Persistent state for a batch job, enabling resume on interruption."""

    requests_jsonl_path: str
    input_file_id: str = ""
    batch_id: str = ""
    output_file_id: str = ""
    status: str = "pending"  # pending | uploaded | submitted | completed | failed
    error_message: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "requests_jsonl_path": self.requests_jsonl_path,
            "input_file_id": self.input_file_id,
            "batch_id": self.batch_id,
            "output_file_id": self.output_file_id,
            "status": self.status,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BatchState:
        return cls(
            requests_jsonl_path=str(d.get("requests_jsonl_path", "")),
            input_file_id=str(d.get("input_file_id", "")),
            batch_id=str(d.get("batch_id", "")),
            output_file_id=str(d.get("output_file_id", "")),
            status=str(d.get("status", "pending")),
            error_message=str(d.get("error_message", "")),
        )


# ---------------------------------------------------------------------------
# Custom ID encoding
# ---------------------------------------------------------------------------


def encode_custom_id(topic_id: str, episode_id: str, paper_id: str) -> str:
    """Encode a target triple into an OpenAI Batch API ``custom_id``.

    The encoded string uses ``|`` as a delimiter and is validated to fit within
    the 64-character limit imposed by the Batch API.

    Raises:
        ValueError: If the encoded ID exceeds 64 characters.
    """
    custom_id = f"{topic_id}{_CUSTOM_ID_SEP}{episode_id}{_CUSTOM_ID_SEP}{paper_id}"
    if len(custom_id) > _CUSTOM_ID_MAX_LEN:
        raise ValueError(
            f"Encoded custom_id {custom_id!r} is {len(custom_id)} characters, "
            f"exceeding the OpenAI Batch API limit of {_CUSTOM_ID_MAX_LEN}."
        )
    return custom_id


def decode_custom_id(custom_id: str) -> tuple[str, str, str]:
    """Decode a ``custom_id`` back to ``(topic_id, episode_id, paper_id)``.

    Raises:
        ValueError: If the custom_id does not have exactly two ``|`` separators.
    """
    parts = custom_id.split(_CUSTOM_ID_SEP, 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid custom_id {custom_id!r}: expected format "
            f"'topic_id{_CUSTOM_ID_SEP}episode_id{_CUSTOM_ID_SEP}paper_id'."
        )
    return parts[0], parts[1], parts[2]


# ---------------------------------------------------------------------------
# Request body construction
# ---------------------------------------------------------------------------


def build_batch_request_body(
    model: str,
    system_prompt: str,
    user_message: str,
    config: HindsightConfig,
) -> dict[str, Any]:
    """Build the ``/v1/chat/completions`` request body for a single batch line.

    Mirrors the OpenAI branch of ``get_response_from_llm`` in
    ``idea_forecast_bench/llm.py`` so that batch results are identical to
    synchronous calls.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "n": 1,
        "stop": None,
        "seed": 0,
    }
    if model.startswith("gpt-5"):
        body["max_completion_tokens"] = 4096
    else:
        body["temperature"] = config.temperature
        body["max_tokens"] = 4096
    return body


# ---------------------------------------------------------------------------
# JSONL preparation
# ---------------------------------------------------------------------------


def prepare_batch_jsonl(
    targets: list[dict[str, Any]],
    grouped_papers: dict[str, Any],
    model: str,
    config: HindsightConfig,
    output_path: Path,
    *,
    already_extracted: set[str] | None = None,
) -> list[BatchTarget]:
    """Build the JSONL requests file for OpenAI Batch API submission.

    For each target, resolves the future paper and training context from
    ``grouped_papers``, builds the hindsight prompt, and writes one JSONL line.

    Targets whose ``custom_id`` is in ``already_extracted`` are skipped.

    Args:
        targets: List of target dicts (same format as
            :func:`~forecaster.hindsight.topic_sampling.select_preview_targets`).
        grouped_papers: Mapping from ``topic_id`` to a sequence of
            :class:`~idea_forecast_bench.models.PaperRecord`.
        model: OpenAI model identifier used in the request body.
        config: Hindsight extraction config.
        output_path: Destination path for the ``.jsonl`` file.
        already_extracted: Optional set of ``custom_id`` strings to skip.

    Returns:
        List of :class:`BatchTarget` objects written to the JSONL file.
    """
    already_extracted = already_extracted or set()
    batch_targets: list[BatchTarget] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0

    with output_path.open("w", encoding="utf-8") as fh:
        for target in targets:
            topic_id = target["topic_id"]
            episode_id = target["episode_id"]
            paper_id = str(target["future_paper_id"])
            cutoff_date = target["cutoff_date"]

            custom_id = encode_custom_id(topic_id, episode_id, paper_id)

            if custom_id in already_extracted:
                skipped += 1
                continue

            topic_papers = list(grouped_papers.get(topic_id, ()))
            train_papers, future_papers, _end_month, _end_date = (
                split_train_future_by_cutoff(
                    papers=topic_papers,
                    cutoff_date=cutoff_date,
                    horizon_months=TOPIC_HINDSIGHT_HORIZON_MONTHS,
                )
            )
            future_lookup = {p.paper_id: p for p in future_papers}
            future_paper = future_lookup.get(paper_id)
            if future_paper is None:
                logger.warning(
                    "Future paper %r not found for topic=%r, episode=%r — skipping.",
                    paper_id,
                    topic_id,
                    episode_id,
                )
                continue

            system_prompt, user_message = build_hindsight_prompt(
                future_paper=future_paper,
                context_papers=train_papers,
                max_context_papers=config.max_context_papers,
            )
            body = build_batch_request_body(model, system_prompt, user_message, config)
            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": _BATCH_ENDPOINT,
                "body": body,
            }
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

            batch_targets.append(
                BatchTarget(
                    topic_id=topic_id,
                    episode_id=episode_id,
                    future_paper_id=paper_id,
                    cutoff_date=cutoff_date,
                    custom_id=custom_id,
                )
            )

    logger.info(
        "Wrote %d batch requests to %s (skipped %d already-extracted).",
        len(batch_targets),
        output_path,
        skipped,
    )
    return batch_targets


# ---------------------------------------------------------------------------
# Upload & submission
# ---------------------------------------------------------------------------


def upload_and_submit_batch(
    client: openai.OpenAI,
    jsonl_path: Path,
    state_path: Path,
) -> BatchState:
    """Upload the JSONL requests file and create a batch job.

    Saves :class:`BatchState` to ``state_path`` immediately after each step so
    that polling can resume if the process is interrupted.

    Args:
        client: Authenticated :class:`openai.OpenAI` client.
        jsonl_path: Path to the prepared ``.jsonl`` requests file.
        state_path: Path where ``batch_state.json`` should be written.

    Returns:
        :class:`BatchState` with ``status="submitted"``.
    """
    state = BatchState(requests_jsonl_path=str(jsonl_path))

    logger.info("Uploading batch requests file %s …", jsonl_path)
    with jsonl_path.open("rb") as fh:
        upload = client.files.create(file=fh, purpose="batch")
    state.input_file_id = upload.id
    state.status = "uploaded"
    _save_state(state, state_path)
    logger.info("Uploaded file: %s", upload.id)

    logger.info("Creating batch job …")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint=_BATCH_ENDPOINT,
        completion_window="24h",
    )
    state.batch_id = batch.id
    state.status = "submitted"
    _save_state(state, state_path)
    logger.info("Batch submitted: %s", batch.id)

    return state


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------

_TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})


def poll_batch_until_done(
    client: openai.OpenAI,
    batch_id: str,
    *,
    initial_interval_sec: float = 30.0,
    max_interval_sec: float = 300.0,
    timeout_sec: float = 86400.0,
) -> str:
    """Poll a batch job with exponential back-off until it reaches a terminal state.

    Args:
        client: Authenticated :class:`openai.OpenAI` client.
        batch_id: The batch job ID returned by the Batch API.
        initial_interval_sec: Starting poll interval in seconds.
        max_interval_sec: Maximum poll interval cap in seconds.
        timeout_sec: Total wall-clock timeout before raising.

    Returns:
        The ``output_file_id`` of the completed batch.

    Raises:
        RuntimeError: If the batch fails, expires, is cancelled, or times out.
    """
    deadline = time.monotonic() + timeout_sec
    interval = initial_interval_sec

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        logger.info(
            "Batch %s status=%s  completed=%s/%s  failed=%s",
            batch_id,
            status,
            batch.request_counts.completed if batch.request_counts else "?",
            batch.request_counts.total if batch.request_counts else "?",
            batch.request_counts.failed if batch.request_counts else "?",
        )

        if status == "completed":
            if not batch.output_file_id:
                raise RuntimeError(
                    f"Batch {batch_id} completed but has no output_file_id."
                )
            return batch.output_file_id

        if status in _TERMINAL_STATUSES:
            raise RuntimeError(
                f"Batch {batch_id} ended with status={status!r}. "
                f"errors={batch.errors!r}"
            )

        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Timed out waiting for batch {batch_id} after {timeout_sec:.0f}s. "
                f"Last status: {status!r}"
            )

        time.sleep(min(interval, deadline - time.monotonic()))
        interval = min(interval * 2, max_interval_sec)


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def download_and_parse_results(
    client: openai.OpenAI,
    output_file_id: str,
) -> tuple[dict[str, Innovation], dict[str, str]]:
    """Download batch results and parse innovations from each response.

    Args:
        client: Authenticated :class:`openai.OpenAI` client.
        output_file_id: The ``output_file_id`` from the completed batch.

    Returns:
        A ``(successes, failures)`` pair where:

        - ``successes`` maps ``custom_id`` → :class:`~forecaster.models.Innovation`
        - ``failures`` maps ``custom_id`` → error message string
    """
    raw_content = client.files.content(output_file_id).text
    successes: dict[str, Innovation] = {}
    failures: dict[str, str] = {}

    for line_no, line in enumerate(raw_content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Line %d: could not parse result JSON: %s", line_no, exc)
            continue

        custom_id = result.get("custom_id", f"<unknown-line-{line_no}>")
        response = result.get("response", {})
        status_code = response.get("status_code")

        if status_code != 200:
            error_body = result.get("error") or response.get("body", {})
            failures[custom_id] = f"HTTP {status_code}: {error_body!r}"
            continue

        try:
            content = response["body"]["choices"][0]["message"]["content"] or ""
            innovation = parse_innovation(content)
            successes[custom_id] = innovation
        except (KeyError, IndexError, TypeError) as exc:
            failures[custom_id] = f"Unexpected response structure: {exc!r}"
        except ValueError as exc:
            failures[custom_id] = f"Parse error: {exc}"

    logger.info(
        "Parsed batch results: %d successes, %d failures.",
        len(successes),
        len(failures),
    )
    return successes, failures


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------


def run_batch_extraction(
    client: openai.OpenAI,
    targets: list[dict[str, Any]],
    grouped_papers: dict[str, Any],
    model: str,
    config: HindsightConfig,
    output_dir: Path,
    *,
    max_retry_rounds: int = 1,
    already_extracted: set[str] | None = None,
) -> tuple[dict[str, Innovation], dict[str, str]]:
    """Run end-to-end hindsight extraction via the OpenAI Batch API.

    Handles JSONL preparation, file upload, batch submission, polling, result
    download, and one optional retry round for parse failures.

    If ``output_dir/batch_state.json`` already exists with
    ``status="submitted"``, polling resumes for that batch rather than
    submitting a new one.

    Args:
        client: Authenticated :class:`openai.OpenAI` client.
        targets: List of target dicts from
            :func:`~forecaster.hindsight.topic_sampling.select_preview_targets`
            or :func:`~forecaster.hindsight.topic_sampling.expand_all_targets`.
        grouped_papers: Mapping from ``topic_id`` to paper sequences.
        model: OpenAI model identifier.
        config: Hindsight extraction config.
        output_dir: Directory for intermediate files and state.
        max_retry_rounds: How many additional batch rounds to run for
            parse-failed requests (default 1).
        already_extracted: Optional set of ``custom_id`` strings to skip.

    Returns:
        A ``(all_successes, all_failures)`` pair.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "batch_state.json"
    jsonl_path = output_dir / "batch_requests.jsonl"

    all_successes: dict[str, Innovation] = {}
    all_failures: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Round loop — first round uses the full target list; subsequent
    # rounds only retry the custom_ids that failed to parse.
    # ------------------------------------------------------------------
    pending_targets = targets
    pending_already_extracted = already_extracted

    for round_no in range(1 + max_retry_rounds):
        if not pending_targets:
            logger.info("No targets remaining for round %d — done.", round_no + 1)
            break

        # Resume if a submitted batch state file exists for round 0
        resumed = False
        if round_no == 0 and state_path.exists():
            saved = BatchState.from_dict(
                json.loads(state_path.read_text(encoding="utf-8"))
            )
            if saved.status in ("submitted", "uploaded") and saved.batch_id:
                logger.info(
                    "Resuming batch %s (status=%s).", saved.batch_id, saved.status
                )
                state = saved
                resumed = True

        if not resumed:
            round_jsonl_path = (
                jsonl_path
                if round_no == 0
                else output_dir / f"batch_requests_retry{round_no}.jsonl"
            )
            round_state_path = (
                state_path
                if round_no == 0
                else output_dir / f"batch_state_retry{round_no}.json"
            )
            batch_targets = prepare_batch_jsonl(
                targets=pending_targets,
                grouped_papers=grouped_papers,
                model=model,
                config=config,
                output_path=round_jsonl_path,
                already_extracted=pending_already_extracted,
            )
            if not batch_targets:
                logger.info("Round %d: no new requests to submit.", round_no + 1)
                break
            state = upload_and_submit_batch(client, round_jsonl_path, round_state_path)
            round_state_path = (
                state_path
                if round_no == 0
                else output_dir / f"batch_state_retry{round_no}.json"
            )

        output_file_id = poll_batch_until_done(client, state.batch_id)

        state.output_file_id = output_file_id
        state.status = "completed"
        _save_state(
            state,
            state_path
            if round_no == 0
            else output_dir / f"batch_state_retry{round_no}.json",
        )

        round_successes, round_failures = download_and_parse_results(
            client, output_file_id
        )
        all_successes.update(round_successes)
        all_failures.update(round_failures)

        # Next retry round only re-attempts parse failures
        if round_failures and round_no < max_retry_rounds:
            logger.info(
                "Round %d: %d failures will be retried.",
                round_no + 1,
                len(round_failures),
            )
            # Reconstruct minimal target dicts from failed custom_ids
            target_index = {
                encode_custom_id(
                    t["topic_id"], t["episode_id"], str(t["future_paper_id"])
                ): t
                for t in targets
            }
            pending_targets = [
                target_index[cid] for cid in round_failures if cid in target_index
            ]
            pending_already_extracted = None  # retry batch only contains failures
            # Remove failures that will be retried from the final failures dict
            for cid in list(round_failures):
                if cid in target_index:
                    del all_failures[cid]
        else:
            break

    return all_successes, all_failures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_state(state: BatchState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
