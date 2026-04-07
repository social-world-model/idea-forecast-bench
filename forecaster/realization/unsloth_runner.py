"""Unsloth GRPO runner — single trainer entry point for METHOD §3.3.

Loads the base model with ``unsloth.FastLanguageModel`` (which patches the
underlying TRL trainers via the entries cached in ``unsloth_compiled_cache/``),
attaches a LoRA adapter, optionally warm-starts from the Prior SFT adapter
(``init_policy_path`` from METHOD §3.1's factorized model), and runs
``trl.GRPOTrainer`` with the reward callable from
``forecaster.realization.reward_compute``.

The reward callable returns the three METHOD §3.3 verifiable rewards
(evidence accuracy, operator adherence, scientific coherence) summed by the
weights in ``config/forecaster/reward.yaml`` — TRL handles the GRPO objective
itself (importance ratio + clip + KL).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from forecaster.models import strict_runtime_manifest_contract
from forecaster.realization.config import GRPOTrainConfig
from forecaster.realization.io import _write_json, _write_jsonl
from forecaster.realization.dataset import build_grpo_dataset_rows

logger = logging.getLogger(__name__)


def _build_policy_manifest(
    *,
    base_model_name: str,
    init_policy_path: str | None,
    predictor_config: str,
    trainer_config_path: str,
    output_dir: Path,
    dataset_path: Path,
    dataset_size: int,
    dry_run: bool,
    selection_config_path: str,
    candidate_pool_size: int,
    output_top_k: int,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Manifest shape consumed by ``examples/run_domain_backtest.py`` and
    ``forecaster/orchestrator.py``. Kept byte-compatible with the legacy
    ``forecaster/realization/trainers/base.py:build_policy_manifest`` output
    so downstream eval scripts work without modification.
    """
    final_checkpoint = output_dir / "checkpoints" / "final_checkpoint"
    payload = {
        "policy_manifest_version": 1,
        "policy_type": "policy_rl",
        "backend": "unsloth",
        "trainer": "grpo",
        "base_model_name": base_model_name,
        "inference_model_name": base_model_name,
        "init_policy_path": init_policy_path or "",
        "predictor_config": predictor_config,
        "output_dir": str(output_dir.resolve()),
        "checkpoint_path": str(final_checkpoint.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_size": dataset_size,
        "trainer_config_path": str(trainer_config_path),
        "reward_mode": "single_idea",
        "selection_config_path": str(selection_config_path),
        "candidate_pool_size": int(candidate_pool_size),
        "output_top_k": int(output_top_k),
        "training_split_policy": "train_only",
        "dry_run": dry_run,
        "strict_contract": strict_runtime_manifest_contract(),
    }
    if diagnostics:
        payload["diagnostics"] = diagnostics
        payload["parse_failure_rate"] = float(diagnostics.get("parse_failure_rate", 0.0))
        payload["invalid_completion_rate"] = float(diagnostics.get("invalid_completion_rate", 0.0))
    return payload


def prepare_grpo_dataset(
    prompt_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], Path]:
    """Build the GRPO trainer dataset rows + write the JSONL preview to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_rows = build_grpo_dataset_rows(
        prompt_rows, data_source="live_idea_bench::grpo"
    )
    dataset_path = output_dir / "trainer_dataset.jsonl"
    _write_jsonl(dataset_path, dataset_rows)
    return dataset_rows, dataset_path


def train_grpo_with_unsloth(
    *,
    model_name: str,
    dataset_rows: list[dict[str, Any]],
    output_dir: Path | str,
    config: GRPOTrainConfig,
    init_policy_path: str | None = None,
    predictor_config: str = "predictor.yaml",
    trainer_config_path: str = "grpo_train.yaml",
    selection_config_path: str = "selection.yaml",
    selection_candidate_pool_size: int = 24,
    selection_output_top_k: int = 5,
    reward_config_path: str = "reward.yaml",
    realization_config_path: str = "realization.yaml",
    similarity_config_path: str = "similarity.yaml",
    runtime_config_path: str | None = None,
    dataset_path: Path | str | None = None,
    diagnostics: dict[str, Any] | None = None,
    use_vllm_server: bool = False,
    vllm_server_host: str = "localhost",
    vllm_server_port: int = 8765,
) -> dict[str, Any]:
    """Run Unsloth-backed GRPO training and return the policy manifest dict.

    Imports of ``unsloth`` / ``trl`` happen lazily so the module can be
    inspected (e.g. by import smoke tests) on machines without the heavy
    runtime stack installed.
    """
    # Lazy heavy imports — Unsloth must be imported BEFORE transformers/trl
    # so its monkey-patches take effect.
    from unsloth import FastLanguageModel  # noqa: E402
    import torch  # noqa: E402
    from datasets import Dataset  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402
    from trl import GRPOConfig, GRPOTrainer  # noqa: E402

    from forecaster.realization.reward_compute import compute_score

    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint_dir = target_dir / "checkpoints" / "final_checkpoint"

    if not dataset_rows:
        raise ValueError("train_grpo_with_unsloth: dataset_rows is empty.")

    # ----- Load model + tokenizer with Unsloth -----
    max_seq_length = int(config.max_prompt_length + config.max_completion_length)
    logger.info(
        "Loading %s with Unsloth (max_seq_length=%d, lora_r=%d)...",
        model_name,
        max_seq_length,
        config.lora_r,
    )
    model, _unsloth_processor = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
        # Unsloth's official guidance for Qwen3.5 GRPO:
        # https://unsloth.ai/docs/models/qwen3.5/fine-tune
        # Quote: "If you'd like to do GRPO, it works in Unsloth if you disable
        # fast vLLM inference and use Unsloth inference instead."
        # vLLM colocate's compile pipeline crashes inside _decompose_size_nodes
        # for Qwen3.5 + torch 2.10. Both stable (0.19.0) and nightly fail.
        fast_inference=False,
    )
    # Qwen3.5 is registered as a VLM (Qwen2VLForConditionalGeneration); Unsloth
    # returns the multi-modal processor whose image_processor would route plain
    # text completions through the image preprocessing path and crash inside
    # GRPOTrainer. Force a text-only AutoTokenizer for both `processing_class`
    # and the prompt rendering pass below.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if init_policy_path:
        # Warm-start from the Prior SFT LoRA adapter (METHOD §3.1
        # factorized model). Unsloth's PEFT-attached model accepts
        # ``load_adapter`` directly.
        try:
            model.load_adapter(init_policy_path, adapter_name="default")
            logger.info("Warm-started from prior SFT adapter: %s", init_policy_path)
        except Exception as exc:
            logger.warning(
                "Could not load init_policy_path=%s as adapter (%s); training from base.",
                init_policy_path, exc,
            )

    # ----- Build dataset (chat template + thinking-mode rendering) -----
    # Pre-render with enable_thinking=True so GRPO rollouts emit
    # "<think>...</think>title\nbody" — the DeepSeek-R1-Zero recipe GRPO was
    # designed for. Passing the rendered string column prevents TRL from
    # re-templating with default kwargs.
    ds_records: list[dict[str, Any]] = []
    for row in dataset_rows:
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            # Tokenizer doesn't support enable_thinking — fall back to plain.
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        ds_records.append({
            "prompt": rendered,
            "extra_info": row.get("extra_info", "{}"),
        })
    dataset = Dataset.from_list(ds_records)

    # ----- Reward callable (METHOD §3.3 verifiable rewards) -----
    def reward_fn(completions: list[str], **kwargs: Any) -> list[float]:
        extra_infos = kwargs.get("extra_info", ["{}"] * len(completions))
        scores: list[float] = []
        for completion, extra_info in zip(completions, extra_infos):
            score = compute_score(
                data_source="live_idea_bench::grpo",
                solution_str=completion,
                ground_truth="",
                extra_info=extra_info,
                reward_config_path=reward_config_path,
                realization_config_path=realization_config_path,
                similarity_config_path=similarity_config_path,
                runtime_config_path=runtime_config_path,
                model_name=model_name,
            )
            scores.append(float(score))
        return scores

    # ----- GRPO config -----
    num_gen = int(config.num_generations)
    grpo_config = GRPOConfig(
        output_dir=str(target_dir / "checkpoints"),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=num_gen,  # one full group per step
        gradient_accumulation_steps=max(1, int(config.gradient_accumulation_steps)),
        learning_rate=config.learning_rate,
        num_generations=num_gen,
        # Note: TRL 1.0.0 GRPOConfig dropped max_prompt_length. The prompt length
        # is now implicit from the tokenizer/model max position. We still pass
        # config.max_prompt_length to FastLanguageModel.from_pretrained above
        # via max_seq_length = max_prompt_length + max_completion_length.
        max_completion_length=int(config.max_completion_length),
        # Qwen3.5 thinking-mode anti-loop sampling (model card recommendation).
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.5,
        beta=float(config.kl_coef),  # KL-to-reference per METHOD §3.3
        logging_steps=int(config.logging_steps),
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="none",
        # 8-bit AdamW: ~50% optimizer-state memory savings vs fp32 state with
        # well-validated negligible quality impact (Dettmers et al. + Unsloth
        # default). Frees VRAM headroom for larger G or seq lengths.
        optim="adamw_8bit",
        # Use available CPUs to prefetch the next batch while the GPU computes.
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        # Safety net: if our per_device_train_batch_size = num_generations
        # estimate OOMs, HF/accelerate will catch the error and retry with a
        # smaller batch. For GRPO this still has to stay a multiple of
        # num_generations, so it's a guard rail rather than a true optimizer.
        auto_find_batch_size=True,
        # vLLM acceleration paths for Qwen3.5 GRPO — what works and what doesn't
        # (verified on this stack 2026-04-07: torch 2.10 + transformers 5.5 +
        # vLLM 0.19.1 nightly + Unsloth 2026.4.4):
        #
        # 1. vLLM colocate (use_vllm=True, vllm_mode="colocate"):
        #    BROKEN — crashes inside vllm/compilation/backends.py
        #    :_decompose_size_nodes when compiled against torch 2.10 because
        #    Unsloth's PEFT/LoRA wrapping inserts FX graph `size` nodes the
        #    rewriter can't handle. Both stable and nightly fail.
        #
        # 2. vLLM server mode on the SAME GPU:
        #    BROKEN — TRL's trl/scripts/vllm_serve.py:87:init_communicator
        #    raises RuntimeError("...same CUDA device for multiple distinct
        #    roles/ranks ... unsupported and will likely lead to program
        #    hangs or incorrect behavior. Ensure that trainer is using
        #    different devices than vLLM server.")
        #
        # 3. vLLM server mode on a DIFFERENT GPU (multi-GPU box):
        #    WORKS — but requires 2+ GPUs. The wrapper script supports it
        #    when --use-vllm-server is set; the operator must arrange the
        #    GPU placement (e.g., CUDA_VISIBLE_DEVICES split).
        #
        # 4. fast_inference=False on FastLanguageModel.from_pretrained
        #    (above) + use_vllm=False here:
        #    WORKS — Unsloth's documented Qwen3.5 GRPO path. Slower than
        #    vLLM (~213s/step on 2B) but stable on a single GPU.
        #
        # When use_vllm_server=False (single-GPU default), we go with path 4.
        # When use_vllm_server=True, we go with path 3 (server mode against
        # a separately-launched trl vllm-serve on a different GPU).
        use_vllm=use_vllm_server,
        vllm_mode="server" if use_vllm_server else "colocate",
        vllm_server_host=vllm_server_host,
        vllm_server_port=vllm_server_port,
        vllm_server_timeout=240.0,
        # Skip the loss contribution of completions that hit max_completion_length
        # without naturally terminating. They carry no useful gradient signal
        # (the model never reached an EOS) and dominate wall time on
        # thinking-mode rollouts that loop. Reduces effective batch but
        # improves both speed and signal-to-noise.
        mask_truncated_completions=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info(
        "Starting GRPO training with Unsloth (%d examples, %d epochs, G=%d)...",
        len(dataset), config.num_train_epochs, num_gen,
    )
    trainer.train()

    final_checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_checkpoint_dir))
    logger.info("Saved final checkpoint to %s", final_checkpoint_dir)

    resolved_dataset_path = (
        Path(dataset_path).resolve()
        if dataset_path is not None
        else (target_dir / "trainer_dataset.jsonl")
    )
    manifest = _build_policy_manifest(
        base_model_name=model_name,
        init_policy_path=init_policy_path,
        predictor_config=predictor_config,
        trainer_config_path=trainer_config_path,
        output_dir=target_dir,
        dataset_path=resolved_dataset_path,
        dataset_size=len(dataset_rows),
        dry_run=bool(config.dry_run),
        selection_config_path=selection_config_path,
        candidate_pool_size=selection_candidate_pool_size,
        output_top_k=selection_output_top_k,
        diagnostics=diagnostics,
    )
    _write_json(target_dir / "policy_manifest.json", manifest)
    return manifest
