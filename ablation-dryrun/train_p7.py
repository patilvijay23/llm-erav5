#!/usr/bin/env python3
"""Train matched P7 hard-shift or centred linear-transition ablations.

Both arms process 20B packed training tokens and consume exactly 13B FineWeb-Edu
and 7B IndicCorpV2 tokens. The only intended variable is the transition shape:

* hard: 50:50 before 10B, then an immediate 80:20 shift;
* linear: a centred linear blend across ``--transition-tokens``.

Each source manifest must contain 10B *trainable* tokens. Evaluation sequences are
prepared in addition to those 10B tokens, so FineWeb replay is exactly 3B tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

from p7.data import MemmapTokenPool, validate_compatible_pools
from p7.model import build_proxy_model, parameter_report
from p7.schedule import (
    FINEWEB,
    INDICCORP,
    MixtureSchedule,
    ScheduleConfig,
    build_schedule,
    affine_permutation_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fineweb-manifest", required=True)
    parser.add_argument("--indic-manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--output-dir", default="runs/p7-hard-shift")
    parser.add_argument(
        "--transition-mode",
        choices=["hard", "linear"],
        default="hard",
        help="P7 hard shift or matched centred linear-transition control.",
    )
    parser.add_argument(
        "--transition-tokens",
        type=int,
        default=256_000_000,
        help=(
            "Width of the centred linear transition. Ignored in hard mode. "
            "256M aligns to 2,000 optimizer steps in the recommended setup."
        ),
    )

    parser.add_argument("--total-tokens", type=int, default=20_000_000_000)
    parser.add_argument("--shift-tokens", type=int, default=10_000_000_000)
    parser.add_argument("--block-size", type=int, default=640)
    parser.add_argument(
        "--expected-source-train-tokens",
        type=int,
        default=10_000_000_000,
        help="Require this many trainable tokens in each source manifest; 0 disables.",
    )
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=25)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-fraction", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data-seed", type=int, default=5678)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--checkpoint-every-steps", type=int, default=5000)
    parser.add_argument("--eval-every-steps", type=int, default=5000)
    parser.add_argument("--eval-sequences-per-source", type=int, default=512)
    parser.add_argument("--keep-last-checkpoints", type=int, default=3)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--allow-model-size-mismatch", action="store_true")
    parser.add_argument("--dry-run-steps", type=int, default=0)
    return parser.parse_args()


def cosine_with_floor(
    *, total_steps: int, warmup_steps: int, min_ratio: float
):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return lr_lambda


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs: dict[str, Any] = dict(
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )
    if torch.cuda.is_available():
        kwargs["fused"] = True
    try:
        return AdamW(groups, **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("fused", None)
        return AdamW(groups, **kwargs)


def read_latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(
        output_dir.glob("checkpoint-step-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def prune_checkpoints(output_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints = sorted(
        output_dir.glob("checkpoint-step-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    for old in checkpoints[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


def save_checkpoint(
    *,
    accelerator: Accelerator,
    output_dir: Path,
    completed_steps: int,
    completed_tokens: int,
    keep: int,
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-step-{completed_steps:08d}"
    accelerator.wait_for_everyone()
    if checkpoint_dir.exists():
        accelerator.wait_for_everyone()
        return
    accelerator.save_state(checkpoint_dir)
    if accelerator.is_main_process:
        state = {
            "completed_steps": completed_steps,
            "completed_tokens": completed_tokens,
            "saved_at_unix": time.time(),
        }
        (checkpoint_dir / "trainer_state.json").write_text(
            json.dumps(state, indent=2) + "\n"
        )
        prune_checkpoints(output_dir, keep)
    accelerator.wait_for_everyone()


def source_batch(
    *,
    schedule: MixtureSchedule,
    step: int,
    micro_step: int,
    accelerator: Accelerator,
    args: argparse.Namespace,
    fineweb_pool: MemmapTokenPool,
    indic_pool: MemmapTokenPool,
) -> tuple[torch.Tensor, int, int, int]:
    draws = schedule.local_draws(
        step=step,
        micro_step=micro_step,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    rows: list[torch.Tensor] = []
    fw_count = 0
    indic_count = 0
    max_pass = 0
    for source, ordinal in draws:
        if source == FINEWEB:
            index, pass_number = affine_permutation_index(
                ordinal, fineweb_pool.train_sequences, args.data_seed ^ 0xF1E
            )
            rows.append(fineweb_pool.get_train(index))
            fw_count += 1
        elif source == INDICCORP:
            index, pass_number = affine_permutation_index(
                ordinal, indic_pool.train_sequences, args.data_seed ^ 0x1D1C
            )
            rows.append(indic_pool.get_train(index))
            indic_count += 1
        else:
            raise RuntimeError(f"Unexpected source id {source}")
        max_pass = max(max_pass, pass_number)
    batch = torch.stack(rows, dim=0).to(accelerator.device, non_blocking=True)
    return batch, fw_count, indic_count, max_pass


@torch.no_grad()
def evaluate_source(
    *,
    model: torch.nn.Module,
    accelerator: Accelerator,
    pool: MemmapTokenPool,
    max_sequences: int,
    per_device_batch_size: int,
) -> float:
    if pool.eval_sequences == 0 or max_sequences == 0:
        return float("nan")
    model.eval()
    requested_limit = min(pool.eval_sequences, max_sequences)
    global_eval_batch = accelerator.num_processes * per_device_batch_size
    limit = (requested_limit // global_eval_batch) * global_eval_batch
    if limit == 0:
        model.train()
        return float("nan")
    local_indices = list(range(accelerator.process_index, limit, accelerator.num_processes))
    loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    token_count = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    for start in range(0, len(local_indices), per_device_batch_size):
        indices = local_indices[start : start + per_device_batch_size]
        if not indices:
            continue
        input_ids = torch.stack([pool.get_eval(index) for index in indices]).to(
            accelerator.device, non_blocking=True
        )
        with accelerator.autocast():
            output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        predicted = input_ids.numel() - input_ids.shape[0]
        loss_sum += output.loss.detach().double() * predicted
        token_count += predicted
    loss_sum = accelerator.reduce(loss_sum, reduction="sum")
    token_count = accelerator.reduce(token_count, reduction="sum")
    model.train()
    return float((loss_sum / token_count.clamp_min(1)).item())


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=str(output_dir / "tensorboard"),
    )
    set_seed(args.seed, device_specific=False)

    fineweb_pool = MemmapTokenPool(args.fineweb_manifest)
    indic_pool = MemmapTokenPool(args.indic_manifest)
    validate_compatible_pools(
        fineweb_pool, indic_pool, required_block_size=args.block_size
    )
    if args.expected_source_train_tokens > 0:
        for label, pool in (("FineWeb-Edu", fineweb_pool), ("IndicCorpV2", indic_pool)):
            actual = pool.train_sequences * pool.block_size
            if actual != args.expected_source_train_tokens:
                raise ValueError(
                    f"{label} manifest has {actual:,} trainable tokens; expected "
                    f"{args.expected_source_train_tokens:,}. Prepare evaluation data "
                    "in addition to the requested training pool."
                )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        use_fast=True,
        trust_remote_code=False,
    )
    if len(tokenizer) != fineweb_pool.manifest.tokenizer_vocab_size:
        raise ValueError("Runtime tokenizer vocabulary differs from prepared pools")
    from p7.data import tokenizer_fingerprint

    if tokenizer_fingerprint(tokenizer) != fineweb_pool.manifest.tokenizer_fingerprint:
        raise ValueError("Runtime tokenizer fingerprint differs from prepared pools")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    global_batch_sequences = (
        accelerator.num_processes
        * args.per_device_batch_size
        * args.gradient_accumulation_steps
    )
    schedule = build_schedule(
        mode=args.transition_mode,
        config=ScheduleConfig(
            total_tokens=args.total_tokens,
            shift_tokens=args.shift_tokens,
            block_size=args.block_size,
            seed=args.seed,
        ),
        global_batch_sequences=global_batch_sequences,
        transition_tokens=args.transition_tokens,
    )
    if args.dry_run_steps > 0:
        total_steps = min(schedule.total_steps, args.dry_run_steps)
    else:
        total_steps = schedule.total_steps

    model = build_proxy_model(
        vocab_size=len(tokenizer),
        block_size=args.block_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=int(tokenizer.eos_token_id),
        pad_token_id=int(tokenizer.pad_token_id),
    )
    report = parameter_report(model)
    if not args.allow_model_size_mismatch and not (
        0.95 <= report["parameters_billions"] <= 1.05
    ):
        raise ValueError(
            f"Proxy has {report['parameters_billions']:.3f}B parameters, outside "
            "the allowed 0.95-1.05B range. Check the tokenizer vocabulary."
        )

    optimizer = build_optimizer(model, args)
    warmup_steps = int(schedule.total_steps * args.warmup_fraction)
    scheduler = LambdaLR(
        optimizer,
        cosine_with_floor(
            total_steps=schedule.total_steps,
            warmup_steps=warmup_steps,
            min_ratio=args.min_learning_rate / args.learning_rate,
        ),
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    accelerator.init_trackers(
        f"p7-{args.transition_mode}-transition",
        config={
            **vars(args),
            **report,
            "world_size": accelerator.num_processes,
            "global_batch_sequences": global_batch_sequences,
            "global_batch_tokens": global_batch_sequences * args.block_size,
            "shift_step": schedule.shift_step,
            "transition_mode": schedule.mode,
            "transition_start_step": schedule.transition_start_step,
            "transition_end_step": schedule.transition_end_step,
            "scheduled_total_steps": schedule.total_steps,
            **schedule.expected_totals(),
        },
    )

    start_step = 0
    resume_path: Path | None = None
    if args.resume_from == "latest":
        resume_path = read_latest_checkpoint(output_dir)
    elif args.resume_from:
        resume_path = Path(args.resume_from)
    if resume_path is not None:
        accelerator.load_state(resume_path)
        state = json.loads((resume_path / "trainer_state.json").read_text())
        start_step = int(state["completed_steps"])

    if accelerator.is_main_process:
        run_config = {
            **vars(args),
            **report,
            "world_size": accelerator.num_processes,
            "global_batch_sequences": global_batch_sequences,
            "global_batch_tokens": global_batch_sequences * args.block_size,
            "shift_step": schedule.shift_step,
            "transition_mode": schedule.mode,
            "transition_start_step": schedule.transition_start_step,
            "transition_end_step": schedule.transition_end_step,
            "transition_steps": schedule.transition_steps,
            "total_steps": schedule.total_steps,
            "expected_source_totals": schedule.expected_totals(),
            "fineweb_manifest": json.loads(Path(args.fineweb_manifest).read_text()),
            "indic_manifest": json.loads(Path(args.indic_manifest).read_text()),
        }
        (output_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n"
        )
        expected = schedule.expected_totals()
        print(json.dumps(expected, indent=2))
        if schedule.mode == "hard":
            transition_text = f"hard shift at step {schedule.shift_step:,}"
        else:
            transition_text = (
                f"linear transition steps {schedule.transition_start_step:,}-"
                f"{schedule.transition_end_step:,}, centred at {schedule.shift_step:,}"
            )
        print(
            f"P7 {schedule.mode}: {schedule.total_steps:,} steps; {transition_text}; "
            f"global batch {global_batch_sequences} sequences "
            f"({global_batch_sequences * args.block_size:,} tokens)."
        )

    metrics_path = output_dir / "metrics.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    wall_start = time.time()

    boundary_steps = {
        schedule.shift_step,
        schedule.transition_start_step,
        schedule.transition_end_step,
    }
    special_eval_steps = {
        candidate
        for boundary in boundary_steps
        for candidate in (boundary - 1, boundary, boundary + 1)
        if 0 <= candidate < schedule.total_steps
    }

    for step in range(start_step, total_steps):
        phase = schedule.phase_for_step(step)
        step_loss = 0.0
        step_fw_sequences, step_indic_sequences = schedule.counts_for_step(step)
        step_start = time.time()

        for micro_step in range(args.gradient_accumulation_steps):
            input_ids, fw_count, indic_count, source_pass = source_batch(
                schedule=schedule,
                step=step,
                micro_step=micro_step,
                accelerator=accelerator,
                args=args,
                fineweb_pool=fineweb_pool,
                indic_pool=indic_pool,
            )
            # Counts are fixed globally by the schedule. Local counts can differ
            # across ranks because the global source pattern is shuffled.

            sync_context = (
                nullcontext()
                if micro_step == args.gradient_accumulation_steps - 1
                else accelerator.no_sync(model)
            )
            with sync_context:
                with accelerator.autocast():
                    output = model(
                        input_ids=input_ids,
                        labels=input_ids,
                        use_cache=False,
                    )
                    loss = output.loss
                accelerator.backward(loss / args.gradient_accumulation_steps)
            step_loss += float(loss.detach().item())

        grad_norm = accelerator.clip_grad_norm_(
            model.parameters(), args.max_grad_norm
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        completed_steps = step + 1
        completed_tokens = (
            completed_steps * global_batch_sequences * args.block_size
        )
        mean_loss = step_loss / args.gradient_accumulation_steps
        elapsed = max(time.time() - step_start, 1e-9)
        tokens_per_second = (
            global_batch_sequences * args.block_size / elapsed
        )
        fw_draws_done, indic_draws_done = schedule.draw_bases(completed_steps)
        max_source_pass = max(
            (max(fw_draws_done - 1, 0) // fineweb_pool.train_sequences),
            (max(indic_draws_done - 1, 0) // indic_pool.train_sequences),
        )
        record = {
            "step": completed_steps,
            "tokens": completed_tokens,
            "phase": phase.name,
            "transition_mode": schedule.mode,
            "fineweb_share": step_fw_sequences / global_batch_sequences,
            "indic_share": step_indic_sequences / global_batch_sequences,
            "fineweb_step_tokens": step_fw_sequences * args.block_size,
            "indic_step_tokens": step_indic_sequences * args.block_size,
            "loss": mean_loss,
            "grad_norm_pre_clip": float(grad_norm),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "tokens_per_second": tokens_per_second,
            "max_source_pass_seen": max_source_pass,
            "wall_seconds": time.time() - wall_start,
        }

        should_log = (
            completed_steps % args.log_every_steps == 0
            or step in special_eval_steps
            or completed_steps == total_steps
        )
        if should_log:
            reduced_loss = accelerator.reduce(
                torch.tensor(mean_loss, device=accelerator.device), reduction="mean"
            )
            record["loss"] = float(reduced_loss.item())
            if accelerator.is_main_process:
                append_jsonl(metrics_path, record)
                print(json.dumps(record, sort_keys=True))
            accelerator.log(
                {
                    key: value
                    for key, value in record.items()
                    if isinstance(value, (int, float))
                },
                step=completed_steps,
            )

        should_eval = (
            args.eval_every_steps > 0
            and completed_steps % args.eval_every_steps == 0
        ) or step in special_eval_steps
        if should_eval:
            fw_eval = evaluate_source(
                model=model,
                accelerator=accelerator,
                pool=fineweb_pool,
                max_sequences=args.eval_sequences_per_source,
                per_device_batch_size=args.per_device_batch_size,
            )
            indic_eval = evaluate_source(
                model=model,
                accelerator=accelerator,
                pool=indic_pool,
                max_sequences=args.eval_sequences_per_source,
                per_device_batch_size=args.per_device_batch_size,
            )
            eval_record = {
                "step": completed_steps,
                "tokens": completed_tokens,
                "event": "evaluation",
                "phase": phase.name,
                "transition_mode": schedule.mode,
                "fineweb_eval_loss": fw_eval,
                "fineweb_eval_perplexity": math.exp(min(fw_eval, 20.0)),
                "indic_eval_loss": indic_eval,
                "indic_eval_perplexity": math.exp(min(indic_eval, 20.0)),
            }
            if accelerator.is_main_process:
                append_jsonl(metrics_path, eval_record)
                print(json.dumps(eval_record, sort_keys=True))
            accelerator.log(
                {
                    key: value
                    for key, value in eval_record.items()
                    if isinstance(value, (int, float))
                },
                step=completed_steps,
            )

        if (
            args.checkpoint_every_steps > 0
            and completed_steps % args.checkpoint_every_steps == 0
        ):
            save_checkpoint(
                accelerator=accelerator,
                output_dir=output_dir,
                completed_steps=completed_steps,
                completed_tokens=completed_tokens,
                keep=args.keep_last_checkpoints,
            )

    save_checkpoint(
        accelerator=accelerator,
        output_dir=output_dir,
        completed_steps=total_steps,
        completed_tokens=total_steps * global_batch_sequences * args.block_size,
        keep=args.keep_last_checkpoints,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
