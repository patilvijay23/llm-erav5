#!/usr/bin/env python3
"""Summarize one P7 transition run and optionally compare hard vs linear arms."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--control-run-dir",
        default=None,
        help="Optional matched control run. Usually the linear arm when run-dir is hard.",
    )
    parser.add_argument(
        "--window-steps",
        type=int,
        default=500,
        help="Logged optimizer-step window before/after the transition.",
    )
    parser.add_argument(
        "--material-reduction",
        type=float,
        default=0.10,
        help="Minimum relative peak reduction used for the warm-control decision.",
    )
    return parser.parse_args()


def safe_median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator and all(math.isfinite(v) for v in (numerator, denominator)):
        return numerator / denominator
    return float("nan")


def load_records(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads((run_dir / "run_config.json").read_text())
    records: list[dict[str, Any]] = []
    with (run_dir / "metrics.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if "grad_norm_pre_clip" in record and "loss" in record:
                records.append(record)
    if not records:
        raise ValueError(f"No training metric records found in {run_dir}")
    return config, records


def summarize_run(run_dir: Path, window_steps: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config, records = load_records(run_dir)
    start = int(config.get("transition_start_step", config["shift_step"]))
    end = int(config.get("transition_end_step", config["shift_step"]))
    centre = int(config["shift_step"])

    # Metric records use completed optimizer steps. `start` is therefore the
    # last fully pre-transition completed step; the transition begins at start+1.
    before = [r for r in records if start - window_steps < int(r["step"]) <= start]
    response = [
        r
        for r in records
        if start < int(r["step"]) <= min(end + window_steps, int(config["total_steps"]))
    ]
    baseline_grad = safe_median([float(r["grad_norm_pre_clip"]) for r in before])
    baseline_loss = safe_median([float(r["loss"]) for r in before])
    peak_grad = max([float(r["grad_norm_pre_clip"]) for r in response], default=float("nan"))
    peak_loss = max([float(r["loss"]) for r in response], default=float("nan"))
    grad_multiplier = safe_ratio(peak_grad, baseline_grad)
    loss_multiplier = safe_ratio(peak_loss, baseline_loss)

    recovery_step: int | None = None
    for record in sorted(response, key=lambda row: int(row["step"])):
        if int(record["step"]) < end:
            continue
        if (
            float(record["grad_norm_pre_clip"]) <= 2.5 * baseline_grad
            and float(record["loss"]) <= 1.15 * baseline_loss
        ):
            recovery_step = int(record["step"])
            break
    recovery_steps = None if recovery_step is None else max(0, recovery_step - end)

    finite = all(
        math.isfinite(value)
        for value in (baseline_grad, peak_grad, baseline_loss, peak_loss)
    )
    summary = {
        "run_dir": str(run_dir),
        "transition_mode": config.get("transition_mode", "hard"),
        "shift_step": centre,
        "transition_start_step": start,
        "transition_end_step": end,
        "window_steps": window_steps,
        "pre_transition_grad_norm_median": baseline_grad,
        "transition_response_grad_norm_peak": peak_grad,
        "gradient_spike_multiplier": grad_multiplier,
        "pre_transition_loss_median": baseline_loss,
        "transition_response_loss_peak": peak_loss,
        "loss_spike_multiplier": loss_multiplier,
        "recovery_steps_after_transition": recovery_steps,
        "passes_stability_gate": bool(
            finite
            and grad_multiplier <= 2.5
            and loss_multiplier <= 1.15
            and recovery_steps is not None
            and recovery_steps <= 500
        ),
    }
    return summary, records


def write_window(
    *,
    run_dir: Path,
    records: list[dict[str, Any]],
    start: int,
    end: int,
    window_steps: int,
) -> None:
    fieldnames = [
        "step",
        "tokens",
        "phase",
        "transition_mode",
        "fineweb_share",
        "indic_share",
        "loss",
        "grad_norm_pre_clip",
        "learning_rate",
        "tokens_per_second",
    ]
    with (run_dir / "p7_transition_window.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if start - window_steps < int(record["step"]) <= end + window_steps:
                writer.writerow({key: record.get(key) for key in fieldnames})


def matched_configs(hard: dict[str, Any], linear: dict[str, Any]) -> tuple[bool, list[str]]:
    keys = [
        "total_tokens",
        "shift_tokens",
        "block_size",
        "seed",
        "data_seed",
        "learning_rate",
        "min_learning_rate",
        "warmup_fraction",
        "weight_decay",
        "global_batch_sequences",
        "expected_source_totals",
        "fineweb_manifest",
        "indic_manifest",
    ]
    mismatches = [key for key in keys if hard.get(key) != linear.get(key)]
    return not mismatches, mismatches


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary, records = summarize_run(run_dir, args.window_steps)
    write_window(
        run_dir=run_dir,
        records=records,
        start=int(summary["transition_start_step"]),
        end=int(summary["transition_end_step"]),
        window_steps=args.window_steps,
    )
    (run_dir / "p7_transition_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    output: dict[str, Any] = {"primary": summary}
    if args.control_run_dir:
        control_dir = Path(args.control_run_dir)
        control, control_records = summarize_run(control_dir, args.window_steps)
        write_window(
            run_dir=control_dir,
            records=control_records,
            start=int(control["transition_start_step"]),
            end=int(control["transition_end_step"]),
            window_steps=args.window_steps,
        )
        (control_dir / "p7_transition_summary.json").write_text(
            json.dumps(control, indent=2, sort_keys=True) + "\n"
        )

        primary_config = json.loads((run_dir / "run_config.json").read_text())
        control_config = json.loads((control_dir / "run_config.json").read_text())
        is_matched, mismatches = matched_configs(primary_config, control_config)
        hard = summary if summary["transition_mode"] == "hard" else control
        linear = control if control["transition_mode"] == "linear" else summary
        if {summary["transition_mode"], control["transition_mode"]} != {"hard", "linear"}:
            raise ValueError("Comparison requires one hard arm and one linear arm")

        grad_reduction = 1.0 - safe_ratio(
            float(linear["gradient_spike_multiplier"]),
            float(hard["gradient_spike_multiplier"]),
        )
        loss_reduction = 1.0 - safe_ratio(
            float(linear["loss_spike_multiplier"]),
            float(hard["loss_spike_multiplier"]),
        )
        warm_preferred = bool(
            is_matched
            and (
                (not hard["passes_stability_gate"] and linear["passes_stability_gate"])
                or grad_reduction >= args.material_reduction
                or loss_reduction >= args.material_reduction
            )
        )
        comparison = {
            "hard": hard,
            "linear": linear,
            "matched_configuration": is_matched,
            "configuration_mismatches": mismatches,
            "linear_gradient_peak_reduction": grad_reduction,
            "linear_loss_peak_reduction": loss_reduction,
            "material_reduction_threshold": args.material_reduction,
            "warm_transition_preferred": warm_preferred,
            "decision_rule": (
                "Prefer the linear control when configurations match and it either "
                "passes gates that hard fails, or reduces a spike multiplier by at "
                "least the material-reduction threshold."
            ),
        }
        (run_dir / "p7_hard_vs_linear_comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n"
        )
        output["comparison"] = comparison

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
