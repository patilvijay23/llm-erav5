#!/usr/bin/env python3
"""Prepare exact FineWeb-Edu and IndicCorpV2 training pools plus held-out data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from p7.data import prepare_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        choices=["fineweb_edu", "indiccorp_v2", "both"],
        help="Pool to prepare.",
    )
    parser.add_argument("--output-root", default="data/p7")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="V5 tokenizer path or Hugging Face ID.",
    )
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--expected-vocab-size", type=int, default=196_608)
    parser.add_argument("--allow-vocab-mismatch", action="store_true")
    parser.add_argument(
        "--train-tokens",
        type=int,
        default=10_000_000_000,
        help=(
            "Exact trainable tokens per source, excluding the evaluation holdout. "
            "The default produces a true 10B-token training pool."
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=640,
        help="640 makes all reportable P7 allocations exactly divisible.",
    )
    parser.add_argument(
        "--eval-sequences",
        type=int,
        default=10_000,
        help=(
            "Held-out sequences prepared in addition to --train-tokens. With the "
            "defaults this adds 6.4M tokens per source."
        ),
    )
    parser.add_argument(
        "--shard-sequences",
        type=int,
        default=1_000_000,
        help="1M x 640 tokens = 640M tokens (~2.56GB uint32) per shard.",
    )
    parser.add_argument("--tokenize-batch-size", type=int, default=128)
    parser.add_argument("--shuffle-buffer", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=5678)
    parser.add_argument(
        "--fineweb-config",
        default="sample-100BT",
        help=(
            "Use sample-100BT, not sample-10BT, because the latter is 10B GPT-2 "
            "tokens and may contain fewer than 10B V5-tokenizer tokens."
        ),
    )
    parser.add_argument("--fineweb-revision", default=None)
    parser.add_argument(
        "--indic-revision",
        default="984b75b20ce408f9ba27c6558e9279e8e1b6edfd",
    )
    parser.add_argument(
        "--indic-splits",
        nargs="*",
        default=None,
        help="Optional explicit IndicCorpV2 split list; default uses all splits.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sources = (
        ["fineweb_edu", "indiccorp_v2"] if args.source == "both" else [args.source]
    )
    manifests: dict[str, str] = {}
    for offset, source in enumerate(sources):
        manifest = prepare_pool(
            source=source,
            output_dir=output_root / source,
            tokenizer_name_or_path=args.tokenizer,
            tokenizer_revision=args.tokenizer_revision,
            expected_vocab_size=args.expected_vocab_size,
            allow_vocab_mismatch=args.allow_vocab_mismatch,
            train_tokens=args.train_tokens,
            block_size=args.block_size,
            eval_sequences=args.eval_sequences,
            shard_sequences=args.shard_sequences,
            tokenize_batch_size=args.tokenize_batch_size,
            seed=args.seed + offset,
            shuffle_buffer=args.shuffle_buffer,
            overwrite=args.overwrite,
            fineweb_config=args.fineweb_config,
            fineweb_revision=args.fineweb_revision,
            indic_revision=args.indic_revision,
            indic_splits=args.indic_splits,
        )
        manifests[source] = str(manifest)
        manifest_data = json.loads(Path(manifest).read_text())
        print(
            f"Prepared {source}: {manifest} | "
            f"train={manifest_data['train_tokens']:,} | "
            f"eval={manifest_data['eval_tokens']:,} | "
            f"total={manifest_data['actual_tokens']:,}"
        )
    (output_root / "p7_manifests.json").write_text(
        json.dumps(manifests, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
