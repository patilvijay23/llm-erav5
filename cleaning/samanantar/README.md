# Session 4 — Data cleanup widget

## Dataset selected

`ai4bharat/samanantar` replaces the earlier reasoning-distillation corpus. The Hugging Face repository reports **49,774,246 English–Indic sentence pairs**, **11 Indic languages**, **7.22 GB** of Parquet data, and a **CC BY-NC 4.0** license. The paper describes **12.4M existing pairs + 37.4M web-mined pairs**.

## Eight cleanup strategies

1. Unicode/text normalization, preserving Indic ZWJ/ZWNJ
2. Canonical bilingual pair schema (`src`, `tgt`, language, provenance)
3. Quality and semantic-alignment filtering
4. Global exact and near deduplication
5. Language/script identification and validation
6. Structured PII masking with alignment-aware review
7. Evaluation decontamination against immutable benchmark fingerprints
8. Reproducible manifests and content hashes

## Run on Hugging Face using streaming

```bash
pip install datasets pandas pyarrow
python clean_samanantar.py --hf-dataset ai4bharat/samanantar --languages as,bn,gu,hi,kn,ml,mr,or,pa,ta,te --output-dir cleaned_samanantar
```

For Samanantar v0.3 with LaBSE metadata, add the correct score column:

```bash
python clean_samanantar.py --input /path/to/v0.3 --language as --semantic-score-field labse_score --semantic-threshold 0.70
```

For decontamination, provide one held-out benchmark sentence per line:

```bash
python clean_samanantar.py --input shard.parquet --language hi --heldout flores_dev_devtest.txt
```

Outputs: `cleaned.jsonl`, `rejected.jsonl`, and `audit_manifest.json`.

## Scope and honesty

The public release states that it is shuffled and deduplicated, but Session 4 still requires a global verification pass after normalization. The widget reports a conservative manual lower bound of 30 visibly misaligned items in the first 100 Assamese viewer rows. It does **not** extrapolate that rate to all 49.8M pairs. Full-corpus semantic, near-duplicate, PII, and benchmark-overlap counts must come from an executed run with the required score metadata and benchmark registry.
