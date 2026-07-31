# P7 — hard mixture shift versus a matched warm-transition control

This package implements **P7** from the V5 mixture-and-curriculum submission:

> Test whether replacing a gradual mixture transition with one hard shift causes
> a larger loss or gradient-norm disturbance in a 1B proxy model.

The experiment uses only:

- `HuggingFaceFW/fineweb-edu`
- `ai4bharat/IndicCorpV2`

The **reportable P7 result requires two matched runs**, not one:

1. **Hard arm:** 50:50 FineWeb/IndicCorp until 10B tokens, then an immediate
   80:20 shift.
2. **Linear control:** the same change linearly blended across a centred
   256M-token window.

The model, tokenizer, data manifests, source-specific permutations, seed,
optimizer and learning-rate schedule are identical. Only the transition shape
changes.

> **Execution status:** the reportable 1B/20B-token runs have not been executed.
> The CPU-only laptop profile validates data and scheduling mechanics only.

## 1. Exact experiment accounting

### Training schedule

| Interval | Hard-arm FineWeb | Hard-arm IndicCorp |
|---|---:|---:|
| 0–10B training tokens | 50% = 5B | 50% = 5B |
| 10–20B training tokens | 80% = 8B | 20% = 2B |
| **Complete run** | **13B** | **7B** |

Each source supplies exactly **10B trainable tokens**. The 13B FineWeb draw
therefore includes exactly **3B tokens of deterministic second-pass replay**.
Only 7B of the 10B IndicCorp training pool is selected.

### Evaluation data is additional

The preparation command adds the held-out sequences **after** the requested
training pool:

| Quantity per source | Tokens |
|---|---:|
| Trainable pool | 10,000,000,000 |
| Evaluation holdout: 10,000 × 640 | 6,400,000 |
| **Total packed data** | **10,006,400,000** |

This avoids the earlier accounting error where the holdout was subtracted from
the nominal 10B pool and FineWeb replay was understated.

## 2. Why the linear control is centred

The recommended linear control uses a **256M-token transition centred on the
10B boundary**:

- transition begins at 9.872B tokens;
- transition midpoint is 10B tokens;
- transition ends at 10.128B tokens.

Across that window, FineWeb rises linearly from 50% to 80%. Its average share is
65%, which is also the average share of the hard arm across a centred window
whose first half is 50% and second half is 80%.

Consequently, both arms consume exactly:

- **13B FineWeb-Edu tokens**;
- **7B IndicCorpV2 tokens**; and
- **20B tokens overall**.

This makes transition shape the intended independent variable.

The experiment tests only the direction **50:50 → 80:20**. It must not be
presented as evidence about an Indic-increasing shift or every transition in the
full ten-lane curriculum.

## 3. 1B proxy architecture

The script constructs a Llama-style dense decoder:

| Setting | Value |
|---|---:|
| Vocabulary | 196,608 expected |
| Hidden size | 1,792 |
| Layers | 18 |
| Attention heads | 14 |
| KV heads | 7 |
| MLP size | 4,864 |
| Tied input/output embeddings | Yes |
| Parameters at vocabulary 196,608 | Approximately 996.4M |
| Fixed sequence length | 640 |

Context is fixed because P7 measures transition stability, not long-context
capability.

With 8 GPUs, per-device batch 1 and gradient accumulation 25:

| Quantity | Value |
|---|---:|
| Global sequences per optimizer step | 200 |
| Tokens per optimizer step | 128,000 |
| Hard-shift step | 78,125 |
| Final step | 156,250 |
| Linear-control transition | Steps 77,125–79,125 |

## 4. Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set the exact tokenizer used by every proxy arm:

```bash
export V5_TOKENIZER=/path/to/v5-tokenizer
```

## 5. Prepare exact training pools and additional holdouts

```bash
python prepare_p7_data.py both \
  --tokenizer "$V5_TOKENIZER" \
  --output-root data/p7 \
  --train-tokens 10000000000 \
  --eval-sequences 10000 \
  --block-size 640
```

Outputs:

```text
data/p7/fineweb_edu/manifest.json
data/p7/indiccorp_v2/manifest.json
data/p7/p7_manifests.json
```

Each manifest records training tokens, evaluation tokens, dataset revision,
tokenizer fingerprint, packed-shard hashes and Indic split composition.

FineWeb preparation defaults to `sample-100BT`, then stops at exactly 10B
**V5-tokenizer training tokens** plus the holdout. The published `sample-10BT`
name is based on GPT-2 tokenization, so it is not assumed to equal 10B V5 tokens.

IndicCorpV2 is sampled uniformly by document across active language-script
splits. Reuse the same manifests in both arms; regenerating them would introduce
a data-composition confound.

## 6. Launch the hard P7 arm

```bash
accelerate launch \
  --config_file configs/accelerate_fsdp_8gpu.yaml \
  train_p7.py \
  --transition-mode hard \
  --fineweb-manifest data/p7/fineweb_edu/manifest.json \
  --indic-manifest data/p7/indiccorp_v2/manifest.json \
  --tokenizer "$V5_TOKENIZER" \
  --output-dir runs/p7-hard \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 25 \
  --block-size 640 \
  --total-tokens 20000000000 \
  --shift-tokens 10000000000 \
  --expected-source-train-tokens 10000000000 \
  --seed 1234 \
  --data-seed 5678
```

## 7. Launch the matched linear control

```bash
accelerate launch \
  --config_file configs/accelerate_fsdp_8gpu.yaml \
  train_p7.py \
  --transition-mode linear \
  --transition-tokens 256000000 \
  --fineweb-manifest data/p7/fineweb_edu/manifest.json \
  --indic-manifest data/p7/indiccorp_v2/manifest.json \
  --tokenizer "$V5_TOKENIZER" \
  --output-dir runs/p7-linear \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 25 \
  --block-size 640 \
  --total-tokens 20000000000 \
  --shift-tokens 10000000000 \
  --expected-source-train-tokens 10000000000 \
  --seed 1234 \
  --data-seed 5678
```

The code refuses a transition width that does not align with optimizer-step
boundaries or that cannot be centred symmetrically around 10B tokens.

## 8. Resume either arm

```bash
accelerate launch \
  --config_file configs/accelerate_fsdp_8gpu.yaml \
  train_p7.py \
  --transition-mode hard \
  --fineweb-manifest data/p7/fineweb_edu/manifest.json \
  --indic-manifest data/p7/indiccorp_v2/manifest.json \
  --tokenizer "$V5_TOKENIZER" \
  --output-dir runs/p7-hard \
  --resume-from latest
```

Use the same original arguments when resuming. Replace `hard` with `linear` and
include `--transition-tokens 256000000` for the control arm.

## 9. Metrics recorded

Each run writes `metrics.jsonl` containing:

- processed tokens and optimizer step;
- transition mode and phase;
- exact FineWeb and IndicCorp share;
- training loss;
- pre-clipping gradient norm;
- learning rate and throughput;
- source-pass count, exposing FineWeb replay; and
- fixed-holdout loss for each source.

Evaluation is forced around the transition start, centre and end, in addition to
the normal evaluation interval.

## 10. Analyse and compare the two arms

```bash
python analyze_p7.py \
  --run-dir runs/p7-hard \
  --control-run-dir runs/p7-linear \
  --window-steps 500 \
  --material-reduction 0.10
```

Outputs include:

```text
runs/p7-hard/p7_transition_summary.json
runs/p7-hard/p7_transition_window.csv
runs/p7-linear/p7_transition_summary.json
runs/p7-linear/p7_transition_window.csv
runs/p7-hard/p7_hard_vs_linear_comparison.json
```

The analyzer checks that the arms use matching manifests, seeds, optimizer
settings, batch size and complete-run source totals.

### Stability gates

A run passes the initial gate only when:

- peak gradient norm is no more than **2.5×** the pre-transition median;
- peak loss is no more than **1.15×** the pre-transition median; and
- both signals recover inside the gate within **500 optimizer steps** after the
  transition finishes.

The warm transition is preferred when the configurations match and either:

1. the hard arm fails while the linear arm passes; or
2. the linear arm reduces the gradient or loss spike multiplier by at least 10%.

The 10% materiality threshold is itself a declared hypothesis and should be
reported alongside raw values and seed spread.

## 11. CPU-only laptop smoke test

An 8GB CPU-only laptop cannot instantiate and train the fixed 996M model with
AdamW. The laptop profile validates preparation, exact source accounting,
scheduling, replay and file plumbing only.

Prepare **1.28M trainable tokens plus 100 held-out sequences per source**:

```powershell
python prepare_p7_data.py both `
  --tokenizer C:\path\to\v5-tokenizer `
  --output-root data\p7-laptop `
  --train-tokens 1280000 `
  --eval-sequences 100 `
  --block-size 128 `
  --shard-sequences 1000 `
  --tokenize-batch-size 8 `
  --shuffle-buffer 1000
```

Per source this produces:

| Quantity | Tokens |
|---|---:|
| Trainable pool | 1,280,000 |
| Evaluation holdout: 100 × 128 | 12,800 |
| **Total packed data** | **1,292,800** |

The scaled 2.56M-token schedule still consumes exactly:

- 1.664M FineWeb tokens;
- 896K IndicCorp tokens;
- 384K FineWeb replay tokens; and
- leaves 384K IndicCorp training tokens unused.

Run lightweight validation:

```powershell
pytest -q
python -m py_compile prepare_p7_data.py train_p7.py analyze_p7.py p7\data.py p7\model.py p7\schedule.py
```

Do not describe the laptop output as evidence about 1B-model stability.

## 12. Validation

```bash
pytest -q
python -m py_compile prepare_p7_data.py train_p7.py analyze_p7.py p7/*.py
```

The tests verify:

- exact hard-arm 50:50 and 80:20 counts;
- exact 13B/7B complete-run totals;
- a centred linear transition with the same totals;
- optimizer-step alignment;
- unique source draws within each pass;
- deterministic second-pass FineWeb replay; and
- packed-shard/memory-map round trips with separate training and evaluation
  token accounting.

## 13. Repository files

```text
prepare_p7_data.py                 exact train pool + additional holdout
train_p7.py                        hard and linear 1B FSDP training arms
analyze_p7.py                      per-run summary and matched comparison
p7/data.py                         streaming tokenization and mmap pools
p7/model.py                        approximately 996M proxy architecture
p7/schedule.py                     deterministic hard and linear schedules
configs/accelerate_fsdp_8gpu.yaml  recommended 8-GPU FSDP2 launch
configs/p7.json                    experiment accounting
SUBMISSION_README.md               assignment with corrected P7 section
tests/                              schedule and data-accounting tests
```

## Dataset references

- FineWeb-Edu: <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>
- IndicCorpV2: <https://huggingface.co/datasets/ai4bharat/IndicCorpV2>
- Hugging Face streaming datasets: <https://huggingface.co/docs/datasets/main/stream>
- Accelerate FSDP: <https://huggingface.co/docs/accelerate/usage_guides/fsdp>
