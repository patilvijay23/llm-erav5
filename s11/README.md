# Adam, Bias Correction, Update Ratios, Schedules, and Learning-Rate Scaling

This submission reproduces Adam by hand, verifies the calculation against PyTorch, measures the practical effect of bias correction, instruments update-to-weight ratios layer-by-layer, compares cosine and WSD under an early stop, and sweeps learning rates at three widths before extrapolating to width 4,096.

The support code is in `optimizer_assignment.py`. All plots, CSVs, and JSON summaries are written to `results/`.

> **Comparison policy:** I use identical initialization, identical batches, identical optimizer family, and identical evaluation data wherever two methods are compared. The only intended difference in each A/B test is the variable named in that experiment. For the cosine-vs-WSD comparison, the script exposes **separate peak learning rates** (`--cosine-lr` and `--wsd-lr`) so both sides can be tuned before accepting the comparison.

---

## Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch numpy matplotlib
```

Run everything:

```bash
python optimizer_assignment.py --part all --device auto --outdir results
```

Run only one section:

```bash
python optimizer_assignment.py --part 1
python optimizer_assignment.py --part 2
python optimizer_assignment.py --part 3
python optimizer_assignment.py --part 4
python optimizer_assignment.py --part 5
```

For a CPU-only laptop, first validate with a cheap smoke test:

```bash
python optimizer_assignment.py --part 1
python optimizer_assignment.py --part 2
python optimizer_assignment.py --part 3 --part3-width 128 --part3-steps 60 --batch-size 32
python optimizer_assignment.py --part 4 --part4-width 128 --batch-size 32
```

The required width-1,024 sweep is better suited to a GPU/Colab run.

---

# 1. Reproduce Adam by hand

For a scalar parameter \(w\), gradient \(g_t\), learning rate \(\alpha\), and Adam constants \(\beta_1,\beta_2,\epsilon\):

\[
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t
\]

\[
v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2
\]

Because both moving averages start at zero, they are biased toward zero during the first few updates. Adam corrects that with:

\[
\hat m_t = \frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t = \frac{v_t}{1-\beta_2^t}
\]

and updates the parameter with

\[
\Delta w_t =
\alpha \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon},
\qquad
w_t = w_{t-1}-\Delta w_t.
\]

I used:

- initial weight \(w_0=1.5\)
- gradients: `[0.20, -0.10, 0.05, -0.30, 0.15]`
- learning rate `0.01`
- \(\beta_1=0.9\)
- \(\beta_2=0.999\)
- \(\epsilon=10^{-8}\)

### Hand calculation

| t | gradient | m | v | m-hat | v-hat | Adam step | weight after |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.20000 | 0.020000 | 0.0000400000 | 0.200000 | 0.0400000000 | 0.010000 | 1.490000 |
| 2 | -0.10000 | 0.008000 | 0.0000499600 | 0.042105 | 0.0249924962 | 0.002663 | 1.487337 |
| 3 | 0.05000 | 0.012200 | 0.0000524100 | 0.045018 | 0.0174874950 | 0.003404 | 1.483932 |
| 4 | -0.30000 | -0.019020 | 0.0001423576 | -0.055307 | 0.0356428361 | -0.002929 | 1.486862 |
| 5 | 0.15000 | -0.002118 | 0.0001647153 | -0.005172 | 0.0330090065 | -0.000285 | 1.487147 |

The script repeats these calculations and then performs the same five updates with `torch.optim.Adam` in float64. It asserts that `m`, `v`, and the resulting weight agree to floating-point precision.

Expected conclusion:

> My scalar implementation and PyTorch agree well beyond several decimal places. This verifies the recurrence, bias correction, sign convention, and epsilon placement.

Generated evidence:

- `results/part1_adam_hand_vs_torch.csv`

---

# 2. Disable bias correction

I compare two scalar Adam optimizers over the first 20 updates:

1. normal Adam with \(\hat m_t,\hat v_t\)
2. an otherwise identical implementation using raw \(m_t,v_t\)

The experiment deliberately uses a constant gradient so that the transient introduced by zero-initialized moments is easy to see.

Run:

```bash
python optimizer_assignment.py --part 2
```

Generated plots:

- `results/part2_bias_correction_weights.png`
- `results/part2_bias_correction_difference.png`

Generated data:

- `results/part2_bias_correction.csv`

### What “stops mattering” means

This phrase needs an explicit numerical definition. I use:

> Bias correction has stopped materially affecting the update once the **relative difference between the corrected and uncorrected step is below 1% and remains below 1% for every later observed step**.

The script reports the first step satisfying that rule within the 20-step window.

If it prints `None`, the correct report is not to invent a crossing point. Instead report:

> Under the 1% criterion, bias correction still materially affects the update at step 20.

That is a better answer than visually declaring the curves “close enough.”

### Why the early difference is large

With \(m_0=v_0=0\), both exponential moving averages are artificially small at the beginning. The correction factors divide out exactly the amount of zero-initialization bias expected after \(t\) steps. Since \(\beta_2=0.999\), the second moment has an especially long transient.

---

# 3. Log update-to-weight ratio for every layer

For each parameter tensor \(W\), immediately before and after `optimizer.step()` I compute

\[
\text{update-to-weight ratio}
=
\frac{\|\Delta W\|_2}{\|W\|_2}
=
\frac{\|W_{\text{after}}-W_{\text{before}}\|_2}
{\|W_{\text{before}}\|_2}.
\]

Run:

```bash
python optimizer_assignment.py --part 3
```

Default configuration:

- width: 256
- warmup: 30 optimizer steps
- base LR: \(10^{-3}\)
- optimizer: AdamW
- model: 3-layer MLP

Generated outputs:

- `results/part3_update_weight_ratio.csv`
- `results/part3_update_weight_ratio.png`

The CSV logs, for **every parameter tensor and every step**:

- loss
- learning rate
- weight norm
- update norm
- update/weight ratio

### Identifying where warmup stops changing the ratio

The LR warmup is linear:

\[
\alpha_t = \alpha_{\max}\min(1,t/T_\text{warmup})
\]

with `T_warmup = 30`.

Therefore the schedule stops directly changing the update scale at **step 30**. The important empirical check is that the update/weight curves lose the systematic LR-ramp trend at the same point. The plot marks step 30 with a vertical dashed line.

My report should distinguish two ideas:

- **Scheduler fact:** warmup ends at step 30 by construction.
- **Observed optimizer behavior:** after approximately that point, the ratios are no longer being pushed upward by an increasing LR; remaining movement comes from gradients, Adam moments, and changing parameter norms.

This avoids overclaiming that the ratio must become numerically flat.

---

# 4. Cosine vs WSD, both planned for 300 steps and stopped at 200

Both runs:

- start from exactly the same model weights
- see exactly the same batch at every matching optimizer step
- use AdamW
- are evaluated on the same fixed held-out synthetic set
- are *planned* as 300-step schedules
- are forcibly stopped at step 200

## Cosine

After warmup, cosine begins decaying immediately:

\[
s_\text{cos}(t)=r_\min+(1-r_\min)
\frac{1+\cos(\pi p)}{2}.
\]

## WSD

WSD has three phases:

1. **Warmup**
2. **Stable** LR plateau
3. **Decay** near the intended end of training

The default 300-step WSD schedule uses:

- warmup: 30 steps
- stable region through step 240
- decay: last 60 steps

Because training is stopped at 200, the WSD run has **not entered its decay phase at all**. This is the central point of the experiment.

Run:

```bash
python optimizer_assignment.py --part 4
```

Outputs:

- `results/part4_loss_cosine_vs_wsd.png`
- `results/part4_lr_cosine_vs_wsd.png`
- `results/part4_schedule_comparison.csv`
- `results/part4_result.json`

### Tune both sides

Do not compare schedules only at one arbitrarily shared peak LR. Tune the peak LR of each schedule independently, then rerun the final pair:

```bash
python optimizer_assignment.py \
  --part 4 \
  --cosine-lr <best_cosine_lr> \
  --wsd-lr <best_wsd_lr>
```

A practical tuning grid is:

```text
2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3
```

Use the same initialization/batches for every candidate and select each schedule's LR using the same evaluation metric.

### Result to report

Copy the two values from `results/part4_result.json`:

```text
Cosine loss at step 200: <value>
WSD loss at step 200:    <value>
Model I would keep:      <cosine or WSD>
```

The model to keep is the one with the lower fixed-set evaluation loss **after each schedule has been tuned fairly**.

### Interpretation

At an unexpected early stop, cosine has already spent some of its LR budget decaying, while WSD is still in its stable phase. This is exactly why WSD is attractive when the eventual stopping point may change: it postpones the irreversible decay phase until late in the planned run.

That statement is a schedule-mechanics explanation, not a claim that WSD must always win. The measured losses decide this run.

---

# 5. Learning-rate sweep at widths 256, 512, and 1,024

The required sweep is:

```bash
python optimizer_assignment.py \
  --part 5 \
  --widths 256 512 1024 \
  --lrs 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032
```

Each width/LR pair:

- starts from a deterministic seed
- trains for the same number of steps
- uses the same scheduler form
- is evaluated on the same fixed held-out set

Outputs:

- `results/part5_lr_sweep.csv`
- `results/part5_lr_sweep.png`
- `results/part5_result.json`

The plot shows loss against LR on a log LR axis and marks the minimum observed loss for all three widths.

## Extrapolating to width 4,096

I do **not** simply copy the LR from width 1,024. Instead I fit the empirical power law

\[
\eta^\*(d)=C d^a
\]

through the three observed optimal pairs:

\[
(256,\eta^\*_{256}),\quad
(512,\eta^\*_{512}),\quad
(1024,\eta^\*_{1024}).
\]

Taking logs gives a straight-line fit:

\[
\log \eta^\* = \log C + a\log d.
\]

The script then evaluates the fitted relation at \(d=4096\) and writes:

```text
recommended_lr_width_4096
power_law_slope
confidence
```

to `part5_result.json`.

### Confidence statement

My confidence should be **moderate at best**, even with clean minima, because 4,096 is an extrapolation beyond the largest measured width and the fit uses only three points.

If any measured optimum is at the smallest or largest LR in the grid, confidence should be lower because the true minimum may lie outside the sweep. In that case I would widen the grid before submitting a final 4,096 recommendation.

A strong final statement is:

> I would start width 4,096 at `<recommended_lr_width_4096>`. Confidence is `<confidence>` because this is a three-point extrapolation rather than a direct measurement. I would validate it with a narrow local sweep around that value before committing a full training run.

---

# Fair-comparison checklist

The warning in the assignment matters more than any single curve. Before accepting an optimizer or scheduler comparison I verify:

1. **Same initialization.** A/B runs load an identical copied `state_dict`.
2. **Same data order.** Matching steps use matching deterministic batches.
3. **Same number of optimizer steps.**
4. **Same evaluation set and metric.**
5. **Same optimizer hyperparameters except the variable under test.**
6. **Tune both sides.** Cosine and WSD get independent peak-LR tuning.
7. **Do not infer a minimum from a sweep edge.** Expand the LR grid when needed.
8. **Use more than one seed for a publication-level claim.** The supplied script is a clean assignment harness; for a stronger claim I would repeat the final candidates over 3–5 seeds and report mean ± standard deviation.

---

# Files produced

| File | Purpose |
|---|---|
| `optimizer_assignment.py` | Complete experiment harness |
| `results/part1_adam_hand_vs_torch.csv` | Manual vs PyTorch Adam values |
| `results/part2_bias_correction.csv` | Corrected vs uncorrected first 20 updates |
| `results/part2_bias_correction_weights.png` | Weight trajectories |
| `results/part2_bias_correction_difference.png` | Relative update difference |
| `results/part3_update_weight_ratio.csv` | Per-layer update/weight ratios |
| `results/part3_update_weight_ratio.png` | Ratio curves + warmup boundary |
| `results/part4_schedule_comparison.csv` | Cosine/WSD loss and LR by step |
| `results/part4_loss_cosine_vs_wsd.png` | Training-loss comparison |
| `results/part4_lr_cosine_vs_wsd.png` | Schedule shape through step 200 |
| `results/part4_result.json` | Final losses and keep decision |
| `results/part5_lr_sweep.csv` | Width/LR/loss sweep |
| `results/part5_lr_sweep.png` | LR curves with three minima |
| `results/part5_result.json` | Best LRs + width-4096 extrapolation |

---

# Short conclusion

This exercise exposed three optimizer behaviors that are easy to hide behind a library call.

First, Adam's bias correction is not cosmetic: zero-initialized moments create a large early transient, and the correction removes it analytically. Second, warmup can be seen directly in update-to-weight ratios rather than only in an LR plot, which makes the scale of actual parameter motion observable. Third, schedule comparisons depend on the stopping rule: a 300-step cosine schedule has already decayed by an unexpected stop at 200, while a 300-step WSD schedule can still be in its stable phase.

The final LR-scaling conclusion should come from measured minima rather than a memorized scaling rule. The width-4,096 LR is therefore reported as an extrapolated starting point, not as a known optimum.
