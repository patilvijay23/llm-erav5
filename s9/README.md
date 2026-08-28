# Session 9 — Loss Functions & Output Heads

This repository contains one Google-Colab-compatible notebook implementing the Session 9 loss harness end to end.

**Notebook:** `session9_loss_harness.ipynb`

The notebook is deterministic (`seed=42`), runs top-to-bottom, and uses a tiny causal GRU only to keep the focus on the assignment's output-head and loss mechanics. The same loss harness applies to a Transformer trunk.

## Part 1 — Seven required checks

1. **Tensor shapes**
  - `tokens`: `(3, 8)` = `[B, T]`
  - `hidden`: `(3, 8, 64)` = `[B, T, D]`
  - `logits`: `(3, 8, 512)` = `[B, T, V]`
  - shifted logits: `(3, 7, 512)` = `[B, T-1, V]`
  - shifted targets: `(3, 7)` = `[B, T-1]`
  - flattened CE rows: `(21, 512)` with 21 targets
2. **Shift verification**
  - The notebook prints **token strings rather than IDs**, e.g. `the -> capital`, `capital -> of`, `of -> india`, ...
  - An assertion confirms the harness uses `token[t]` as input and `token[t+1]` as target.
3. **Padding mask**
  - Contributors before masking: **21**
  - Contributors after masking: **16**
  - **5 padded targets** are removed from cross-entropy.
4. **Packed-document boundary**
  - Packed boundary shown explicitly as `'.' -> 'the'`.
  - Loss before boundary mask: **6.237519**
  - Loss after boundary mask: **6.237935**
  - Contributors: **14 -> 13**
  - The loss can move either direction. The correctness criterion is that the artificial cross-document target no longer contributes.
5. **Perplexity sanity check**
  - Untrained CE: **6.238752 nats**
  - Untrained perplexity: **512.22**
  - Vocabulary size: **512**
  - `PPL / V = 1.0004x`, so the untrained model is essentially uniform as expected.
6. **Tied vs. untied output head**
  - Configuration: `V=512`, `D=64`
  - Untied total parameters: **90,496**
  - Tied total parameters: **57,728**
  - Saving: **32,768 = V × D** parameters.
7. **Ordinary vs. chunked cross-entropy peak memory**
  - Ordinary peak incremental memory: **263.10 MiB**
  - Chunked peak incremental memory: **31.50 MiB**
  - Ratio: **8.35x** ordinary/chunked on the validated CPU reference run.
  - Before reporting memory, the notebook asserts that ordinary and chunked CE match in both **loss and gradients**.
  - On CUDA, the notebook automatically uses `torch.cuda.max_memory_allocated()`; on CPU it uses isolated subprocess RSS sampling.



## Part 2 — `t+2` output head

A second output head shares the same causal trunk and predicts two positions ahead:

`L_total = L_t+1 + L_t+2`

Reference validation run after 100 steps:

- `t+1` loss: **0.564501**
- `t+2` loss: **0.617737**
- sum: **1.182239**

Both losses fall from roughly `ln(512)`, but the `t+2` head remains higher. The notebook uses an online two-state Markov source so the difference is interpretable: one-step conditional entropy is **0.3251 nats**, while two-step conditional entropy is **0.4714 nats**. Predicting `t+2` therefore carries more irreducible uncertainty than predicting `t+1`, so its loss settles higher.

<b>Colab notebook rerun with 1000 steps:</b>

```json
"part2": {
    "loss_t1": 0.3183801770210266,
    "loss_t2": 0.46480923891067505,
    "loss_sum": 0.7831894159317017,
    "loss_gap_t2_minus_t1": 0.14642906188964844
```

## Run instructions

Open `session9_loss_harness.ipynb` in Google Colab and run **Runtime → Run all**. A GPU runtime is recommended for the assignment's peak-memory measurement, but the notebook also has a CPU measurement fallback. The final cell prints a Markdown-ready submission summary and writes `session9_results.json`.

### Note
Notebook was uploaded to colab, run on free tier T4 and then downloaded again to be committed to the git repo.