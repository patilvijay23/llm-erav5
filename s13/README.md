# Session 13 — 20M LLM Reversibility Experiment

This repo contains my Colab experiment for the Session 13 assignment: train a ~20M LLM for 50M tokens, repeat at the same batch size with reversibility, then push the reversible model to the maximum practical batch size and compare final loss, throughput, and peak GPU memory.

## Experiment

| Run | Architecture | Batch |
|---|---|---|
| A | standard residual transformer | fixed batch = 8 |
| B | reversible midpoint/leapfrog | fixed batch = 8 |
| C | reversible midpoint/leapfrog | maximum safe batch discovered on the assigned GPU |

A vs B isolates the memory/compute effect of reversibility at equal batch. A vs C asks whether saved activation memory enables enough additional batch to recover or improve throughput.

## Data and tokenizer

I use `allenai/peS2o`, config `v2`. I stream a medium ~64 MB corpus sample, train an 8,192-entry WordPiece vocabulary, save it in BERT format, then load that exact vocabulary with `BertTokenizerFlash`. FlashTokenizer is used to encode the training/validation caches.

I verified this design because FlashTokenizer's public API is an optimized WordPiece encoding engine; it does not expose a raw-corpus vocabulary trainer. The notebook therefore does not falsely claim that FlashTokenizer itself trained the vocabulary.

Documents are separated with `[SEP]` as EOS. Cross-document next-token transitions are masked.

## Model

The intended configuration is approximately 19.87M parameters: vocab 8,192; hidden size 384; 12 layers; 6 heads; MLP width 1,024; sequence length 512; RMSNorm; causal SDPA; tied input/output embeddings; dropout 0.

The notebook calculates and asserts the actual parameter count.

## Reversibility

The reversible variant is the Session 13 midpoint/leapfrog recurrence:

`p[l+1] = p[l-1] + 2h f_l(p[l])`

with inverse:

`p[l-1] = p[l+1] - 2h f_l(p[l])`

I use `h = 0.25`. A one-step Euler starter constructs the second boundary state, but the stack recurrence being tested is midpoint/leapfrog.

The lesson separately mentions a Lightning LM blend coefficient of 0.5. I do not inject an extra blend term into this simplified model because it is not present in the displayed Session 13 midpoint equation.

The custom autograd function saves only the final two stack states, reconstructs prior states during backward, and re-runs each local block to calculate gradients. This is not ordinary activation checkpointing relabeled as reversibility.

Dropout is zero because reconstruction must be deterministic.

## Correctness guardrail

Before long training, the notebook compares the custom reversible backward to ordinary autograd on the exact same three-block midpoint recurrence and checks output, input-gradient, and parameter-gradient errors. The full experiment should not proceed unless this audit passes.

## Training

All three runs use the same data, vocabulary, parameter count, sequence length, seed, AdamW family, cosine LR schedule, gradient clipping, next-token loss, and document-boundary mask. Each targets 50M processed tokens.

The last optimizer step can overshoot 50M by less than one micro-batch; exact `tokens_seen` is reported.

Precision is chosen from the actual Colab GPU: fp16 + GradScaler for T4-class hardware, bf16 on Ampere-or-newer GPUs where appropriate.

## Maximum batch

The reversible maximum-batch search performs a real forward, loss, backward, grad clip, AdamW state allocation, and optimizer step. It expands until OOM, then binary-searches the boundary. The long run uses 95% of the largest passing batch as an allocator-fragmentation safety margin.

## Metrics

The notebook reports final 100-step mean training loss, final validation loss, validation perplexity, overall tokens/s, peak allocated GPU memory, peak reserved memory, wall time, mean grad norm, p95 grad norm, exact batch size, steps, and exact tokens processed.

## Measured results

Hardware: **Tesla T4**  
Precision: **torch.float16**  
Parameters: **19,867,008**  
Vocabulary: **8,192**  
Reversible variant: **midpoint/leapfrog**, h=0.25  
Maximum safe reversible batch: **121**

| run | variant | batch | tokens | final val loss | token/s | peak GiB | wall min |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline_fixed | standard residual | 8 | 50.004M | 0.2454 | 51,776 | 1.29 | 16.1 |
| reversible_fixed | midpoint/leapfrog | 8 | 50.004M | 0.2504 | 41,396 | 0.58 | 20.1 |
| reversible_max_batch | midpoint/leapfrog | 121 | 50.057M | 0.2428 | 50,724 | 5.28 | 16.4 |

### Reversibility numerical audit
- output max abs error: `0`
- input-gradient max abs error: `3.0618e-07`
- parameter-gradient max abs error: `1.01328e-06`

## Findings

The experiment showed a clear **memory-versus-compute trade-off** from reversibility.

At the same batch size of 8, the standard model used **1.29 GiB** of peak GPU memory, while the reversible midpoint model used only **0.58 GiB**. This is a reduction of approximately **55% in peak memory**, showing that reconstructing intermediate states during backward substantially reduced activation storage.

The memory saving came with a throughput cost. At batch size 8, the baseline processed **51,776 tokens/s**, while the reversible model processed **41,396 tokens/s**, about **20% slower**. This is expected because the reversible backward pass must re-evaluate transformer blocks while reconstructing the previous hidden states.

The memory reduction allowed the reversible model to increase batch size from **8 to 121**, which is about a **15.1× larger batch**. The batch-size probe reached the configured upper probe limit of 128, and the notebook applied its 95% safety margin to select **121** for the full training run.

At this larger batch size, the reversible model achieved **50,724 tokens/s**, compared with **51,776 tokens/s** for the standard baseline. Therefore, increasing the batch recovered almost all of the throughput lost to reversible recomputation: the maximum-batch reversible run reached approximately **98% of baseline throughput** and was about **22.5% faster than the reversible run at batch size 8**.

The larger batch naturally increased total memory usage. Peak memory rose from **0.58 GiB at batch 8 to 5.28 GiB at batch 121**. This shows that after removing much of the depth-dependent activation storage, **batch-dependent tensors and temporary working memory became increasingly important**. The aggregate CUDA measurement alone does not identify one specific tensor as the new bottleneck, but memory was clearly scaling primarily with the much larger batch rather than with stored activations across transformer depth.

All three models completed approximately **50M training tokens without instability**. Final validation losses were also very close: **0.2454** for the baseline, **0.2504** for reversible training at batch 8, and **0.2428** for reversible training at batch 121. The differences are small enough that there is no evidence in this experiment that midpoint reversibility materially degraded model quality. In fact, the maximum-batch reversible run obtained the lowest measured validation loss, although one run per configuration is not enough to conclude that reversibility itself caused the improvement.

The numerical audit also supports the correctness of the custom reversible backward implementation. The forward output difference was **0**, while the maximum input-gradient and parameter-gradient errors were only **3.06×10⁻⁷** and **1.01×10⁻⁶**, respectively.

Overall, my experiment demonstrates the purpose of reversibility clearly: **it does not make each training step cheaper, instead it trades additional computation for significantly lower activation memory**. At the same batch size this made training slower, but the saved memory allowed a roughly **15× larger batch**, which recovered almost all of the original model's throughput while maintaining comparable loss.

## Colab procedure

Open the notebook, select a GPU runtime, first run with `RUN_FULL=False`, confirm the numerical audit succeeds, then set `RUN_FULL=True` and Run All. Download the executed notebook plus `session13_results.json`, `README_RESULTS.md`, `loss_curves.png`, `peak_memory.png`, and `throughput.png`.

Free Colab does not guarantee one GPU model. The notebook records the actual accelerator; a T4 result should not be presented as an L4/A100 result.

## Sources

- ERA V5 Session 13 — Distributed Training II, Model and Pipeline Parallel
- Gal et al. (2025), *Reversing Large Language Models for Efficient Training and Fine-Tuning*
- AllenAI peS2o
- FlashTokenizer
