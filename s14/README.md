# Session 14 — Dense LLM → Mixture-of-Experts

## Assignment

Ask: **Train a standard model and convert it into a Mixture-of-Experts model. The model size and training data are our choice, but both stages must continue training and reduce loss.**

For this experiment, I trained a small dense causal language model from scratch on **allenai/peS2o**, converted its feed-forward layers into a sparse Mixture-of-Experts architecture using **sparse upcycling**, and then continued training the MoE on the same data. I first tried re-using the model from s13 assignment, but the error became worse on continuining training, hence started a new model from scratch.

The experiment follows this sequence:

```text
Dense model from scratch
        │
        │ 5M tokens
        ▼
Trained dense model
        │
        │ Sparse upcycling
        ▼
4-expert / Top-2 MoE
        │
        │ 5M additional tokens
        ▼
Trained MoE
```

The important result is that the dense model first learned successfully, the dense → MoE conversion preserved what it had already learned, and the MoE then continued reducing loss.

Results from colab run are present in the notebook as well as in `s14/s14_moe/README_RESULTS.md`.

---



## Dataset

`allenai/peS2o`**, configuration** `v2`

peS2o is a corpus of scientific papers and provides substantial natural-language text for a small causal language-model experiment.

The same tokenized data pipeline is used for both the dense and MoE phases so that the architecture conversion is the primary change between them.

Documents are separated using `[SEP]` as an EOS token. The next-token loss masks the transition from one document into the next so unrelated papers are not treated as continuous text.

---



## Tokenizer

I trained an **8,192-token WordPiece vocabulary** from approximately **32 MB of peS2o text**.

The pipeline is:

1. Stream a medium-sized peS2o corpus.
2. Train an 8,192-token WordPiece vocabulary.
3. Save the vocabulary in BERT format.
4. Load it with **FlashTokenizer (**`BertTokenizerFlash`**)**.
5. Use FlashTokenizer to encode the training and validation corpus.
6. Verify its token IDs against `BertTokenizerFast`.

FlashTokenizer is used as the fast WordPiece encoding engine; the vocabulary itself is trained using the WordPiece trainer from the Hugging Face `tokenizers` library.

---



## Dense Model

The starting model is a standard dense causal transformer.


| Component             | Value          |
| --------------------- | -------------- |
| Parameters            | **19,867,008** |
| Vocabulary            | 8,192          |
| Hidden size           | 384            |
| Layers                | **12**         |
| Attention heads       | 6              |
| FFN width             | 1,024          |
| Sequence length       | 512            |
| Batch size            | 8              |
| Dense training tokens | ~5M            |


The model uses:

- causal scaled-dot-product attention;
- RMSNorm;
- SiLU feed-forward layers;
- tied token embedding and output-head weights;
- AdamW optimizer;
- cosine learning-rate decay;
- gradient clipping.

The dense transformer block is conceptually:

```text
token representation
        │
        ▼
self-attention
        │
        ▼
dense FFN
        │
        ▼
next layer
```

---



## Dense Training

The dense model was trained from scratch for **5.001M tokens** on a Tesla T4 using fp16.

The untrained model started with a fixed validation loss of approximately:

```text
9.229876
```

After 5M tokens:

```text
0.428352
```

The mean training loss also dropped substantially:

```text
first 100 steps : 3.058409
last 100 steps  : 0.441072
```

This demonstrates that the dense model had learned a meaningful language-model function before being converted into an MoE.

---



# Converting the Dense Model into an MoE



## What Changes?

The attention layers are left unchanged.

Only the dense feed-forward network in each transformer block is replaced by a **router plus multiple expert FFNs**.

```text
Dense transformer

attention
   │
   ▼
one FFN


MoE transformer

attention
   │
   ▼
 router
   │
   ├──── Expert 0
   ├──── Expert 1
   ├──── Expert 2
   └──── Expert 3
          │
        Top-2
          │
          ▼
weighted expert output
```

An expert is simply a feed-forward network. It has no attention or embeddings of its own.

---



## MoE Layout

I used:


| MoE setting                | Value     |
| -------------------------- | --------- |
| Routed experts per layer   | **4**     |
| Experts selected per token | **Top-2** |
| Routing                    | Dropless  |
| Router precision           | fp32      |
| Balance-loss coefficient   | 0.001     |
| Router z-loss coefficient  | 0.001     |


For every token, the router computes one score for each expert:

r=xW_r

The scores are converted to probabilities and the two highest-scoring experts are selected.

Their retained probabilities are renormalized so that:

w_1+w_2=1

and the MoE output becomes:

y=w_1E_1(x)+w_2E_2(x)

---



# Sparse Upcycling

Instead of randomly initializing the MoE experts, I used **sparse upcycling**.

Every new expert starts as an exact copy of the already-trained dense FFN.

```text
trained dense FFN
       │
       ├── copy → Expert 0
       ├── copy → Expert 1
       ├── copy → Expert 2
       └── copy → Expert 3
```

Therefore, at the instant of conversion:

E_0(x)=E_1(x)=E_2(x)=E_3(x)

Since the two selected routing weights sum to one,

w_1E_1(x)+w_2E_2(x)=E_{\text{dense}}(x)

This allows the model to gain additional expert capacity **without throwing away the function learned by the dense model**.

---



## Conversion Audit

I explicitly compared the trained dense model with the newly created MoE **before performing any MoE optimizer step**.


| Measurement                           | Result           |
| ------------------------------------- | ---------------- |
| Dense loss before conversion          | 0.00019579       |
| MoE loss immediately after conversion | 0.00019579       |
| Absolute loss difference              | **2.33 × 10⁻¹⁰** |
| Maximum logit difference              | **4.20 × 10⁻⁵**  |


The effectively zero loss discontinuity shows that the dense → MoE conversion preserved the learned model function.

---



# Load Balancing

A new MoE router can collapse by sending most tokens to only a few experts.

To discourage this, I used the Switch-style auxiliary balancing term:

L_{\text{balance}}

N\sum_{i=1}^{N}f_iP_i

where:

- N is the number of experts;
- f_i is the fraction of routed assignments received by expert i;
- P_i is the average router probability for expert i.

The complete training objective is:

L

L_{\text{LM}}
+
0.001L_{\text{balance}}
+
0.001L_z

I also log expert utilization and **MaxVio** instead of assuming that the balancing term works.

Routing is **dropless** in this implementation: there is no expert-capacity cutoff, so every token selected for an expert is processed.

---



# MoE Training

After conversion, I continued training the model for another **5.001M peS2o tokens**.

Training loss continued to fall:

```text
first 100 MoE steps : 0.513094
last 100 MoE steps  : 0.301471
```

The fixed validation loss also improved:

```text
MoE start : 0.428346
MoE final : 0.308779
```

Therefore, the conversion did not merely create a valid MoE architecture—the MoE continued learning after upcycling.

---



# Measured Results

Hardware: **Tesla T4**  
Precision: **torch.float16**  
Dataset: **allenai/peS2o v2**  
Tokenizer: **8,192 WordPiece + FlashTokenizer**


| Phase | Tokens | First-100 Train Loss | Last-100 Train Loss | Fixed Val Loss | token/s | Peak GPU Memory |
| ----- | ------ | -------------------- | ------------------- | -------------- | ------- | --------------- |
| Dense | 5.001M | 3.058409             | 0.441072            | **0.428352**   | 50,402  | 1.29 GiB        |
| MoE   | 5.001M | 0.513094             | 0.301471            | **0.308779**   | 29,434  | 2.49 GiB        |


Overall, the experiment processed approximately **10M training tokens**.

---



# Total vs Active Parameters

The dense model contains:

```text
19,867,008 parameters
```

After adding four experts per transformer layer:

```text
MoE total parameters  = 48,196,992
MoE active parameters ≈ 29,322,624 per token
```

So the MoE has approximately:

```text
2.43× the total parameters of the dense model
```

while only about:

```text
1.48× the dense parameter count is active for a token
```

This illustrates the main motivation for MoE: **model capacity can increase faster than per-token active computation**.

---



# Expert Behavior

The final average routing loads were:


| Expert   | Mean Load |
| -------- | --------- |
| Expert 0 | 23.27%    |
| Expert 1 | 23.73%    |
| Expert 2 | 25.32%    |
| Expert 3 | 27.68%    |


Perfectly even routing would give each expert approximately **25%** of assignments.

The aggregate utilization is therefore reasonably balanced, although routing was **not perfectly balanced**:

```text
Final mean MaxVio: 0.672
Dead layer/expert pairs on fixed evaluation: 5
```

This is an important result rather than something to hide. The auxiliary loss pushed the overall expert utilization toward a balanced distribution, but some individual layer/expert combinations were still unused on the final fixed evaluation.

A larger model or longer run could explore whether those experts eventually receive useful traffic.

---



# Did the Experts Actually Become Different?

At conversion, all experts are exact copies:

```text
mean expert divergence = 0.00000000
```

After MoE training:

```text
mean expert divergence = 0.00350762
```

This shows that the cloned experts received different routed gradients and began to differentiate.

This metric does **not by itself prove semantic specialization**, but it confirms that the experts did not remain identical copies.

---



# Quality Improvement

The dense model finished with fixed validation loss:

```text
0.428352
```

The MoE finished at:

```text
0.308779
```

That is approximately a **27.9% lower validation loss** after the additional MoE training.

The assignment's main requirement is therefore satisfied:

```text
Dense:
3.058409 → 0.441072

Dense → MoE conversion:
loss discontinuity ≈ 2.33e-10

MoE:
0.513094 → 0.301471
```

Both phases learned, and the model continued improving after conversion to MoE.

---



# Compute and Memory Trade-off

The MoE increased capacity, but this simple implementation was more expensive to run.

### Throughput

```text
Dense : 50,402 token/s
MoE   : 29,434 token/s
```

The teaching implementation was about **42% slower** after conversion.

### Peak GPU Memory

```text
Dense : 1.29 GiB
MoE   : 2.49 GiB
```

The higher memory usage is expected because all four experts must still be stored on the single GPU even though only two are active for a particular token.

The implementation also dispatches experts using ordinary PyTorch indexing. Production MoE systems typically use grouped GEMMs, block-sparse kernels and expert parallelism, so these throughput numbers should **not** be interpreted as representative of optimized production MoE performance.

---



# What I Learned

The main thing I learned from this assignment is that converting a dense model to an MoE is not simply a matter of adding several FFNs.

Three distinct problems have to be handled:

**1. Preserve what the dense model already learned.**  
Sparse upcycling solved this by cloning the trained FFN weights. The near-zero conversion loss discontinuity demonstrated that the learned function was preserved.

**2. Decide which expert each token should use.**  
The router learned a token-dependent top-2 assignment rather than running every expert.

**3. Prevent routing collapse.**  
The auxiliary balancing objective kept aggregate expert utilization reasonably close to uniform, although the remaining dead layer/expert pairs show that load balancing is still a real training problem. A lower top-K from a larger pool of experts will liikely tend to become imbalanced more than a top-2 from 4 experts experiment run here.

The experiment also made the distinction between **total parameters** and **active parameters** concrete. The MoE stored 48.2M parameters, while only about 29.3M were active for an individual token.

Most importantly, the model did not stop learning after the structural change:

> **I trained a dense model, converted its learned FFNs into experts without losing the learned function, and then continued training the resulting MoE to a lower loss.**

---



# Files

```text
session14_dense_to_moe_pes2o_colab.ipynb
README.md
README_RESULTS.md
session14_results.json
dense_to_moe_loss.png
router_maxvio.png
final_expert_load.png
```

The submitted notebook contains the full tokenizer preparation, peS2o preprocessing, dense training, MoE conversion, conversion audit, MoE continuation, routing diagnostics and result-generation code.

---

