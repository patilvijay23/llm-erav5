# Adam Optimizer Assignment

## Experimental setup

All GPU experiments were run with PyTorch on `Tesla T4`.

The main T4 model used for the update-ratio and scheduler experiments has:

- width: **2,048**
- linear depth: **6**
- input dimension: **256**
- output dimension: **64**
- batch size: **256**
- mixed precision: **True**

The learning-rate scaling experiment uses the assignment-required widths **256, 512, and 1,024**.

A/B comparisons use identical initialization, step-matched data, optimizer settings, and evaluation data. Cosine and WSD are tuned independently before the final comparison.

---

## 1. Adam reproduced by hand

For gradient `g_t`, Adam computes

`m_t = beta1*m_(t-1) + (1-beta1)*g_t`

`v_t = beta2*v_(t-1) + (1-beta2)*g_t^2`

with bias-corrected moments

`m_hat_t = m_t / (1-beta1^t)`

`v_hat_t = v_t / (1-beta2^t)`.

The parameter update is

`w_t = w_(t-1) - lr * m_hat_t / (sqrt(v_hat_t) + eps)`.

I used one scalar weight, initial value **1.5**, with gradients:

`[0.20, -0.10, 0.05, -0.30, 0.15]`

and Adam settings `lr=0.01`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`.

| t | g | m | v | m_hat | v_hat | step | weight after |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.20000 | 0.02000000 | 0.0000400000 | 0.20000000 | 0.0400000000 | 0.01000000 | 1.49000000 |
| 2 | -0.10000 | 0.00800000 | 0.0000499600 | 0.04210526 | 0.0249924962 | 0.00266337 | 1.48733663 |
| 3 | 0.05000 | 0.01220000 | 0.0000524100 | 0.04501845 | 0.0174874950 | 0.00340429 | 1.48393234 |
| 4 | -0.30000 | -0.01902000 | 0.0001423576 | -0.05530678 | 0.0356428361 | -0.00292949 | 1.48686183 |
| 5 | 0.15000 | -0.00211800 | 0.0001647153 | -0.00517203 | 0.0330090065 | -0.00028467 | 1.48714650 |

PyTorch reproduced the same moment values and final weights to floating-point precision. The largest weight disagreement across the five updates was **0.000e+00**.

---

## 2. Effect of disabling bias correction

I plotted the first twenty scalar Adam updates with and without bias correction.

I define “stops mattering” as the first step after which the relative update-magnitude difference is **below 1% and remains below 1%**.

Under that definition:

- relative difference at step 20: **524.09%**
- first persistent <1% step: **3925**

The result shows why bias correction is primarily an early-training correction but can remain numerically relevant much longer than a 20-step plot suggests, especially because `beta2=0.999` gives the second moment a long time constant.

Plots:

- `part2_weights.png`
- `part2_relative_difference_first20.png`

---

## 3. Update-to-weight ratio

For every parameter tensor I logged

`||Delta W||_2 / ||W||_2`

at every optimizer step.

The run uses linear warmup for **30 steps**, so the scheduler stops increasing the learning rate at **step 30**. The plot shows the same boundary in the parameter-update ratios. After that point, variation in the ratio comes from gradients, Adam's moving moments, and the changing parameter norms rather than from an increasing warmup LR.

Plot:

- `part3_update_weight_ratio.png`

Raw log:

- `part3_update_weight_ratio.csv`

---

## 4. Cosine vs WSD with an unexpected stop at step 200

Both schedules are designed for **300 steps**, but both runs are stopped at **step 200**.

The WSD schedule uses:

- warmup: first 30 steps
- stable phase: through step 240
- decay: final 60 planned steps

Therefore WSD has **not started its decay phase** when training is interrupted at step 200, while cosine has already been decaying.

Crucially, I tuned the peak LR separately for each schedule before comparing them.

Measured result:

| Schedule | Tuned peak LR | Eval loss at step 200 |
|---|---:|---:|
| Cosine | 0.0016 | 0.09294282 |
| WSD | 0.0008 | 0.10149142 |

**Model I would keep: cosine.**

That decision is based on the lower fixed-set evaluation loss after tuning both sides, rather than on an assumed schedule advantage.

Plots:

- `part4_schedule_shapes.png`
- `part4_loss_cosine_vs_wsd.png`
- `part4_lr_cosine_vs_wsd.png`

---

## 5. Learning-rate sweep across model width

I swept learning rate independently at widths **256, 512, and 1,024**.

Measured minima:

| Width | Best LR | Eval loss |
|---:|---:|---:|
| 256 | 0.0064 | 0.22051067 |
| 512 | 0.0016 | 0.20521576 |
| 1,024 | 0.0008 | 0.17589451 |

I fit an empirical power law

`lr*(width) = C * width^a`

through the three measured optima.

- fitted exponent `a`: **-1.5000**
- extrapolated starting LR at width 4,096: **8.90899e-05**
- confidence: **moderate**

**I would start width 4,096 at approximately 8.90899e-05.**

My confidence is **moderate** because this is an extrapolation from only three measured widths, not a direct width-4,096 sweep. Before committing an expensive training run I would perform a narrow local sweep around this predicted value.

All three minima are interior to the tested LR grid, which makes the trend more credible, but the 4,096 value is still an extrapolation.

Plot:

- `part5_lr_sweep.png`

---

## Fair-comparison checklist

1. Same initialization for paired A/B comparisons.
2. Same data at corresponding steps.
3. Same number of optimizer steps.
4. Same evaluation set and metric.
5. Same optimizer family and non-tested hyperparameters.
6. Cosine and WSD each receive independent LR tuning.
7. LR sweep minima are checked for boundary hits.
8. The width-4,096 value is reported as an extrapolated starting point, not as a directly measured optimum.

This matters because an optimizer or schedule claim is not meaningful if one side is tuned and the other is not.
