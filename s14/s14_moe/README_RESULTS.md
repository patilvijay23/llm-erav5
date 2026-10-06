## Measured results

Hardware: **Tesla T4**  
Precision: **torch.float16**  
Dataset: **allenai/peS2o v2**  
Tokenizer: **8,192 WordPiece + FlashTokenizer encoding**  
Dense parameters: **19,867,008**  
MoE total parameters: **48,196,992**  
MoE active parameters/token (approx.): **29,322,624**  
Experts per layer: **4**, top-k: **2**  

| phase | tokens | first-100 train loss | last-100 train loss | fixed val loss | token/s | peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| dense | 5.001M | 3.058409 | 0.441072 | 0.428352 | 50,402 | 1.29 |
| MoE | 5.001M | 0.513094 | 0.301471 | 0.308779 | 29,434 | 2.49 |

### Dense → MoE conversion audit
- dense loss immediately before conversion: `0.00019579`
- MoE loss immediately after conversion: `0.00019579`
- absolute conversion loss difference: `2.3283064e-10`
- maximum logit difference: `4.196167e-05`

### Router / expert behavior
- final mean MaxVio: **0.672**
- dead layer/expert pairs in final fixed evaluation: **5**
- mean expert loads: **E0=23.27%, E1=23.73%, E2=25.32%, E3=27.68%**
- mean expert weight divergence: **0.00000000 → 0.00350762**

### Assignment conclusion
- dense first-100 → last-100 loss: **3.058409 → 0.441072**
- MoE first-100 → last-100 LM loss: **0.513094 → 0.301471**
- conversion loss discontinuity: **2.33e-10**