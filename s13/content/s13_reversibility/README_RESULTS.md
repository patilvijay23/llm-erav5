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