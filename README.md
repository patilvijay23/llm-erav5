# llm-erav5
repo for course work

# V5 Model Training Curriculum

# Mixture and Curriculum Specification

> **Status:** testable design hypothesis, not a claimed training result. The 1B and 3B proxy runs are specified as the testing methodology but have **not** been executed.

## 1. Decision and scope

Proposed V5 model is a **120B param dense, India-first model** trained with a **196,608-token vocabulary** and a curriculum that reaches **256K context**.

The model will be optimised for:

- Coding and repository-scale software engineering.
- Long-horizon agentic work involving planning, tool calls, observations, failures and recovery.
- Controllable reasoning at low, medium, high and ultra effort.
- Native understanding and generation across Indic languages.

### Training token requirements
120B param model means a minimum of 2.4T tokens as per Chinchilla scaling law. The proposed token budget target in the ERAv5 sessions is 10T-30T range. Hence, the base-model training budget is fixed at 24 trillion tokens, or approximately 200 training tokens per parameter.

The 24T point is chosen instead of immediately committing to the 30T ceiling because the limiting factor is not raw web availability. It is the availability of sufficiently clean and licensed native Indic, agentic, reasoning, and long-context data.


### Token accounting
| Budget component | Tokens | Purpose |
|---|---:|---|
| Unique-equivalent data | **20.0T** | Clean human-origin data plus unique provenance-stamped synthetic records |
| Controlled replay | **2.8T** | Bounded reuse after the foundation and capability-building phases |
| Protected anneal reserve | **1.2T** | Highest-quality data held back until learning-rate cooldown |
| **Total** | **24.0T** | Base pretraining plus final annealing |

The 24T budget does not include later supervised fine-tuning, preference optimisation or reinforcement learning. Those stages use different data formats, masks and reward signals.

**Mixture thesis:** compared with an inventory-proportional baseline, the proposed blend should improve the geometric mean of coding, agentic, reasoning, Indic and long-context pillars without causing a material regression in general knowledge. Every percentage below remains provisional until the 1B screen and 3B confirmation tests pass.

## Full-run capability mixture

Primary capability lanes are mutually exclusive and sum to 100%. Language, provenance, difficulty, context length and reasoning effort are secondary tags. A Hindi mathematical proof belongs to the mathematics or reasoning lane once while also counting toward Hindi coverage.

| Capability lane | Full-run share | Full tokens | Main-run tokens | Anneal tokens | Primary outcomes |
|---|---:|---:|---:|---:|---|
| General web and reference | **17%** | 4.080T | 4.068T | 12B | Broad knowledge, reference use and held-out web perplexity |
| India-first institutional and regional knowledge | **10%** | 2.400T | 2.352T | 48B | Indian law, policy, education, economy, culture and regional knowledge |
| Indic language foundation | **18%** | 4.320T | 4.128T | 192B | Native generation, comprehension, translation and code-switching |
| Code and software engineering | **20%** | 4.800T | 4.512T | 288B | Code generation, debugging, patching, tests and repository work |
| Mathematics | **6%** | 1.440T | 1.320T | 120B | Calculation, proof and verified mathematical problem solving |
| Science and technical text | **4%** | 960B | 912B | 48B | Scientific reasoning, manuals and technical knowledge |
| Explicit reasoning | **7%** | 1.680T | 1.440T | 240B | Verifiable, controllable low-to-ultra reasoning |
| Agentic and tool use | **5%** | 1.200T | 1.056T | 144B | Planning, tool selection, recovery and task completion |
| Long-context learning | **7%** | 1.680T | 1.584T | 96B | Cross-document and repository-scale dependency handling |
| Books, dialogue and date-stamped news | **6%** | 1.440T | 1.428T | 12B | Long-form fluency, dialogue and temporally grounded events |
| **Total** | **100%** | **24.000T** | **22.800T** | **1.200T** | |

