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


| Budget component         | Tokens    | Purpose                                                                  |
| ------------------------ | --------- | ------------------------------------------------------------------------ |
| Unique-equivalent data   | **20.0T** | Clean human-origin data plus unique provenance-stamped synthetic records |
| Controlled replay        | **2.8T**  | Bounded reuse after the foundation and capability-building phases        |
| Protected anneal reserve | **1.2T**  | Highest-quality data held back until learning-rate cooldown              |
| **Total**                | **24.0T** | Base pretraining plus final annealing                                    |


The 24T budget does not include later supervised fine-tuning, preference optimisation or reinforcement learning. Those stages use different data formats, masks and reward signals.

**Mixture thesis:** compared with an inventory-proportional baseline, the proposed blend should improve the geometric mean of coding, agentic, reasoning, Indic and long-context pillars without causing a material regression in general knowledge. Every percentage below remains provisional until the 1B screen and 3B confirmation tests pass.

## Full-run capability mixture

Primary capability lanes are mutually exclusive and sum to 100%. Language, provenance, difficulty, context length and reasoning effort are secondary tags.

An India-First model will require focused India-First datasets, hence **the proposal is to separate Indic data into two lanes: India-first institutional/regional knowledge, and Indic language foundation**. Indic datasets already used or considered in v4 (Sangraha, Samanantar) can generally be considered under the Indic language foundation lane. Supporting a separate India-First lane will require additional data collection, generation, and/or separation from existing general Indic datasets, which may or may not be feasible. A 28% allocation to Indic dataset is extremely ambitious and hence must be grounded in actual data availability later.


| Capability lane                                  | Full-run share | Full tokens | Main-run tokens | Anneal tokens | Primary outcomes                                                       |
| ------------------------------------------------ | -------------- | ----------- | --------------- | ------------- | ---------------------------------------------------------------------- |
| General web and reference                        | **17%**        | 4.080T      | 4.068T          | 12B           | Broad knowledge, reference use and held-out web perplexity             |
| India-first institutional and regional knowledge | **10%**        | 2.400T      | 2.352T          | 48B           | Indian law, policy, education, economy, culture and regional knowledge |
| Indic language foundation                        | **18%**        | 4.320T      | 4.128T          | 192B          | Native generation, comprehension, translation and code-switching       |
| Code and software engineering                    | **20%**        | 4.800T      | 4.512T          | 288B          | Code generation, debugging, patching, tests and repository work        |
| Mathematics                                      | **6%**         | 1.440T      | 1.320T          | 120B          | Calculation, proof and verified mathematical problem solving           |
| Science and technical text                       | **4%**         | 960B        | 912B            | 48B           | Scientific reasoning, manuals and technical knowledge                  |
| Explicit reasoning                               | **7%**         | 1.680T      | 1.440T          | 240B          | Verifiable, controllable low-to-ultra reasoning                        |
| Agentic and tool use                             | **5%**         | 1.200T      | 1.056T          | 144B          | Planning, tool selection, recovery and task completion                 |
| Long-context learning                            | **7%**         | 1.680T      | 1.584T          | 96B           | Cross-document and repository-scale dependency handling                |
| Books, dialogue and date-stamped news            | **6%**         | 1.440T      | 1.428T          | 12B           | Long-form fluency, dialogue and temporally grounded events             |
| **Total**                                        | **100%**       | **24.000T** | **22.800T**     | **1.200T**    |                                                                        |




### Why these proportions

**Code** receives 20% because coding is a primary product capability rather than a secondary benchmark. This lane includes source code, tests, repository structure, issues, pull requests, documentation, build files, execution output and code-repair records.

The Stack v2 reports approximately 900B training tokens, so the 4.8T code target cannot be met by repeatedly sampling one public code corpus. It requires additional permissively licensed repositories, repository-level history, executable synthetic code, tests and carefully bounded replay.

**Supplementary options for Code:**

- HuggingFaceCode/stack-v3-train: about 4.9T tokens, 173M repositories and 713 languages. Licensing check required to understand actual usable token quantity. It is a Github snapshot.
- bigcode/commitpack: A 4 TB collection of permissively licensed GitHub commits across 350 programming languages. CommitPack is particularly valuable for code editing, bug fixing, patch generation, instruction-to-diff learning, understanding how repositories change.

**Agentic data** receives 5%, despite severe supply constraints, because the model must learn to plan, act, observe, recover and continue. This lane is not filled by ordinary question-answer pairs.

Public inventories illustrate the scarcity:

- SWE-Gym has 2,438 software-engineering instances.
- SWE-smith reports roughly 50,000 executable training tasks.
- ToolBench reports 126,486 instances involving 16,464 APIs.
- AgentTrek reports 10,398 successful web trajectories across 127 websites.

Therefore, most of the 1.2T agentic allocation must be built rather than collected. Session 5 identifies this supply problem and requires the plan to distinguish unique data, repeated data and synthetic generation. **Additional sources to consider**: AgentTrove, ToolMind, API-Bank, ToolACE, WebLINX.

**Explicit reasoning** receives 7%, but reasoning data is not treated as one uniform pile. It is divided by domain, verification method and effort length Sources to consider: OpenThoughts3-1.2M, Nemotron-Cascade-SFT Stage 2 (filtering needed), Superior-Reasoning-SFT-gpt-oss-120b.

**Indic foundation** receives 18%, with an additional 10% **India-first knowledge** lane. This makes India-first capability a structural property of the corpus rather than an instruction-tuning patch applied after English-heavy pretraining.

**Long context** receives 7%, because a model cannot learn long-horizon operation merely by increasing the inference context window. It must train on sequences where information from distant parts of the context is genuinely required.

## Inventory map


| Lane         | Candidate inventory                                                                                                        | Supply verdict                                                                           |
| ------------ | -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| General web  | FineWeb-Edu, FineWeb-Edu-score-2, licence-cleared reference pools, Wikipedia and open Common Pile components               | Large, but quality, deduplication and English dominance remain risks                     |
| India-first  | India Code, gazettes, Parliament and courts, RBI, SEBI, MOSPI, NCERT, NIOS, NPTEL, state portals, regional books and press | Severely starved after licence and OCR filtering. **Research to find datasets required** |
| Indic        | Sangraha, cleaned Samanantar, BPCC/IndicTrans2-family resources, licensed native books and transcripts                     | Verified native supply is much smaller than headline supply                              |
| Code         | The Stack v2, permissive repositories, issues, pull requests, tests, CI logs, documentation and executable synthetic code  | Raw supply is large; licences, secrets and repository deduplication bind                 |
| Mathematics  | FineMath, InfiWebMath, OpenWebMath, proof corpora, textbooks and verifier-generated problems                               | High-quality unique supply is well below target                                          |
| Science      | peS2o, openly licensed papers, textbooks, manuals and standards                                                            | Rights, extraction quality and citation integrity constrain supply                       |
| Reasoning    | Worked proofs, verified solutions, OpenR1-Math-220k, OpenThoughts-style records and verifier-generated traces              | Most long and difficult traces require verified generation                               |
| Agentic      | SWE-Gym, SWE-smith, ToolBench, AgentTrek-style data and generated browser/terminal/repository trajectories                 | Most starved lane; **much must be built rather than collected**                          |
| Long context | Coherent repositories, books, legal records, standards, papers and multi-document trajectories                             | Random concatenation is plentiful but receives no budget credit                          |



