# V5 Model Training Curriculum

# Mixture and Curriculum Specification

> **Status:** testable design hypothesis, not a claimed training result. The 1B and 3B proxy runs are specified as the testing methodology but have **not** been executed.

## 1. Decision and scope

Proposed V5 model is a **120B param dense, India-first model** trained with a provisional **196,608-token vocabulary** and a curriculum that reaches **256K context**.

The model will be optimised for:

- Coding and repository-scale software engineering.
- Long-horizon agentic work involving planning, tool calls, observations, failures and recovery.
- Controllable reasoning at low, medium, high and ultra effort.
- Native understanding and generation across Indic languages.



### Training token requirements

Using the Chinchilla compute-optimal heuristic of approximately 20 training tokens per parameter, a 120B dense model corresponds to a 2.4T-token reference point. This is not a minimum or a guarantee of optimal downstream performance; the proposed 3B token-extension ablation tests whether V5 remains data-limited at that point. The proposed token budget target in the ERAv5 sessions is 10T-30T range. Hence, the base-model training budget *can be* fixed at 24 trillion tokens, or approximately 200 training tokens per parameter.

The course target of 10–30T tokens would be possible only if the scarce lanes could be filled without hiding extreme repetition or unverified generation. At 24T, even a 10% Indic-language lane would require 2.4T tokens. Sangraha reports roughly 251B total tokens, of which only about 64B are verified, while IndicCorpV2 is reported at roughly 21B tokens. After cross-source deduplication, licence filtering and V5 tokenisation, a 24T run would force several passes plus very large synthetic expansion before the model has demonstrated that those extra tokens are useful.

Hence, I'm choosing a 2.4T budget total. A proxy budget ablation is allowed to reject this decision if models remain strongly data-limited at the 20-token-per-parameter point

### Token accounting


| Budget component         | Tokens   | Purpose                                                                  |
| ------------------------ | -------- | ------------------------------------------------------------------------ |
| Unique-equivalent data   | **2T**   | Clean human-origin data plus unique provenance-stamped synthetic records |
| Controlled replay        | **280B** | Bounded reuse after the foundation and capability-building phases        |
| Protected anneal reserve | **120B** | Highest-quality data held back until learning-rate cooldown              |
| **Total**                | **2.4T** | Base pretraining plus final annealing                                    |


This budget excludes later supervised fine-tuning, preference optimisation and reinforcement learning. Those stages use different data shapes, masks and reward signals.

**Mixture thesis:** compared with an inventory-proportional baseline, this blend should improve the weighted geometric mean of coding, agentic, reasoning, India/Indic and long-context pillars without a material general-knowledge regression. Every allocation remains provisional until the proxy runs pass.

## 2. Full-run capability mixture

Primary capability lanes are mutually exclusive and sum to 100%. Language, provenance, difficulty, context length and reasoning effort are secondary tags.

An India-First model will require focused India-First datasets, hence **the proposal is to separate Indic data into two lanes: India-first institutional/regional knowledge, and Indic language foundation**. Indic datasets already used or considered in v4 (Sangraha, Samanantar) can generally be considered under the Indic language foundation lane. Supporting a separate India-First lane will require additional data collection, generation, and/or separation from existing general Indic datasets, which may or may not be feasible.


| Capability lane                                  | Share    | Full tokens | Main unique-equivalent | Controlled replay | Anneal   | Primary outcome                                                        |
| ------------------------------------------------ | -------- | ----------- | ---------------------- | ----------------- | -------- | ---------------------------------------------------------------------- |
| General web and reference                        | **25%**  | **600B**    | 598.8B                 | 0                 | 1.2B     | Broad knowledge and reference use                                      |
| India-first institutional and regional knowledge | **5%**   | **120B**    | 96.4B                  | 20B               | 3.6B     | Indian law, policy, education, economy, culture and regional knowledge |
| Indic-language foundation                        | **10%**  | **240B**    | 195.6B                 | 30B               | 14.4B    | Native generation, comprehension, translation and code-switching       |
| Code and software engineering                    | **24%**  | **576B**    | 482.4B                 | 60B               | 33.6B    | Generation, debugging, patching, tests and repository work             |
| Mathematics                                      | **7%**   | **168B**    | 121.0B                 | 35B               | 12.0B    | Calculation, proof and verified problem solving                        |
| Science and technical text                       | **5%**   | **120B**    | 100.2B                 | 15B               | 4.8B     | Scientific reasoning, manuals and technical knowledge                  |
| Explicit reasoning                               | **8%**   | **192B**    | 120.6B                 | 45B               | 26.4B    | Verifiable and controllable reasoning depth                            |
| Agentic and tool use                             | **6%**   | **144B**    | 94.6B                  | 35B               | 14.4B    | Planning, tool use, recovery and completion                            |
| Long-context learning                            | **6%**   | **144B**    | 110.6B                 | 25B               | 8.4B     | Cross-document and repository-scale dependencies                       |
| Books, dialogue and date-stamped news            | **4%**   | **96B**     | 79.8B                  | 15B               | 1.2B     | Long-form fluency, dialogue and temporal grounding                     |
| **Total**                                        | **100%** | **2.400T**  | **2.000T**             | **280B**          | **120B** |                                                                        |




### Why these proportions

**General web receives 25%.** At 2.4T, 600B high-quality general tokens can be supplied without repeated dependence on one web corpus. The lane remains large enough to protect breadth but is no longer allowed to dominate the model.

**Code** receives 24% because coding is a primary product capability rather than a secondary benchmark. This lane includes source code, tests, repository structure, issues, pull requests, documentation, build files, execution output and code-repair records.

The Stack v2 reports approximately 900B training tokens. It requires additional permissively licensed repositories, repository-level history, executable synthetic code, tests and carefully bounded replay.

**Supplementary options for Code:**

- HuggingFaceCode/stack-v3-train: about 4.9T tokens, 173M repositories and 713 languages. Licensing check required to understand actual usable token quantity. It is a Github snapshot.
- bigcode/commitpack: A 4 TB collection of permissively licensed GitHub commits across 350 programming languages. CommitPack is particularly valuable for code editing, bug fixing, patch generation, instruction-to-diff learning, understanding how repositories change.

**Agentic data** receives 6%, despite severe supply constraints, because the model must learn to plan, act, observe, recover and continue. This lane is not filled by ordinary question-answer pairs.

Public inventories illustrate the scarcity:

- SWE-Gym has 2,438 software-engineering instances.
- SWE-smith reports roughly 50,000 executable training tasks.
- ToolBench reports 126,486 instances involving 16,464 APIs.
- AgentTrek reports 10,398 successful web trajectories across 127 websites.

Therefore, most of the 144B agentic allocation must be built rather than collected. Session 5 identifies this supply problem and requires the plan to distinguish unique data, repeated data and synthetic generation. **Additional sources to consider**: AgentTrove, ToolMind, API-Bank, ToolACE, WebLINX.

**Explicit reasoning** receives 8%, but reasoning data is not treated as one uniform pile. It is divided by domain, verification method and effort length Sources to consider: OpenThoughts3-1.2M, Nemotron-Cascade-SFT Stage 2 (filtering needed), Superior-Reasoning-SFT-gpt-oss-120b.

**Indic foundation** receives 10%, with an additional 5% **India-first knowledge** lane. This makes India-first capability a structural property of the corpus rather than an instruction-tuning patch applied after English-heavy pretraining.

**Long context** receives 6%, because a model cannot learn long-horizon operation merely by increasing the inference context window. It must train on sequences where information from distant parts of the context is genuinely required.

## 3. Inventory map


| Lane         | Candidate inventory                                                                                                                |
| India-first  | India Code, gazettes, Parliament and court material, RBI, SEBI, MOSPI, NCERT, NIOS, NPTEL, state portals, regional books and press |
| Indic        | Sangraha, IndicCorpV2, fully cleaned Samanantar, BPCC/IndicTrans2-family data, licensed books and transcripts                      |
| Code         | The Stack v3/v2, permissive repositories, CommitPack, issues, pull requests, tests, CI logs, docs and executable synthetic code    |
| Mathematics  | FineMath, InfiWebMath, OpenWebMath, proof corpora, textbooks and verifier-generated problems                                       |
| Science      | peS2o, openly licensed papers, textbooks, manuals and standards                                                                    |
| Reasoning    | Worked proofs, OpenR1-Math-220k, OpenThoughts3-1.2M, filtered reasoning SFT and verifier-generated traces                          |
| Agentic      | SWE-Gym, SWE-smith, ToolBench, AgentTrek, AgentTrove, API-Bank, ToolACE, WebLINX and generated environments                        |
| Long context | Coherent repositories, books, legal records, standards, papers and multi-document trajectories                                     |

### Suply verdict
| Lane           | Target |                                 Known headline supply | Defensible conclusion                                                                |
| -------------- | -----: | ----------------------------------------------------: | ------------------------------------------------------------------------------------ |
| General        |   600B |                            FineWeb-Edu is much larger | Feasible, but the 240B source-family cap requires at least three source families     |
| Code           |   576B |                                 Large headline supply | Feasible after licence filtering; no more than 240B can come from one source family  |
| Verified Indic |    96B | Approximately 64B Sangraha verified + 21B IndicCorpV2 | **At least 11B short before deduplication and filtering; actual gap will be larger** |
| Mathematics    |   168B |         Roughly tens of billions across named corpora | Requires bounded replay and verified generation                                      |
| Reasoning      |   192B |    Sample counts known, token supply not yet measured | Zero firm supply credit until V5 tokenization                                        |
| Agentic        |   144B |        Thousands to hundreds of thousands of examples | Majority must be generated; exact percentage pending trajectory tokenization         |
| Long context   |   144B |                                Raw documents abundant | Dependency-qualified token supply unmeasured                                         |


## 4. Indic slot: four-tier accounting

The **Indic-language foundation lane is 240B tokens (10% of the run)**. The separate 120B India-first lane is not hidden inside this table.


| Indic provenance tier      | Share of Indic lane | Share of full run | Tokens   | Admission rule                                                                                   | Supply consequence                                                                                                |
| -------------------------- | ------------------- | ----------------- | -------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| Verified human-origin      | **40%**             | **4.0%**          | **96B**  | Attributable native text or audited OCR/transcript with verified language, script and provenance | Sangraha verified plus deduplicated IndicCorpV2 can cover most of the target; the usable overlap must be measured |
| Unverified human-origin    | **10%**             | **1.0%**          | **24B**  | Likely human text with incomplete provenance; strict quality and sampling caps                   | Approximately one Sangraha-unverified-scale pass; no anneal admission                                             |
| Translated                 | **25%**             | **2.5%**          | **60B**  | Aligned, script-correct translation with entity/number preservation and low translationese       | Samanantar/BPCC supply is plausible only after full cleaning and V5 tokenisation                                  |
| Synthetic, non-translation | **25%**             | **2.5%**          | **60B**  | Native, Romanised or code-switched generation with teacher, prompt and verifier lineage          | Generation is explicit and capped; it never counts as verified native text                                        |
| **Total**                  | **100%**            | **10%**           | **240B** |                                                                                                  |                                                                                                                   |




### Indic supply gates

- Translated and synthetic records never count as verified native text.
- Sangraha’s combined synthetic material must be split into translated and non-translation synthetic records before sampling.
- Samanantar receives zero budget credit until corpus-wide semantic alignment, conflicting-target detection, exact/near deduplication, script-aware language validation, PII handling, benchmark decontamination and V5 tokenisation are complete.
- No single teacher may generate more than 25% of the synthetic tier.
- At least 5% of every priority language’s synthetic allocation receives native-speaker audit before admission.



### Language allocation inside the Indic lane


| Language group                                                                                                                           | Share   | Tokens    |
| ---------------------------------------------------------------------------------------------------------------------------------------- | ------- | --------- |
| Native-script priority languages: Hindi, Bengali, Marathi, Telugu, Tamil, Urdu, Gujarati, Kannada, Malayalam, Odia, Punjabi and Assamese | **80%** | **192B**  |
| Other scheduled Indian languages                                                                                                         | **12%** | **28.8B** |
| Genuine Romanised and code-switched Indic                                                                                                | **8%**  | **19.2B** |


Language minima do not override quality, licensing, contamination or source-pass gates.

## 5. Source-pass limits

The smaller budget permits stricter repetition control:

- default maximum: **two effective passes** per source family;
- maximum after a passed proxy exception: **four effective passes** for verified native Indic, executable code repair, replayable Tier-A agent trajectories and externally verified reasoning;
- maximum contribution from one source family: **10% of the full run (240B)**; and
- semantic duplicates across datasets count against the same pass budget.

Every exception records the source ID, repeated-token count, manifest hash, proxy result, memorisation checks and approval.

## 6. OPUS selector and protected always-on floor

OPUS may rank and reject candidates only within **88% of each main-run batch**. The remaining **12% is an always-on capability floor**:


| Protected lane            | Minimum batch share |
| ------------------------- | ------------------- |
| Indic-language foundation | **7%**              |
| India-first knowledge     | **1%**              |
| Explicit reasoning        | **2%**              |
| Agentic and tool use      | **2%**              |
| **Total floor**           | **12%**             |


The combined India/Indic floor remains **8%**, matching the protected Indic principle used in V4, while reasoning and agentic data receive separate protection. The floor is a lower bound, not the final target.

Selector rules:

- initial retention hypothesis: 40% of selector-eligible candidates;
- proxy direction balanced across general, code, India/Indic, reasoning, agentic and long-context pillars;
- per-language minima applied before global ranking;
- cumulative lane drift limited to one percentage point at phase boundaries;
- anneal records remain invisible before cooldown; and
- benchmark and sealed-evaluation records are never candidates.



## 7. Protected anneal reserve

The final **120B tokens (5% of training)** are immutable before cooldown.


| Anneal component                           | Reserve share | Tokens    |
| ------------------------------------------ | ------------- | --------- |
| Code with execution and tests              | **28%**       | **33.6B** |
| Verified reasoning                         | **22%**       | **26.4B** |
| Verified human-origin Indic                | **12%**       | **14.4B** |
| Replayable Tier-A agent trajectories       | **12%**       | **14.4B** |
| Mathematics with answer/proof verification | **10%**       | **12.0B** |
| Dependency-bearing long context            | **7%**        | **8.4B**  |
| Science and technical data                 | **4%**        | **4.8B**  |
| India-first institutional knowledge        | **3%**        | **3.6B**  |
| General reference                          | **1%**        | **1.2B**  |
| Books, dialogue and news                   | **1%**        | **1.2B**  |
| **Total**                                  | **100%**      | **120B**  |


The reserve excludes unverified Indic, unverifiable synthetic reasoning, random long-context concatenation, non-replayable trajectories, unresolved code licences and evaluation-derived prompts.

At least **70% of reserve tokens** require an external signal: tests, proof/answer checker, environment replay, citation validation or native-speaker audit.

## 8. Curriculum



### Phase 1 — General and Indic foundation

- **Interval:** 0–400B.
- **Budget:** 400B unique-equivalent tokens.
- **Context:** 8K.
- **Difficulty:** D0 and D1 dominate.


| Lane                | Share |
| ------------------- | ----- |
| General             | 36%   |
| India-first         | 7%    |
| Indic               | 15%   |
| Code                | 16%   |
| Mathematics         | 5%    |
| Science             | 5%    |
| Reasoning           | 4%    |
| Agentic             | 2%    |
| Long context        | 3%    |
| Books/dialogue/news | 7%    |


**Objective:** multilingual language modelling, native Indic fluency, broad knowledge, basic programming and mathematical notation. Only short reasoning and atomic tool formats appear. Long reasoning and Tier-A trajectories remain untouched.

### Phase 2 — Coding, reasoning and tool-use build

- **Interval:** 400B–1.2T.
- **Budget:** 800B unique-equivalent tokens.
- **Context:** 8K increasing to 32K.
- **Difficulty:** D1 and D2 dominate.


| Lane                | Share |
| ------------------- | ----- |
| General             | 25%   |
| India-first         | 5%    |
| Indic               | 10%   |
| Code                | 27%   |
| Mathematics         | 7%    |
| Science             | 5%    |
| Reasoning           | 8%    |
| Agentic             | 6%    |
| Long context        | 4%    |
| Books/dialogue/news | 3%    |


**Objective:** executable coding, repository structure, worked reasoning, short multi-step tool use, structured outputs and verification.

### Phase 3 — Agentic and long-context hardening

- **Interval:** 1.2T–2.28T.
- **Budget:** 800B new unique-equivalent tokens plus 280B controlled replay.
- **Context:** 32K increasing to 128K, with sampled 256K sequences.
- **Difficulty:** D2 and D3 dominate.


| Lane                | Share      |
| ------------------- | ---------- |
| General             | 23.592593% |
| India-first         | 4.481481%  |
| Indic               | 7.925926%  |
| Code                | 24.296296% |
| Mathematics         | 7.407407%  |
| Science             | 5.111111%  |
| Reasoning           | 7.925926%  |
| Agentic             | 6.814815%  |
| Long context        | 8.481481%  |
| Books/dialogue/news | 3.962963%  |


**Objective:** repository-scale work, browser and terminal operation, failed-call recovery, cross-document synthesis and sustained task state. The replay controller enforces source caps and cannot access the anneal reserve.

### Phase 4 — Final anneal

- **Interval:** 2.28T–2.40T.
- **Budget:** 120B protected tokens.
- **Context:** mostly 32K–128K, with diagnostic 256K sequences.
- **Learning rate:** continuous cooldown.
- **Selector:** exploration disabled.

**Objective:** concentrate verified code, reasoning, native Indic, agentic and long-context data when the model is ready to extract the greatest value.

The weighted average of Phases 1–3 equals the 2.28T main-run allocation. Adding Phase 4 produces the complete 2.4T mixture.

## 9. Stable mixture transitions

Mixtures are never changed in a hard step.


| Transition        | Linear blending window |
| ----------------- | ---------------------- |
| Phase 1 → Phase 2 | **4B tokens**          |
| Phase 2 → Phase 3 | **6B tokens**          |
| Phase 3 → anneal  | **8B tokens**          |


Embeddings remain trainable unless a proxy validates freezing.

Initial abort gates:

- warn when gradient norm exceeds 2× the preceding rolling median for ten steps;
- stop when it exceeds 4×;
- stop when token-normalised loss exceeds 1.15× baseline for fifty steps; and
- restore the previous checkpoint and lengthen the transition before resuming.

Monitor global and per-layer gradient norms, token-normalised loss, per-lane validation loss, language-specific loss, activation outliers and optimiser-state anomalies.

## 10. Difficulty bands

Difficulty is independent of sequence or reasoning length. Verbosity receives no credit.


| Band                         | Target  | Definition                                                          | Concrete example                                                                                       |
| ---------------------------- | ------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **D0 — Atomic**              | **25%** | One direct transformation or fact with straightforward verification | Translate a two-sentence vaccination notice into Marathi while preserving dates and dosage numbers     |
| **D1 — Structured**          | **35%** | Two to five dependent operations or one clear tool interaction      | Implement a GST slab calculator from a specification and pass supplied unit tests                      |
| **D2 — Compositional**       | **30%** | Multiple documents, tools or domains with intermediate checking     | Read an RBI circular and transaction log, diagnose a validation error and emit a compliant API payload |
| **D3 — Long-horizon/expert** | **10%** | Ambiguity, delayed feedback, recovery or proof-level depth          | Investigate a multi-file repository defect, recover from a failed command and produce a passing patch  |




## 11. Reasoning-effort bands

Reasoning is measured by useful retained solution length, not raw generation length.


| Effort     | Target  | Retained reasoning length | Concrete example                                                                                                             |
| ---------- | ------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Low**    | **30%** | ≤128 tokens               | Select the correct API and arguments, then state one validation check                                                        |
| **Medium** | **35%** | 129–512 tokens            | Solve a multi-step school algebra problem and independently verify it                                                        |
| **High**   | **25%** | 513–2,048 tokens          | Trace a failing unit test, isolate the defect, produce a patch and validate it                                               |
| **Ultra**  | **10%** | 2,049–32,768 tokens       | Complete a repository-and-browser investigation with failed calls, recovery, evidence synthesis and a replayable final state |


No more than **2 percentage points of all retained reasoning traces** may exceed 8,192 tokens until a proxy demonstrates value. Every High or Ultra record requires an answer checker, tests, proof verifier, simulator, replayable environment, citation check or independent judge with sampled human audit.

## 12. Agentic lane

The 144B agentic allocation is divided toward Codex-style work:


| Agentic sub-lane                               | Share   | Tokens    |
| ---------------------------------------------- | ------- | --------- |
| Repository and software-engineering agents     | **45%** | **64.8B** |
| API and structured tool calling                | **25%** | **36.0B** |
| Browser and GUI operation                      | **10%** | **14.4B** |
| Terminal, data and operating-system tasks      | **10%** | **14.4B** |
| Indian public-service and enterprise workflows | **10%** | **14.4B** |


A valid trajectory records the objective, plan, tool schema, environment state, model action, observation, updated plan, failures or branches, recovery, final state and verifier result.

### Agentic training mask

During trajectory SFT or annealing:

- user requests are context-only;
- tool observations are context-only; and
- loss applies to model plans, tool calls, recovery decisions and final answers.

Training on tool observations is forbidden because it teaches the model to imitate or fabricate tool output instead of calling the tool.

## 13. Long-context lane

The 144B long-context lane must contain genuine distant dependencies.


| Sequence band | Share   | Tokens    |
| ------------- | ------- | --------- |
| 8K–32K        | **40%** | **57.6B** |
| 32K–64K       | **30%** | **43.2B** |
| 64K–128K      | **20%** | **28.8B** |
| 128K–256K     | **10%** | **14.4B** |


Qualifying records include coherent repositories, multi-section legal questions, scientific synthesis across documents and trajectories whose final decision depends on early observations. Each record states why it cannot be solved reliably from a short local window. Random concatenation receives zero long-context credit.

## Training-stage boundaries

| Stage                       | Inside 2.4T? | Learning signal                                               |
| --------------------------- | -----------: | ------------------------------------------------------------- |
| Base pretraining            |          Yes | Full causal next-token loss                                   |
| Final annealing/midtraining |          Yes | Declare whether full causal loss or selective masking is used |
| Supervised fine-tuning      |           No | Response-only masking                                         |
| Reinforcement learning      |           No | Verifiable reward                                             |


### Base pretraining

- Ordinary next-token loss.
- Reasoning-rich documents, worked solutions and proofs.
- Agentic preparation through repositories, tool schemas, API documentation, issue–patch–test chains and short verified calls.



### Annealing and supervised fine-tuning

- Response-only masking.
- Full multi-step trajectories.
- Reasoning-effort labels.
- Heavy use of execution, proof and answer verification.



### Reinforcement learning

- Code execution and test outcomes.
- Mathematical and proof checkers.
- Environment completion and recovery success.
- Tool-call correctness under cost and step limits.

The 2.4T base run creates the capability substrate; it does not assume pretraining alone creates the final aligned coding agent.

## 14. Cleaning priorities

Cleaning follows the starvation map:

1. **Expand India-first verified data:** law, policy, education, public services, economy, regional material and institutional sources.
2. **Finish Samanantar at corpus scale:** semantic alignment, conflicting-target checks, global exact/near deduplication, script-aware validation, PII handling and FLORES/IN22 decontamination.
3. **Build executable agentic data:** preserve environment image, repository commit, tool schema, action, observation, failures, recovery and verifier outcome.
4. **Clean code at repository level:** resolve licences, remove secrets/generated files, deduplicate repositories, execute tests and preserve date-based holdouts.
5. **Verify reasoning:** recompute answers, execute code, run proof checkers, reject trace–answer contradictions, deduplicate templates and remove evaluation-derived prompts.
6. **Score long-context dependence:** reject random concatenation and records answerable from a short local span.

No proxy or full-run arm starts until every admitted shard has:

- deterministic shard and record IDs;
- source, acquisition date and licence status;
- cleaning-code hash and tokenizer version;
- token count, language, script, tier and capability tags;
- deduplication cluster and contamination status;
- synthetic teacher/prompt lineage where applicable; and
- verifier or environment outcome where applicable.

Each lane reports collected unique, generated unique, replayed and reserve tokens separately. A proxy candidate pool must be at least **1.5× the sampled requirement**, or its replay/generation factor must be declared before training.

## 15. Proxy experiments



### 1B recipe screen

Train a 1B dense decoder for **20B tokens per arm**, preserving the 20-token-per-parameter budget ratio. Use the same tokenizer, optimiser family, packing, context schedule, evaluation cadence and held-out manifests. Run one seed for every arm and rerun the top three with two additional seeds.


| Arm    | Change                                                                            | Hypothesis tested                                                       |
| ------ | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **P0** | Allocate by cleaned inventory supply                                              | Does deliberate capability allocation beat passive supply accounting?   |
| **P1** | Proposed mixture, 12% floor, reserve and warm transitions                         | Main hypothesis                                                         |
| **P2** | Remove the protected floor; selector controls 100%                                | Does the selector starve India/Indic, reasoning and agentic capability? |
| **P3** | Raise combined India/Indic from 15% to 20%, taken from general and code           | Is the 15% ceiling too low?                                             |
| **P4** | Indic tiers become 20% verified, 10% unverified, 45% translated and 25% synthetic | Does translation-heavy volume reduce native quality?                    |
| **P5** | Move reasoning and agentic allocations into general web and code                  | Do explicit scarce lanes buy measurable capability?                     |
| **P6** | Spend reserve data proportionally from the start                                  | Is late holdback better than early exposure?                            |
| **P7** | Use hard mixture shifts instead of warm transitions                               | Are gradual transitions required for stability?                         |




### 3B confirmation

Train the top two 1B recipes at **3B parameters and 60B tokens each**, using two seeds. Extend the winner to **90B tokens** to test whether the 20-token-per-parameter budget is undertraining the model.

A recipe is not recommended for 120B because it wins once at 1B.

### Acceptance criteria

P1 is accepted only when:

- weighted strategic-pillar geometric mean is at least **2% above P0 at 1B** and **1% above P0 at 3B**;
- no release pillar regresses by more than **1.5 absolute points or 3% relative**, whichever is stricter;
- P1 beats P2 by at least **3% relative** on the India/Indic–reasoning–agentic aggregate, with at most 1% general-knowledge loss;
- P1 beats P6 by at least **2%** on anneal-sensitive capabilities;
- 1B and 3B recipe rankings have **Spearman ρ ≥ 0.70**;
- warm transitions keep peak gradient norm at or below **2.5×** the preceding rolling median;
- transition loss stays at or below **1.15×** baseline and recovers within 500 optimiser steps; and
- every result reports seed spread, token-normalised loss and exact manifest hashes.



### Budget decision rule

The 2.4T budget hypothesis is rejected or escalated for review if extending the 3B winner from 60B to 90B tokens produces:

- at least **1% additional weighted strategic-pillar gain**;
- improvement in at least four of coding, agentic, reasoning, India/Indic and long-context evaluation; and
- no unacceptable increase in memorisation or replay dependence.

If the extension does not clear those gates, the 2.4T budget remains the recommended full-run stopping point.

## 16. Evaluation pillars


| Pillar            | Core measurements                                                                                                                           |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Coding            | HumanEval+/MBPP+, LiveCodeBench, sealed repository tasks, patch correctness, test pass rate and regression rate                             |
| Agentic           | SWE-bench Verified, function calling, browser/terminal completion, argument validity, failed-call recovery and completion under cost limits |
| Reasoning         | Verifier-confirmed correctness, effort calibration, accuracy per reasoning token and overthinking rate                                      |
| India/Indic       | Per-language perplexity, native generation, translation, code-switching, Indian knowledge and native-speaker evaluation                     |
| Long context      | RULER/LongBench-style tests, repository dependencies, recall by position, 256K completion and an 8K-truncation comparison                   |
| General knowledge | Held-out web/reference perplexity and broad knowledge evaluations                                                                           |


No priority Indic language is hidden inside an aggregate score. Evaluation manifests are sealed before data collection and never enter training.

## Final commitment

V5 will train a **120B dense India-first model on a provisional 2.4T-token curriculum** with:

- **24% code**;
- **6% explicit agentic data**;
- **8% explicit reasoning**;
- **15% combined India-first and Indic data**, consisting of 5% India-first knowledge and 10% Indic-language foundation;
- **6% dependency-bearing long context**;
- a **12% protected always-on floor**;
- a **120B immutable anneal reserve**;
- gradual, monitored mixture transitions; and
- 1B and 3B proxy experiments capable of rejecting both the mixture and the total token budget.

**Every percentage is a hypothesis until an ablation demonstrates that the capability gained is worth the tokens displaced.**

## Research anchors

- Chinchilla scaling: [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)
- FineWeb-Edu: [https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- Sangraha: [https://huggingface.co/datasets/ai4bharat/sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)
- IndicCorpV2: [https://huggingface.co/datasets/ai4bharat/IndicCorpV2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2)
- Samanantar: [https://huggingface.co/datasets/ai4bharat/samanantar](https://huggingface.co/datasets/ai4bharat/samanantar)
- The Stack v3: [https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train](https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train)
- CommitPack: [https://huggingface.co/datasets/bigcode/commitpack](https://huggingface.co/datasets/bigcode/commitpack)
- FineMath: [https://huggingface.co/datasets/HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath)
- OpenWebMath: [https://huggingface.co/datasets/open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math)
- OpenR1-Math-220k: [https://huggingface.co/datasets/open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- OpenThoughts3-1.2M: [https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)
- SWE-Gym: [https://huggingface.co/datasets/SWE-Gym/SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym)
- SWE-smith: [https://huggingface.co/datasets/SWE-bench/SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith)
- ToolBench: [https://github.com/OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench)
- AgentTrek: [https://agenttrek.github.io/](https://agenttrek.github.io/)
- LongAlign: [https://arxiv.org/abs/2401.18058](https://arxiv.org/abs/2401.18058)
- ProLong: [https://arxiv.org/abs/2410.02660](https://arxiv.org/abs/2410.02660)

