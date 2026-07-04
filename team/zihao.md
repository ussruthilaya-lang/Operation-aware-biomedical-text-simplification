# Zihao — Evaluation Metrics & Model Baselines

**Repo:** https://github.com/ussruthilaya-lang/Operation-aware-biomedical-text-simplification
**Branch:** `feature/zihao-biobert-evaluation` (merged to main)
**Role:** SARI/FKGL/Compression Metrics, Prelim Evaluation Harness, Results/Conclusion Sections

---

## STATUS

| Phase | Status |
|---|---|
| Prelim Report | ✅ Delivered — reviewed, integrated into `paper/acl_latex.tex` |
| Full Paper | 🔄 In progress |

---

## WHAT'S BUILT — Prelim Phase Complete

### Core Metrics (`src/evaluation/metrics.py`)

```python
compute_corpus_sari(sources, hypotheses, references_list)
compute_fkgl(text)
compute_compression_ratio(source, hypothesis)
```

Uses `easse` for SARI — the field-standard implementation, not a hand-rolled version, so results are directly comparable to every other PLABA/TSAR paper cited in the report rather than risking small tokenization or n-gram-handling differences that would make numbers non-comparable. `textstat` for FKGL, same reasoning.

### Prelim Evaluation Harness (`src/evaluation/run_evaluation.py`)

```python
BASELINES = [
    ('baseline1_no_simplification', b1_simplify),
    ('baseline2_rule_based_chv', b2_simplify),
    ('baseline3_direct_llm', b3_simplify),
]
```

Registry-style loop over baselines rather than hardcoded separate calls — adding a new system to evaluate means adding one line to the list, not touching five different places in the file. This is the same pattern production ML eval harnesses use when benchmarking multiple model checkpoints or prompt variants against a fixed metric suite.

**Real bug found and fixed during integration:** the original version built its validation sample by reading `val.csv`'s `input_text` column directly and treating each row as one sentence. That column is actually **abstract-level** (mean 230 words, up to 822) — not sentence-level. This was invisible for baseline1 (no-op) and baseline2 (word-level substitution), since neither is sensitive to input length. It became catastrophically visible for baseline3: given a 200-word block under a "simplify this sentence" prompt, FLAN-T5 pattern-matched to title/abstract-generation instead of patient-facing simplification, producing a compression ratio of 0.137 and warning preservation of 0.176 — numbers that looked like a dramatic safety finding but were actually a data pipeline bug. Root-caused by comparing this run against Sophakotra's original 30-sentence result (0.950 preservation on genuinely sentence-level data) and finding a 5x unexplained gap, which triggered the investigation.

**Fix:** `run_evaluation.py` now builds its sample using the same sentence-level builder (`pseudo_labeler.build_pairs_from_plaba_json()`) that Sophakotra's classifier already used correctly — establishing one shared, correct definition of "a validation sample" across the codebase instead of two files independently (and inconsistently) defining it.

**Corrected prelim numbers, random 50-sentence sample (seed=42), post-fix:**

| System | SARI↑ | FKGL↓ | Compression | Warning Pres. |
|---|---|---|---|---|
| No simplification | 16.82 | 15.31 | 1.00 | 1.000 |
| Rule-based CHV | 18.66 | 15.17 | 1.02 | 1.000 |
| Direct LLM (FLAN-T5) | 22.28 | 14.89 | 0.945 | 1.000 |

**A second, subtler finding surfaced by this fix:** all three baselines now show *perfect* warning preservation on the random sample — apparently contradicting Sophakotra's original 0.950 finding on her 30-sentence warning-bearing stress test. Both numbers are correct; they are not measuring the same thing. A random 50-sentence sample has low statistical power to catch a rare event (only 33.1% of sentences contain warnings, and an even smaller fraction of those get dropped), so it can miss a real, rare safety failure that a stratified stress test reliably catches. Reported **both** numbers in the paper rather than picking the more convenient one — this became one of the report's genuine methodological contributions: safety-critical NLP evaluation needs stratified sampling, not random sampling alone, to reliably detect rare-but-critical failures.

### Results and Conclusion Sections

Results section reports the random-sample table above alongside Sophakotra's stratified stress-test table, explicitly framing the discrepancy as a finding rather than an inconsistency to hide. Conclusion restates the diagnostic hypothesis and lays out the three concrete additions needed for the full paper (contextual classifiers, full pipeline evaluation, extended safety metrics).

---

## LITERATURE REVIEW — Contributed Citations

Two citations from the original six were found to be duplicated against the same source paper during integration and corrected — `layscifive` and `bartwcts` had both been accidentally pointed at Li et al. 2024's general survey paper rather than distinct source papers; consolidated into a single correct citation (Li et al. 2024, BeeManc) since that is in fact where both checkpoints originate.

1. **Knappich et al. 2023 — BoschAI (PLABA 2023 winner).** Defines the SARI ceiling the full paper's neural baselines aim to compare against; their edit-weighted loss function for combating conservative copying is directly relevant to why our diagnostic framing matters.
2. **Lewis et al. 2020 — BART.** Backbone architecture underlying most PLABA systems, including the BART-w-CTs baseline planned for the full paper; comparing against it isolates how much of the operation-aware system's improvement comes from the diagnostic step versus generic seq2seq fine-tuning.
3. **Alva-Manchego et al. 2021 — (Un)Suitability of Automatic Metrics.** SARI and FKGL correlate only moderately with human judgment — motivates why the team's custom safety-preservation metrics are necessary rather than optional.
4. **Cripwell et al. 2022 — Controllable Sentence Simplification via Operation Classification.** Direct prior art for predict-then-execute operation classification; the same structural intuition extended here to the biomedical domain and span level.
5. **Devaraj et al. 2021 — Paragraph-level Simplification of Medical Texts.** Identifies safety-critical information loss as a recurring failure mode in medical simplification specifically — direct biomedical grounding for why the preservation metrics matter.

---

## KEY ENGINEERING DECISIONS

| Decision | Why |
|---|---|
| Registry-style `BASELINES` list instead of hardcoded per-system calls | Adding a new system to the evaluation means one line, not five separate edits — the same pattern real ML eval infrastructure uses |
| `easse` and `textstat` instead of hand-rolled metrics | Standardized implementations mean results are directly comparable to every cited PLABA/TSAR paper, not vulnerable to small tokenization differences |
| Reused `pseudo_labeler`'s sentence-level builder instead of reading `val.csv` directly | Fixes the granularity bug at its root by establishing one shared definition of "a validation sample," rather than patching the symptom in one place while leaving the same risk in every other file that might read `val.csv` directly in the future |
| Reported both random-sample and stratified-stress-test warning numbers, not just one | Picking the more convenient number would have been dishonest; reporting the discrepancy turned a confusing result into a genuine methodological finding |

---

## FULL PAPER — ACTIVE WORK

### Remaining tasks

1. **Lay-SciFive inference** (`src/models/layscifive_inference.py`) — BeeManc checkpoint, expected to score highest SARI but lowest safety preservation if the paper's central hypothesis holds.
2. **BART-w-CTs inference** (`src/models/bart_inference.py`) — second BeeManc-derived neural baseline.
3. **BioBERT fine-tuning** (`src/models/biobert_classifier.py`) — coordinate with Sophakotra since this extends her classifier progression; use Google Colab's free T4 GPU if no local GPU access.
4. **Full test-set evaluation** — re-run the harness on the complete 148-sentence test set, not the 50/30-sample prelim figures, and apply the same stratified-sampling discipline established in the prelim (a random sample AND a warning-bearing stress-test sample, reported separately, for every system).
5. **Entity and numerical preservation metrics** — coordinate with Sruthilaya, who owns the underlying detectors these metrics extend.

### Team dependencies I'm tracking

- Rishabh's readability-vs-safety scatter plot is blocked on this full test-set evaluation landing.
- The final results table needs all six systems (baseline1/2/3, Lay-SciFive, BART-w-CTs, operation-aware) run through the exact same harness for a fair comparison — no system gets evaluated on a different sample or sampling method than the others.

### Standing rule adopted from prelim
Before trusting any generation-based evaluation number, verify the actual model outputs on a handful of examples by hand — a metric contradiction (like high SARI paired with catastrophic warning loss) is a signal to investigate the pipeline, not a result to report at face value. This exact discipline is what caught the granularity bug before it reached the paper.