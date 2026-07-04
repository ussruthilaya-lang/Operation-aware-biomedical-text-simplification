# Sophakotra — Operation Classification & Generation Pipeline

**Repo:** https://github.com/ussruthilaya-lang/Operation-aware-biomedical-text-simplification
**Branch:** `feature/sophakotra-classifier` (merged to main)
**Role:** Pseudo-Labeling, Operation Classifier, Direct-LLM Baseline, Classifier/Baselines Sections

---

## STATUS

| Phase | Status |
|---|---|
| Prelim Report | ✅ Delivered — reviewed, integrated into `paper/acl_latex.tex` |
| Full Paper | 🔄 In progress |

---

## WHAT'S BUILT — Prelim Phase Complete

### Pseudo-Label Pipeline (`src/data/pseudo_labeler.py`)

Since PLABA has no public span-level operation labels, labels are derived by weak supervision from aligned source→target sentence pairs. A pure-Python Levenshtein distance implementation (zero external dependencies, so every teammate can run it without installing anything) computes normalized character-level edit distance between source and target spans.

```python
label_span(source_span, target_span)   # length-ratio rule + edit-distance tiebreak
label_sentence_pair(source, target)
label_dataset(df)
build_pairs_from_plaba_json(json_path, exclude_pmids)
```

**Labeling rule:** target ≥15% longer → Explanation; target ≤15% shorter → Generalization; similar length with sufficient character-level change → Substitution. Edit distance is the most direct measurable signal for what type of change was actually made — standard weakly supervised practice when gold labels are inaccessible.

**Leakage control:** `build_pairs_from_plaba_json()` explicitly excludes every validation and test PMID from the training pairs at the source level — not just row-level exclusion, but PMID-level, so no sentence from a held-out abstract leaks into training under a different adaptation. This produced **6,307 leakage-free sentence-level training pairs** (43.2% Substitution, 34.4% Explanation, 22.5% Generalization), sentence-aligned from `data.json` by iterating every available adaptation per abstract to maximize training volume.

**Correction made during integration:** the taxonomy was originally attributed to Heineman et al.'s SALSA framework. Closer reading of Ondov et al. 2025 (PLABA's own creators reflecting on the TREC track) revealed the actual source is PLABA's own original five-operation scheme (Attal et al. 2023) — SALSA is a general-purpose edit-evaluation framework, unrelated to PLABA's taxonomy. Fixed the citation and, in the process, found a genuinely valuable comparison point: Ondov et al. report the true operation distribution (Substitution 64.7%, Explanation 19.3%, Generalization 6.3%, Omission 9.1%, Exemplification 0.6%) and real baseline classifier performance on this exact task (human inter-annotator agreement 0.46 F1, reported baseline classifiers 0.33–0.39 macro-F1) — both now cited directly in the classifier section as external validation.

### TF-IDF + Logistic Regression Classifier (`src/classifier/tfidf_classifier.py`)

First rung of a deliberate three-model progression (TF-IDF → BERT-base → BioBERT), each isolating one variable: does surface lexical signal alone predict the operation, does general contextual pretraining help, does biomedical-domain pretraining help further.

```python
train(X_train, y_train)
predict(X_test)
evaluate(X_test, y_test)   # accuracy, macro-F1, per-class F1, confusion matrix
build_val_set()            # PMID-based split, matches pseudo_labeler's leakage control
```

Uses `(1,2)`-gram TF-IDF (`max_features=50000`) with class-weighted logistic regression to handle the minority Generalization class.

**Prelim result:** 0.431 accuracy, 0.424 macro-F1 on the held-out validation set (n=1,431) — roughly twice the macro-F1 of a majority-class baseline (~0.20), and within/slightly above the literature's reported range for this exact task once the Ondov comparison was found. The confusion matrix shows the strongest confusion is between **Substitution and Explanation**, not Explanation and Generalization as originally hypothesized in the proposal — suggesting TF-IDF features are dominated by the presence of jargon tokens (a Substitution signal) rather than sentence-level structural cues like added clauses (an Explanation signal). This is a more interesting, more honest finding than confirming the original hypothesis, and it directly motivates the contextual models next: BERT/BioBERT can represent structural cues bag-of-words features cannot.

**Flagged for full-paper fix (not blocking prelim):** `VECTORIZER` and `MODEL` are currently mutable module-level globals rather than encapsulated in a returned object (e.g., an sklearn `Pipeline`). This risks silent wrong results if `predict()` is called before `train()`, and isn't safe for parallel experiments. Needs refactoring before BERT-base/BioBERT are added, so three model versions don't share global state.

### Direct-LLM Baseline (`src/baselines/baseline3_direct_llm.py`)

Deliberately naive baseline — unguided prompting with no operation guidance, representing the "simplify everything uniformly" approach the whole system argues against. Built on **local, open-weight FLAN-T5-base** (not a paid API), so the baseline is fully reproducible by anyone who clones the repo. Model and tokenizer loaded once and cached; explicit seq2seq loading (not the deprecated `text2text-generation` pipeline) for version stability across transformers releases.

**Original prelim result (30-sentence warning-bearing stress test):** 0.950 warning preservation — the only baseline of the three that drops safety-critical content, versus 1.000 for both no-simplification and rule-based CHV substitution. This is the paper's central empirical motivation.

**Important finding surfaced during full-corpus verification:** re-running this baseline on a random 50-sentence sample (rather than the warning-stratified stress test) showed **all three baselines at perfect warning preservation** — an apparent contradiction. Root cause was two-fold: (1) a separate data pipeline bug where the evaluation script read abstract-level text instead of sentence-level text, since fixed; (2) after that fix, the discrepancy remains but is now understood correctly — random sampling has low statistical power to catch a rare event (only 33.1% of sentences contain warnings), so a small random sample can miss real, rare safety failures that a targeted stress test reliably catches. Both numbers are correct; they measure different things. This became a genuine methodological contribution of the paper: **safety-critical NLP evaluation needs stratified sampling, not just random sampling.**

---

## LITERATURE REVIEW — Contributed Citations

1. **Attal et al. 2023 — PLABA dataset (taxonomy source, corrected).** The original five-operation taxonomy our three-operation scope is drawn from — corrected from an earlier mistaken SALSA attribution.
2. **Ondov et al. 2025 — Lessons from the PLABA Track.** PLABA's own creators reflecting on the TREC track; source of the true operation distribution and a genuine external classifier benchmark our model is compared against.
3. **BERT (Devlin et al. 2019)** and **BioBERT (Lee et al. 2020)** — motivate the contextual and biomedical-domain classifier stages planned for the full paper.
4. **TSAR 2025 findings (Alva-Manchego et al.)** — field-level evidence that no PLABA-adjacent system classifies operation type before generating, situating the classifier as the genuine research contribution.

---

## KEY ENGINEERING DECISIONS

| Decision | Why |
|---|---|
| Pure-Python Levenshtein, zero dependencies | Every teammate can run the pseudo-labeler with no install step — removes a whole class of environment friction |
| PMID-level (not row-level) leakage exclusion | Row-level exclusion would still let sentences from the same abstract leak across train/val/test via a different adaptation; PMID-level is the correct unit of exclusion |
| Local FLAN-T5-base instead of a paid API for baseline3 | Fully reproducible by any teammate or grader with no API key dependency |
| Classifier sees source only, never target | At real inference time no target exists — using target-derived features would be leakage that doesn't reflect deployment conditions |
| Fixed `LABELS` order in evaluation | Confusion matrix rows/columns stay consistently interpretable across every run |

---

## FULL PAPER — ACTIVE WORK

### Remaining tasks

1. **Refactor `tfidf_classifier.py`'s mutable globals** into an explicit, stateless pipeline object before adding BERT-base/BioBERT — required so three model versions don't silently share state during comparison experiments.
2. **BERT-base fine-tuned classifier** — isolates the value of general contextual pretraining over TF-IDF's bag-of-words features.
3. **BioBERT fine-tuned classifier** (`dmis-lab/biobert-base-cased-v1.2`) — lr 2e-5, batch 16, 3–5 epochs, AdamW, class-weighted loss. Isolates the value of biomedical-domain pretraining specifically on top of general contextual pretraining.
4. **3×3 confusion matrix comparison** across TF-IDF, BERT-base, and BioBERT — the Explanation↔Substitution confusion found in TF-IDF is the specific pattern to check: does a contextual model actually resolve it?
5. **Operation routing logic** (`src/pipeline/operation_router.py`) — Substitution → CHV lookup (Rishabh's `chv_lookup.py`); Explanation/Generalization → constrained LLM prompts.
6. **Operation-constrained prompts** (`src/pipeline/constrained_prompts.py`) — explicit, per-operation instructions rather than a generic "simplify this," so the LLM executes what the classifier decided rather than deciding for itself.

### Team dependencies I'm tracking

- The operation router needs Sruthilaya's entity/numerical preservation metrics finalized before full-pipeline evaluation is meaningful.
- Rishabh's span reassembly logic depends on the router's output format being finalized first.
- Zihao's full-corpus evaluation should reuse the same stratified-sampling discipline established here — flagged directly in his task brief.

### Standing rule adopted from prelim
Any evaluation involving a rare event (like warning-bearing sentences at a 33% base rate) must be run on both a random sample and a targeted stratified sample before a safety claim is trusted — a random sample alone can silently hide the exact failure being tested for.