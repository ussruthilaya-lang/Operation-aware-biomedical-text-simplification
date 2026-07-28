# Project Status — Operation-Aware Biomedical Text Simplification
**Last updated:** 07/28/26 by Sruthilaya
**Deadline:** Team finishes remaining work by Thursday; final combine + submission after.

---

## 0. Session update (07/28) — Son's PR reviewed, 2 real bugs found + fixed, 1 training run in progress

Son's PR ("Track 1: Pretraining Comparison" + "Track 2: Feature Integration" +
Baseline 4 pipeline) was reviewed line-by-line against the actual code, not
just the PR summary. Two findings changed the paper's conclusions; full
detail below in Section 2 and Section 4.5.

**Bug 1 — Track 2's "features don't help" conclusion was an artifact of an
uncontrolled experiment, now fixed.** `feature_augmented_classifier.py` built
its TF-IDF vectorizer with `max_features=5000`, while the reported "0.424
baseline" it was compared against used `max_features=50000`
(`tfidf_classifier.py`). Two variables changed at once (vocab size shrunk
10x AND features added), so the reported -0.1% to -1.4% "features add
noise" finding wasn't isolating the effect of the features at all — it was
mostly measuring the vocab shrink. **Fixed:** `max_features=50000` in both
Model 2 and Model 3, matching the baseline exactly. Corrected out-of-domain
comparison, plus a per-feature-group ablation to see exactly which of the 8
features carry signal — see Section 2 for corrected numbers. Also corrected
"9 features" → 8 (the actual count) throughout the file/docstrings.

**Bug 2 — the Baseline-4 pipeline was wired to an untrained classifier, now
fixed but re-training in progress.** `biobert_classifier.py` trains BioBERT
across 3 epochs and tracks `best_val_f1`, but never called
`save_pretrained()` anywhere — the trained weights were discarded when the
script exited. Meanwhile `end_to_end_pipeline.py` loaded a **fresh copy** of
`dmis-lab/biobert-base-cased-v1.2` with `num_labels=3`, which attaches a
**randomly initialized, untrained** 3-way classification head every time
(the base checkpoint has no such head). The entire operation-aware pipeline
was therefore routing sentences based on random noise, not the +9.7%
macro-F1 classifier reported in Track 1 — this is exactly why the PR's
baseline table showed `Operation-Aware | Pending CHV` with no real numbers;
the architecture literally could not produce a meaningful prediction yet.
**Fixed:** `biobert_classifier.py` now saves the best-epoch checkpoint to
`models/biobert_operation_classifier/`; `end_to_end_pipeline.py` now loads
from that checkpoint and **raises a loud `FileNotFoundError`** instead of
silently falling back to an untrained model if the checkpoint is missing.
Re-training BioBERT to produce the first real checkpoint is running now
(CPU-only, ~3 epochs over 6,307 sentences — this is the long pole before the
operation-aware-vs-baselines comparison can be run for real).

**CHV wiring also fixed while in there:** `OperationAwarePipeline` defaulted
`chv_lookup_fn` to a placeholder that returned the input sentence unchanged
— meaning Substitution-routed sentences would score artificially perfect on
warning/entity preservation for the wrong reason (they were never touched).
Now defaults to Rishabh's real `chv_substitute()` (`baseline2_rule_based_chv.py`,
built on `src/data/chv_lookup.py`), so Substitution actually happens.

**Net effect:** neither Track 1 (BioBERT +9.7%, real) nor Track 2 (domain
features, corrected below) needed to be thrown out — but Track 2's framing
flips from negative to positive, and the pipeline needed two real bugs fixed
before it can produce a trustworthy headline number. This is going into
Limitations either way: **Track 1/2 were both evaluated on val, not test**
— one final test-set run with the frozen final configuration is still
needed before these numbers go in the paper's headline table.

---

## 1. Architecture — how the pieces fit (please read before your part)

```
[Pseudo-labeler]  source/target pairs -> length-ratio heuristic -> S/E/G label
       |
       v
[TF-IDF + LR classifier]  source sentence TEXT ONLY -> predicted operation
       (trains on pseudo-labels; no target text at inference — Sophakotra)

[5 complexity detectors]  run SEPARATELY, parallel analysis
       (NER, warning, syntactic, numerical, UMLS jargon — Sruthilaya)
       -> feeds coverage-gap argument + safety preservation metrics
       -> NOT currently wired into the classifier as features
```

**Currently these are two complementary contributions, not one fused pipeline.**
Detection answers "what kind of complexity exists" (motivates the coverage-gap
finding + safety metrics). Classification answers "which operation applies"
(routing decision for generation). This is fine and intellectually honest as
framed — they don't need to be literally fused for the paper to hold together,
**as long as we frame it as two complementary diagnostic contributions, not
claim a single wired pipeline we don't actually have.**

### Open question for Sophakotra (needs a decision, not urgent unless you have spare time)
Should complexity detector flags be added as classifier features (e.g. UMLS
jargon detected + high syntactic depth -> stronger Explanation signal)?
**Caution if attempted:** pseudo-labels are defined by length ratio
(target/source word count) — syntactic depth correlates with sentence length,
so adding it as a feature risks the model partially rediscovering its own
label definition rather than learning real complexity signal. UMLS jargon
flags are safer to add (not length-correlated) if you want to try a light
version. **Recommendation: skip fusion given the 2-day timeline unless you
have real spare bandwidth — document as future work instead** (fits naturally
alongside the existing TF-IDF+LR -> BERT -> BioBERT upgrade path already
planned).

### Also flagging: unused function in `pseudo_labeler.py`
A `_levenshtein`/`_normalized_edit_distance` function exists with a docstring
implying it should factor into Substitution detection ("high character edit
distance"), but `label_span()` never calls it — dead code or abandoned
design? Worth a 30-second confirmation either way.

---

## 2. Results — current, verified numbers

**Complexity detection, n=530 abstracts (verified split — see Section 4 for
why this differs from an earlier n=635 estimate):**

| Detector | Morphological baseline | Real UMLS (full pipeline) |
|---|---|---|
| NER | — | 98.7% |
| Warning cues | — | 31.3% |
| Syntactic depth | 94.0% | 94.0% |
| Numerical expressions | — | 60.4% |
| Jargon (UMLS) | 86.8% | **100.0%** |
| 3+ detectors firing | 90.8% | 98.1% |

**Headline result:** real UMLS lookup achieves complete jargon coverage vs.
86.8% for a morphological heuristic — validated against both targeted test
sentences and a random sample of real abstracts (manual review, no residual
false positives after 3 rounds of filtering).

**Detector independence (new this session):** pairwise phi-coefficient
correlation among the 5 detectors is near-zero across the board (strongest:
0.078) — detectors fire largely independently, supporting the "5 orthogonal
complexity signals" architectural claim rather than redundant overlap.

**Classifier — Track 1 (pretraining comparison), val set, n=1,431:**

| Model | Macro-F1 | Accuracy | Substitution F1 | Explanation F1 | Generalization F1 |
|---|---|---|---|---|---|
| TF-IDF + LR (baseline) | 0.424 | 0.431 | 0.476 | 0.385 | 0.410 |
| DistilBERT | 0.392 | 0.407 | 0.412 | 0.305 | 0.459 |
| BioBERT | **0.465** | 0.472 | 0.501 | 0.471 | 0.423 |

BioBERT beats the TF-IDF baseline by +9.7% macro-F1; DistilBERT (general
contextual pretraining, no biomedical adaptation) actually *underperforms*
TF-IDF by -7.5%. Clean, interpretable result: biomedical-domain pretraining
helps, generic contextual pretraining alone does not. Recommended as the
final classifier for the pipeline.

**Classifier — Track 2 (feature integration), val set, n=1,431 — corrected
this session (see Section 0):**

| Model | Macro-F1 | Δ vs baseline | Sub F1 | Exp F1 | Gen F1 |
|---|---|---|---|---|---|
| TF-IDF only (baseline) | 0.424 | — | 0.476 | 0.385 | 0.410 |
| TF-IDF + UMLS only (3 feat) | 0.443 | +0.019 | 0.498 | 0.433 | 0.398 |
| TF-IDF + UMLS + Numerical (6 feat, warning dropped) | 0.445 | +0.021 | — | — | — |
| TF-IDF + all 8 feat (UMLS+warning+numerical) | 0.449 | +0.025 | 0.500 | 0.441 | 0.405 |

Per-feature-group ablation (isolating each group's independent contribution):
UMLS jargon features alone account for +0.019 of the total +0.025 gain
(76%) — the clear load-bearing signal, confirmed by coefficient inspection
(mean |coef| for the 8 numeric features = 0.124 vs. 0.084 for an average
TF-IDF token — genuinely weighted, not noise). Warning features contribute
~0 alone (+0.0008) — expected and correct, since warning language tracks
output *safety*, not *which operation* a sentence needs; their real value
is in the safety-preservation metric, not classification, and they're
excluded from the recommended feature set for exactly this reason.
Numerical features are the subtle one: alone they slightly hurt
Generalization F1 (0.410→0.387) despite `numerical_present` correlating
2.2x higher for Generalization (28.4%) than Explanation (12.9%) in the raw
data — they only help once paired with UMLS, most plausibly because they
disambiguate jargon-bearing sentences rather than standing alone.
**Recommended reported model: UMLS + Numerical (6 features)** — the
principled combination; all-8 is marginally higher (+0.004) but that margin
is within noise for n=1,431 and includes a feature group with no
independent justification for being there.

**Why Generalization doesn't improve much either way:** ties directly to
the pseudo-label distribution mismatch (Substitution 43.2% / Explanation
34.4% / Generalization 22.5% pseudo-labeled vs. Ondov et al.'s real
human-annotated 64.7% / 19.3% / 6.3% — Generalization is ~3.6x
over-represented in the pseudo-labels). Source-sentence word count is
highest for pseudo-labeled Generalization (27.6 words) vs. Explanation
(19.9) — backwards from what a clean length-ratio signal should produce —
and numerical density is also highest for Generalization. Read together:
the length-ratio heuristic is likely mislabeling "dropped a stats-heavy
clause" as Generalization when it's closer to Omission, so the Generalization
label itself is noisier than Substitution/Explanation, capping how much any
feature (including numerical density, which is real signal on its own) can
improve that specific class. **This is a data-quality ceiling, not a
modeling ceiling** — the more honest and more interesting framing for the
paper than "the classifier's macro-F1 is capped at ~0.45."

---

## 3. Reproducible pipeline — how to rebuild everything from scratch

All infra is scripted and tested end-to-end; nothing depends on a specific
person's laptop or a specific still-running VM.

1. **Real UMLS 2026AA index** (10.7M concepts) — backed up in GCS bucket
   `nlp-text-simplification-umls`. To rebuild on a fresh VM: run
   `infra/setup_quickumls.sh`, then re-download index from bucket or rerun
   `quickumls.install` against a fresh UMLS license download (~20 min build).
2. **NER environment** (scispaCy, isolated due to spaCy version conflict with
   QuickUMLS) — `infra/setup_ner_env.sh`, tested idempotent.
3. **PLABA data** — small (7.2MB), lives in `data/plaba/` in the repo /
   bucket, not regenerated.
4. **Full detector pipeline** — `src/evaluation/run_complexity_analysis.py`,
   supports `UMLS_BACKEND=morphological` or `UMLS_BACKEND=quickumls`.
5. **Preservation metrics** — `entity_preservation.py`,
   `numerical_preservation.py`, `warning_preservation.py` (all in
   `src/evaluation/`).
6. **Human eval sample** — `src/evaluation/human_eval_sampling.py` generates
   `results/human_eval_sample.csv` (40 sentences, warning-stratified, 3
   baselines each) + rubric in `human_eval_rubric.md`.

**No manual, undocumented steps remain** — anyone on the team should be able
to reproduce every number in Section 2 from a clean checkout + these scripts.

---

## 4. What's done (Sruthilaya) — fully complete, verified

- QuickUMLS full integration (real UMLS, 3-layer filtering: semtype,
  threshold, Zipf-frequency — validated against random samples, not just
  cherry-picked sentences)
- Entity + numerical preservation metrics
- Human evaluation sampling + rubric
- **3 real bugs found and fixed** via systematic audit (same rigor as UMLS
  validation): `warning_lexicon.py` negation blindness, `numerical_extractor.py`
  decimal truncation, `syntactic_depth.py` fallback misranking (now fails
  loudly instead of silently guessing)
- Per-pair detector correlation analysis
- `environment.yml` + `plaba_loader.py` fixed and documented
- **n=635 vs n=530 discrepancy fully resolved**: `train.csv` has 635 rows but
  531 unique PMIDs (530 after excluding 1 row with a blank PMID) — 104 PMIDs
  have 2 adaptation-version rows each (427×1 + 104×2 = 635). The earlier
  "635 abstracts" figure counted rows, not deduplicated abstracts. **530 is
  the correct, methodologically sound number going forward.**
- All infra reproducible and scripted (see Section 3)

## 4.5. What's done this session (Sruthilaya, reviewing Son's PR)

- Reviewed Son's Track 1/Track 2/Baseline-4 PR line-by-line against the
  actual code (not just the PR summary)
- **Fixed Track 2's uncontrolled ablation** (`max_features` mismatch) —
  corrected numbers now show domain features genuinely help (+2.5%
  macro-F1), reversing the PR's "features add noise" conclusion; added a
  per-feature-group ablation (`train_model_2_5` in
  `feature_augmented_classifier.py`) isolating UMLS/warning/numerical's
  individual contributions
- **Fixed the untrained-classifier bug** in the Baseline-4 pipeline —
  `biobert_classifier.py` now saves a checkpoint (`models/biobert_operation_classifier/`);
  `end_to_end_pipeline.py` now loads it and fails loudly instead of silently
  using a randomly initialized head
- **Fixed CHV wiring** — `OperationAwarePipeline` now defaults to Rishabh's
  real `chv_substitute()` instead of a passthrough placeholder
- Re-training BioBERT to produce the first real checkpoint (in progress as
  of this update — required before the operation-aware-vs-baselines
  comparison can be run)

## 5. What's pending — by owner

| Owner | Pending |
|---|---|
| Sruthilaya (in progress now) | BioBERT checkpoint re-training running; once done, run `OperationAwarePipeline` (BioBERT + CHV + FLAN-T5) against the val set through the same SARI/FKGL/warning/entity/numerical-preservation harness Zihao built, for the headline operation-aware-vs-3-baselines comparison — **this is the single most important number the paper's thesis rests on, and it still doesn't exist** |
| Sophakotra / Son | Confirm which of Track 1's BioBERT is the final classifier (recommended, given +9.7% macro-F1); confirm `_levenshtein` dead-code question; Main Idea section — should describe the corrected Track 1/2 story, not the original PR framing |
| Whole team | **One final test-set run** (not val) needed for both the classifier tables and the operation-aware pipeline comparison — everything reported so far (Track 1, Track 2, Zihao's baselines) was evaluated on val, which is appropriate for model selection but not for the paper's headline numbers |
| Zihao | Full-corpus evaluation already done on val (n=1,141, both overall and warning-stratified splits) — needs the same treatment re-run once operation-aware system's real numbers exist, so all 4 systems are compared on identical splits |
| Rishabh | Readability-vs-safety scatter plot — still not built (blocked on operation-aware numbers, which is now the active blocker, not Zihao's numbers which already landed); Related Work condensing; fix duplicate bib-key issue (`Ondov2025`/`Attal2023` in `paper/final_paper/custom.bib` duplicate `ondov2025lessons`/`attal2023plaba` already in `paper/custom.bib`) |
| Whole team | Introduction, Problem section, Conclusion, Ethics Statement; final assembly + 8-page trim Thursday |
| Sruthilaya | Paper section review/polish once other sections land; final combine Thursday |

---

## 6. Known limitations (for the paper's Limitations section — please don't duplicate, add here if you find more)

- Entity preservation metric: string-match only, doesn't credit valid
  paraphrases ("hepatocellular carcinoma" -> "liver cancer") as preserved —
  future work: CUI-based matching
- Jargon detector: multi-word phrases (e.g. "urine output") bypass the
  single-word frequency filter, occasional borderline false positive
- Warning-preservation paraphrase map is hand-built, not learned (documented
  since prelim)
- Pseudo-label distribution mismatch: length-ratio heuristic produces
  Substitution 43.2% / Explanation 34.4% / Generalization 22.5%, vs. Ondov et
  al.'s real human-annotated 64.7% / 19.3% / 6.3% — Generalization is ~3.6x
  over-represented in pseudo-labels, likely because the heuristic conflates
  "dropped a stats-heavy clause" with genuine conceptual generalization. The
  classifier's ~0.42-0.47 macro-F1 ceiling should be read against this —
  plausibly a label-quality ceiling, not purely a model-capacity ceiling
- Track 1 and Track 2 classifier numbers, and Zihao's full-corpus baseline
  evaluation, were all computed on the **val** split — a final **test**-split
  run with the frozen configuration is still needed before these numbers are
  reported as the paper's headline results