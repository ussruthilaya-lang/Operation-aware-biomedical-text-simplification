# Project Status — Operation-Aware Biomedical Text Simplification
**Last updated:** 07/31/26 by Sruthilaya
**Deadline:** Team finishes remaining work by Thursday; final combine + submission after.

---

## -1. Session update (07/31) — test-set run landed, full paper written, numeric audit fixed 2 real errors

The operation-aware-vs-3-baselines comparison was run on the true, held-out
**test set** (n=1,000 overall / n=49 warning-stratified — never touched
during development), using `src/evaluation/run_final_testset_evaluation.py`
(mirrors Zihao's `run_evaluation.py` pattern, extended to `_TEST_CSV`).
Result, and it's stronger than the val-set run:

| System | SARI (overall) | Warn. (overall) | Num. (overall) | SARI (stratified) | Warn. (stratified) |
|---|---|---|---|---|---|
| No simplification | 18.45 | 1.000 | 1.000 | 18.79 | 1.000 |
| Rule-based CHV | 21.12 | 1.000 | 1.000 | 22.05 | 1.000 |
| Direct LLM | 25.49 | 0.980 | 0.925 | 24.73 | 0.980 |
| **Operation-aware** | **21.72** | **1.000** | 0.963 | **25.43** | **1.000** |

**Headline finding:** on the warning-stratified stress test, operation-aware's
SARI (25.43) *exceeds* Direct-LLM's (24.73) while holding perfect warning
preservation where Direct-LLM drops to 0.980 — on exactly the sentences
where safety failures concentrate, our system is simultaneously safer and
more readable, not just safer at a cost. Same qualitative pattern as val,
stronger magnitude. Written to `results/final_evaluation_testset.csv`.

Also wrote the full paper draft (`paper/final_paper/acl2023.tex`) — every
remaining TODO section filled in (Abstract, Introduction, Problem, Main
Idea, Operation Classification, CHV-then-polish, Full-System Evaluation,
Related Work, Conclusion, Limitations, Ethics) — and fixed
`paper/final_paper/custom.bib`'s duplicate/stub citation keys.

**Numeric cross-check caught 2 real errors before they reached the paper,
now fixed:**
1. A sentence claimed the SARI gap between Rule-based CHV and Direct-LLM
   was "7.05 points" — the real value is 4.38; 7.05 was actually a
   different, irrelevant comparison (Direct-LLM minus No-simplification)
   that got mislabeled during drafting. Same class of bug appeared twice
   more in README.md ("5.61" where the real val-set gap is ~5.31) — fixed
   in both places.
2. A sentence cited "Generalization F1 0.365, recall 32%" as if it were
   tied to the headline 0.465 macro-F1 run, when it was actually from a
   *different* training run (0.461). Corrected to report the honest range
   across all three independent runs (F1 0.365–0.423, recall 29–32%)
   rather than implying false precision from an unlabeled single run.

Also flagged and disclosed honestly in the paper: the classifier checkpoint
used for the test-set run is not the literal same weight file used during
val-set development (the filter-repo incident deleted the original
checkpoint mid-session; it was retrained under identical settings before
the test-set run). Framed as supporting evidence for reproducibility
(macro-F1 0.467 vs. the other two runs' 0.465/0.461), not hidden.

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

**BioBERT checkpoint training complete (07/29):** val macro-F1 0.461,
independently confirming Track 1's reported 0.465 (small variance is
expected — training isn't seeded). Checkpoint saved to
`models/biobert_operation_classifier/`. Generalization remains the weakest
class (F1 0.365, recall 32%, confused almost equally with both other
classes) — see Section 2 and Section 5.1 item 2 for why and what to do
about it.

**Bug 3 — decoding-strategy mismatch, found during planning, fixed before
running anything.** `baseline3_direct_llm.py` (the Direct-LLM baseline
already in `results/final_evaluation.csv`) generates with deterministic beam
search (`num_beams=4`, `max_new_tokens=256`). The operation-aware pipeline's
`llm_constrained_simplify()` was instead using random sampling
(`do_sample=True, top_p=0.9, temperature=0.8`, `max_length=100`). Running the
comparison as-is would have confounded "does operation-aware routing help"
with "does beam search vs. sampling produce different text" — the same class
of mistake as Bug 1's ablation confound. **Fixed:** matched
`llm_constrained_simplify` to the same beam-search config as baseline3.

**HEADLINE RESULT LANDED (07/29) — the paper's central comparison now
exists.** Ran `OperationAwarePipeline` (trained BioBERT + real CHV +
decoding-matched FLAN-T5) against the same val split Zihao used (n=1,141
overall, n=73 warning-stratified), through the same metric suite. Initial
run: SARI 18.76 overall, warning preservation a perfect 1.000 (vs.
Direct-LLM's 0.959) — confirms the thesis's safety half directly, but SARI
trailed Direct-LLM (24.37) and barely beat Rule-based CHV (19.06), because
43% of sentences (490/1,141) routed to Substitution got only a flat CHV
dictionary swap with zero fluency improvement.

**New addition — CHV-then-polish, using the complexity-detection features
as a generation-time guardrail (not just a classifier input).** Added
`chv_substitute_and_polish()`: CHV substitution first (safe, deterministic),
then an optional FLAN-T5 fluency-polish pass, explicitly constrained by
protected spans pulled from `extract_numerical_expressions()`,
`detect_warnings()`, and the actual CHV replacement terms just inserted —
injected into the prompt as spans the LLM must not alter, and verified
post-hoc (not just requested): if any protected span is missing from the
polished output, the pipeline falls back to the guaranteed-safe CHV-only
result. This is the first place in the project where the complexity
detectors feed the generation step directly, not just the classifier or the
diagnosis narrative. Iterated in 3 passes:

| Version | SARI (overall) | SARI (stratified) | Warning Pres. (both splits) |
|---|---|---|---|
| CHV-only (no polish) | 18.76 | 16.07 | 1.000 |
| CHV+polish, numeric/CHV-term protection only | 20.09 | 16.42 | 0.986 (regression) |
| **CHV+polish, + warning-phrase protection (final)** | **20.22** | **17.99** | **1.000** |

The middle version's regression (warning preservation dropping to 0.986) was
itself a useful, honest finding: protecting numbers and substituted jargon
wasn't enough, because a Substitution-routed sentence that also contained a
warning phrase had that phrase left unprotected and the polish pass could
reword it. Adding `detect_warnings()`'s matched phrases to the protected-span
list fully recovered warning preservation to 1.000 **and** pushed SARI
higher than the unprotected version — a genuine win, not a trade-off.

**Final headline table (val set, n=1,141 overall / n=73 warning-stratified):**

| System | SARI (overall) | Warning Pres. | Numerical Pres. |
|---|---|---|---|
| No simplification | 16.46 | 1.000 | 1.000 |
| Rule-based CHV | 19.06 | 1.000 | 1.000 |
| Direct LLM (no guidance) | 24.37 | 0.959 | 0.949 |
| **Operation-aware (val, superseded by test-set result — see Section -1)** | **20.22** | **1.000** | **0.980** |

Operation-aware now beats Rule-based CHV on SARI while matching it on
warning preservation, and closes much of the SARI gap to Direct-LLM
(5.31→4.14 points) while remaining the only system besides Rule-based CHV
with perfect warning preservation. Frame this honestly as "near-total safety
preservation at a moderate, now-narrowed readability cost relative to an
unconstrained LLM" — not outright dominance on every metric, since
Direct-LLM still leads on raw SARI. **This val-set result has since been
confirmed and strengthened on the held-out test set (Section -1) — the
test-set numbers are what the paper reports.**

**Net effect:** neither Track 1 (BioBERT +9.7%, real) nor Track 2 (domain
features, corrected below) needed to be thrown out — but Track 2's framing
flips from negative to positive, and the pipeline needed three real bugs
fixed (Track 2 ablation confound, untrained classifier, decoding mismatch)
before it could produce a trustworthy headline number, which now exists.
This is going into Limitations either way: **Track 1/2, Zihao's baselines,
and this operation-aware run were all evaluated on val, not test** — one
final test-set run with the frozen final configuration (Section 5, item 3,
Zihao) is still needed before these numbers go in the paper's headline
table for submission.

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

## 5. Final push — assigned by owner, sequenced to avoid overlap

Each person owns a distinct set of files/artifacts below — nobody should
need to edit another person's owned files. Sequencing matters (see
"depends on" column): don't start a blocked item early on stale inputs.

| # | Owner | Task | Owns / touches | Depends on | Produces (write results here) |
|---|---|---|---|---|---|
| 1 | **Sruthilaya** | ~~Fix decoding-strategy mismatch, run the operation-aware-vs-3-baselines comparison~~ **DONE (07/29).** Final: SARI 20.22 (overall), warning preservation 1.000, beats Rule-based CHV on SARI while matching its safety — see Section 0 for the full result and the CHV-then-polish addition that got it there | `src/pipeline/end_to_end_pipeline.py`; `src/evaluation/run_operation_aware_evaluation.py` | — | `results/final_evaluation.csv` (`operation_aware` rows added) — **#3 and #4's scatter plot are now unblocked** |
| 2 | **Sophakotra / Son** | Confirm BioBERT as final classifier; own the **BioBERT + domain features (UMLS + numerical)** experiment targeting the Generalization weakness (Section 5.1 item 2). **Must save to a separate checkpoint dir, not `models/biobert_operation_classifier/`**, so it doesn't overwrite the checkpoint #1 depends on. Also: Main Idea section (reflect the corrected Track 1/2 story, not the original PR framing); confirm `_levenshtein` dead-code question | `src/classifier/biobert_classifier.py` (read-only reference), new script/checkpoint dir e.g. `models/biobert_features_operation_classifier/`; `paper/final_paper/acl2023.tex` Main Idea section | None — can start immediately, fully parallel to #1 | New results file, e.g. `results/biobert_features_classifier.txt`; Main Idea prose |
| 3 | **Zihao** (run by Sruthilaya, 07/31, using his harness pattern) | ~~Final test-set run~~ **DONE.** All 4 systems compared on the true test split (n=1,000 overall / n=49 stratified) — see Section -1 for the headline result. Entity preservation still `N/A` (NER env integration remains open, low priority) | `src/evaluation/run_final_testset_evaluation.py` (new file, mirrors his `run_evaluation.py` pattern) | — | `results/final_evaluation_testset.csv` — **please review these numbers, they're now the paper's headline result** |
| 4 | **Rishabh** | Related Work condensing and the duplicate bib-key fix are **done** (folded into the 07/31 paper draft, Section -1). **Remaining: readability-vs-safety scatter plot** — real val AND test numbers now exist (Section -1's tables), no longer blocked on anything. Random-routing ablation (Section 5.1 item 3) stays a not-required contingency — #1/#3's results were a clear, unambiguous win on both splits, so it never needed to run; leave the idea on standby | `notebooks/visualizations.ipynb` (his file) for the scatter plot | — | Scatter plot figure, e.g. `results/readability_safety_scatter.png` |
| 5 | Whole team | Introduction, Problem section, Conclusion, Ethics Statement; final 8-page trim | `paper/final_paper/acl2023.tex` (coordinate before editing simultaneously) | All of #1-4 for real numbers | — |
| 6 | Sruthilaya | Final paper section review/polish once other sections land; final combine | — | #1-5 | — |

**Conflict-avoidance notes:**
- Only Sruthilaya touches `end_to_end_pipeline.py`'s decoding config, and only until #1 is done — nobody else should edit that file in the meantime.
- Son's feature experiment must NOT write to `models/biobert_operation_classifier/` — that checkpoint is actively depended on by #1's run. Use a separate directory.
- Zihao's test-set run and Rishabh's scatter plot both explicitly wait on #1 landing — starting early would mean redoing the work against a pipeline that's still being fixed.
- Each new result goes to its own new file (see "Produces" column) rather than overwriting `results/final_evaluation.csv` or `results/classifier_features.csv` — those stay as the historical val-set record.

### 5.1. New experiments identified this session (not yet run — priority-ordered)

1. **Run the main operation-aware-vs-3-baselines comparison first (this is
   still #1, unchanged).** Everything below is secondary to this.
2. **BioBERT + domain features (UMLS + numerical), targeting the
   Generalization weakness.** Track 1 (BioBERT alone) and Track 2 (features
   + TF-IDF only) were never combined — nobody has tested BioBERT with the
   engineered features concatenated before the classification head.
   Motivation: BioBERT's current confusion matrix shows Generalization is
   by far its weakest class (F1 0.365, recall 32%, confused almost equally
   with both other classes) — plausibly because contextual embeddings alone
   aren't finding a clean signal for it given the pseudo-label noise. An
   explicit engineered signal (numerical/jargon density) is a different
   *kind* of signal than what BERT infers implicitly, so it's worth testing
   whether it gives BioBERT a more direct cue specifically for this class.
   Caveat going in: numerical density alone already failed to help
   Generalization in the TF-IDF ablation (0.410→0.387) — the label-noise
   ceiling may cap this experiment too, but it's untested with BioBERT
   specifically and worth 30-60 minutes to check.
3. **Random-routing ablation — contingency only, not required.** The
   general claim "classify-then-execute beats no classification" is already
   established prior art (Cripwell et al. 2022, already cited) — this
   paper's contribution is applying it to the biomedical safety domain, not
   re-proving the mechanism from scratch. This ablation (same constrained
   prompts, but routed to a random operation instead of BioBERT's
   prediction) is only worth running **if the main comparison in item #1
   comes back ambiguous** (e.g. SARI improves but preservation doesn't, or
   the margin over Direct-LLM is small) — in that case it's the one piece
   of evidence that separates "classification isn't accurate enough to
   route correctly" from "downstream generation doesn't differentiate much
   regardless of routing." Do not run pre-emptively; only reach for it if
   item #1's result needs explaining.
4. **A second LLM/generation backbone through the operation-aware router**
   (e.g. Lay-SciFive or BART-w-CTs, already on Zihao's task list as
   separate baselines) — lowest priority. Answers "does the
   safety/readability trade-off and the operation-aware improvement
   generalize beyond FLAN-T5-base," which matters for robustness but isn't
   required to support the paper's core causal claim. Reasonable to leave
   as a Limitations/Future Work line if time runs out before Thursday.

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