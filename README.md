# Operation-Aware Biomedical Text Simplification

### Complexity-Aware Routing with Safety Preservation

Biomedical text simplification should improve readability without removing
important medical information such as warnings, numbers, or clinical
concepts.

Our project studies this through three connected components:

1. **Complexity detection**
2. **Operation classification and generation**
3. **Safety-aware evaluation**

After reviewing the initial design, we decided not to blindly combine every
detector with the classifier. Instead, we integrate only selected features
that add useful biomedical or safety information without directly
reproducing the length-based pseudo-labeling rule.

---

## Where this stands right now (for reviewers)

**Prelim was delivered and reviewed. The full paper draft is written**, with
the central experiment now run and landed on the held-out test set (see
"The headline result — landed" below) — this section exists so a reviewer
can quickly see what's proven, what's corrected from an earlier draft, and
what's still open, rather than having to infer it from scattered results
files.

### Findings that are solid and reportable

- **Real UMLS integration achieves complete jargon coverage**: 100.0% of
  530 verified training abstracts have at least one accepted UMLS jargon
  match, vs. 86.8% for a morphological heuristic baseline — validated
  against random samples, not cherry-picked sentences, with 3 layers of
  filtering (semantic type, similarity threshold, Zipf word-frequency) to
  control false positives.
- **The 5 complexity detectors are near-independent**: strongest pairwise
  phi-correlation is 0.078 — they capture different complexity dimensions,
  not redundant overlap, supporting the "5 orthogonal signals" design.
- **BioBERT beats TF-IDF by +9.7% macro-F1** (0.465 vs. 0.424) on operation
  classification; a general-domain contextual model (DistilBERT) actually
  *underperforms* the TF-IDF baseline by -7.5% (0.392). Biomedical-specific
  pretraining helps; generic contextual pretraining alone does not.
- **Domain features (UMLS jargon, warning, numerical) improve the
  classifier by +2.5% macro-F1** when added to TF-IDF (0.449 vs. 0.424),
  once compared under a controlled ablation (see "Corrected findings"
  below) — driven almost entirely by UMLS jargon density, confirmed by
  inspecting the trained model's coefficients, not just aggregate F1.
- **Random sampling can hide rare safety failures.** A 50-sentence random
  sample showed all 3 baselines at perfect (1.0) warning preservation; a
  30-sentence warning-*stratified* stress test showed the direct-LLM
  baseline dropping to 0.950. Both numbers are correct — they measure
  different things, and only the stratified sample has the statistical
  power to catch a failure that occurs in ~5-33% of sentences (unit
  depends on abstract- vs. sentence-level measurement — see Limitations).
  This became a genuine methodological contribution: safety-critical NLP
  evaluation needs stratified sampling, not random sampling alone.

### Corrected findings (an earlier draft got these wrong — now fixed)

- **"Domain features don't help the classifier" was an artifact of an
  uncontrolled experiment, not a real finding.** The feature-augmented
  classifier's TF-IDF vectorizer used `max_features=5000` while the
  baseline it was compared against used `max_features=50000` — two
  variables changed at once (10x smaller vocabulary *and* added features),
  so the reported "-0.1% to -1.4%, features add noise" result was mostly
  measuring the vocabulary shrink, not the features. Re-run with matched
  `max_features=50000`: **TF-IDF + UMLS + numerical features improves
  macro-F1 by +2.1-2.5%**, not a decrease. A per-feature-group ablation
  further shows UMLS jargon density is doing ~76% of that work; warning
  features contribute ~0 in isolation (expected — warning language tracks
  output *safety*, not *which operation* a sentence needs, so its value is
  in the safety-preservation metric, not classification).
- **The operation-aware pipeline was wired to an untrained classifier.**
  The BioBERT fine-tuning script never persisted its trained weights, so
  the end-to-end pipeline was loading a fresh copy of base BioBERT with a
  randomly initialized classification head — its operation predictions
  were effectively noise, not the +9.7%-macro-F1 model reported above.
  Fixed: the classifier now saves a checkpoint, and the pipeline now fails
  loudly instead of silently substituting an untrained model. The
  substitution pathway had a matching issue — it defaulted to a
  pass-through placeholder rather than the real CHV lookup, which would
  have made Substitution-routed sentences look artificially safe. Both are
  fixed; checkpoint re-trained (val macro-F1 0.461, confirming Track 1's
  reported 0.465 independently).
- **A third bug (decoding-strategy mismatch) was caught before the first
  real pipeline run, not after.** The Direct-LLM baseline generates with
  deterministic beam search (`num_beams=4`); the operation-aware pipeline's
  LLM step was instead using random sampling — running the comparison as-is
  would have confounded "does operation-aware routing help" with "does
  decoding strategy differ." Fixed to match before running anything.
- **Pseudo-label distribution doesn't match the real, human-annotated
  distribution.** Length-ratio pseudo-labeling gives Substitution 43.2% /
  Explanation 34.4% / Generalization 22.5%; Ondov et al.'s real annotated
  PLABA distribution is 64.7% / 19.3% / 6.3% — Generalization is ~3.6x
  over-represented in the pseudo-labels. The classifier's ~0.42-0.47
  macro-F1 ceiling should be read against this: plausibly a pseudo-label
  quality ceiling as much as a model-capacity ceiling, which is the more
  honest framing for the paper.

### The headline result — landed, on both val and the held-out test set

**The paper's central comparison — the operation-aware system vs. the 3
baselines — has been run on both splits, and the test-set run is the one
that matters for the paper.** Test set (n=1,000 overall, n=49
warning-stratified — the true held-out split, never touched during
development):

| System | SARI (overall) | Warning Pres. | Numerical Pres. |
|---|---:|---:|---:|
| No simplification | 18.45 | 1.000 | 1.000 |
| Rule-based CHV | 21.12 | 1.000 | 1.000 |
| Direct LLM (no guidance) | 25.49 | 0.980 | 0.925 |
| **Operation-aware (this paper)** | **21.72** | **1.000** | **0.963** |

Operation-aware beats Rule-based CHV on SARI (21.72 vs. 21.12) while
matching its perfect warning preservation, and narrows the SARI gap to
Direct-LLM from 4.38 (Rule-CHV vs. Direct-LLM) to 3.77 points without any
of Direct-LLM's safety cost — it also beats Direct-LLM on numerical
preservation (0.963 vs. 0.925).

**On the warning-stratified stress test (n=49) — the exact subset where
safety failures concentrate — the result is even stronger than expected:**

| System | SARI (stratified) | Warning Pres. |
|---|---:|---:|
| No simplification | 18.79 | 1.000 |
| Rule-based CHV | 22.05 | 1.000 |
| Direct LLM (no guidance) | 24.73 | 0.980 |
| **Operation-aware (this paper)** | **25.43** | **1.000** |

**Operation-aware's SARI (25.43) actually exceeds Direct-LLM's (24.73) here,
while holding perfect warning preservation where Direct-LLM drops to
0.980** — on precisely the sentences where safety matters most, our system
is simultaneously safer *and* more readable, not merely safer at a cost.
This replicates the same qualitative pattern seen on val during
development (below), not a fluke of the held-out split. (n=49 is small —
treat the exact magnitude as indicative, not a precise population
estimate, per our own argument about stratified-sample statistical power.)

Read the overall claim as "near-total safety preservation at a moderate,
now-narrowed readability cost relative to an unconstrained LLM, and
outright superiority on the safety-critical subset" — not blanket
dominance on every metric everywhere, since Direct-LLM still leads on
overall-split SARI. This is a more honest and, we think, more defensible
claim than "our system wins on everything."

**How the SARI number got there — CHV-then-polish, using the complexity
detectors as a generation-time guardrail.** The first version of the
pipeline (CHV-only substitution, measured on val during development)
scored SARI 18.76 — barely above Rule-based CHV's 19.06 — because 43% of
sentences (Substitution-routed) got only a flat dictionary word-swap with
no fluency improvement. Fix: after CHV substitution, an optional FLAN-T5
fluency-polish pass runs, constrained by an explicit list of protected
spans pulled directly from `extract_numerical_expressions()`,
`detect_warnings()`, and the CHV terms just substituted — injected into the
prompt as "do not change these," and verified after generation (not just
requested): if any protected span is missing from the output, the pipeline
falls back to the guaranteed-safe CHV-only result. This is the first place
in the project where the complexity detectors feed the generation step
directly, not just the classifier or the diagnostic narrative. An
intermediate version protecting only numbers and CHV terms (not warning
phrases) raised SARI but dropped warning preservation to 0.986 — adding
warning-phrase protection recovered it to a perfect 1.000 while pushing
SARI even higher, a clean win rather than a trade-off.

**Val-set numbers (n=1,141 overall / n=73 stratified), for reference —
same qualitative pattern, used during development/model selection:**

| System | SARI (overall) | Warning Pres. | Numerical Pres. |
|---|---:|---:|---:|
| No simplification | 16.46 | 1.000 | 1.000 |
| Rule-based CHV | 19.06 | 1.000 | 1.000 |
| Direct LLM (no guidance) | 24.37 | 0.959 | 0.949 |
| **Operation-aware (this paper)** | **20.22** | **1.000** | **0.980** |

One methodological note worth stating plainly: the classifier checkpoint
used for the test-set run is not the literal same weight file used during
val-set development — an infrastructure incident deleted the original
checkpoint mid-session, and it was retrained under identical settings
before the test-set run. Its macro-F1 (0.467) is consistent with the other
two independent training runs (0.465, 0.461), which we treat as evidence
the specific training instance doesn't materially affect these results,
not as a methodological problem to hide.

---

## Research Questions

1. What types of complexity commonly occur in biomedical text?
2. Can we predict the required simplification operation?
3. Can selected complexity features improve operation prediction?
4. Can operation-aware generation improve readability while preserving
   important information?

---

## Current Pipeline

```text
Source biomedical text
        |
        |---- Complexity detectors
        |         - Biomedical entities
        |         - UMLS jargon
        |         - Warning cues
        |         - Numerical expressions
        |         - Syntactic complexity
        |
        |---- Selected integration features
        |         - UMLS jargon features
        |         - Numerical features
        |         - (Warning features tracked separately — safety metric,
        |            not a classifier feature; see Corrected Findings)
        |
        v
TF-IDF text features + selected detector features
        |
        v
Operation classifier (BioBERT, recommended — see Track 1 results)
        |
        v
Substitution / Explanation / Generalization
        |
        v
Operation-guided generation
  - Substitution -> CHV lookup (src/data/chv_lookup.py)
  - Explanation / Generalization -> constrained FLAN-T5 prompt
        |
        v
Readability and preservation evaluation
```

The five detectors are still used for corpus analysis and evaluation. Only
selected detector outputs are added to the classifier and generation
pipeline.

---

## Why We Selected These Feature Groups

The three candidate feature groups do not contribute in the same way —
this was confirmed empirically via ablation, not just assumed:

### UMLS jargon features

The load-bearing signal for **operation prediction**. Alone, they account
for +1.9 macro-F1 points of the classifier's total +2.5-point gain (76%).
Coefficient inspection confirms the model weights them meaningfully (mean
|coefficient| 0.124 vs. 0.084 for an average TF-IDF token), not just
correlating by chance.

### Warning features

Contribute ~0 to classification alone (+0.08 points), which is the
*expected* result: warning language indicates whether output is **safe**,
not **which operation** a sentence needs — a different axis entirely. Their
value is in the safety-preservation metric at generation time, not in
boosting classifier F1, and they are excluded from the recommended
classifier feature set for exactly this reason.

### Numerical features

The subtle case: alone, they slightly *hurt* Generalization F1 despite
`numerical_present` correlating 2.2x higher for Generalization (28.4%) than
Explanation (12.9%) sentences. They only turn net-positive once paired with
UMLS features — most plausibly because they help disambiguate
jargon-bearing sentences rather than acting as an independent signal.

**Recommended integration: UMLS + numerical (6 features).** All-8
(including warning) scores marginally higher in our val-set run, but that
margin is within noise for n=1,431 and has no principled justification.

---

## Scoping Decision

The original plan considered combining all five complexity detectors with
the operation classifier.

After reviewing:

- feature leakage risk
- pseudo-label quality
- implementation time
- experimental clarity

we narrowed the current paper to a controlled integration experiment.

We compare:

```text
Model 1: TF-IDF text only                         macro-F1 0.424
Model 2: TF-IDF + UMLS features                    macro-F1 0.443
Model 2.5: TF-IDF + UMLS + numerical (recommended) macro-F1 0.445
Model 3: TF-IDF + UMLS + warning + numerical       macro-F1 0.449
```

Syntactic-depth and length-related features are excluded from the current
classifier experiment — they correlate too closely with the length-ratio
pseudo-labeling rule itself and would risk the model partially
rediscovering its own label definition rather than learning real
complexity signal.

---

## Complexity Detection

We use five interpretable detectors.

### Biomedical entities

SciSpaCy identifies diseases, drugs, procedures, and other biomedical
concepts.

### UMLS jargon

QuickUMLS with UMLS 2026AA detects biomedical terminology, filtered by
semantic type, match similarity, and Zipf word-frequency (threshold 3.8,
validated against 16 known common/jargon words) to exclude common English
words that happen to also be UMLS concepts.

### Warning cues

A curated lexicon detects warnings, risks, and negated warning language
(negation-aware as of this session's bug fix — see Validation and Bug
Fixes).

### Numerical expressions

Detects dosages, percentages, measurements, confidence intervals,
p-values, and similar values.

### Syntactic complexity

Dependency-tree depth estimates sentence structure complexity. Falls back
to an approximate heuristic if spaCy is unavailable, but that fallback
now raises loudly rather than silently returning a misleading value (see
Validation and Bug Fixes).

---

## Current Complexity Results

The analysis was run on **530 unique PLABA training abstracts**.

The original file contains 635 rows, but some PMIDs have multiple
adaptation versions.

```text
635 training rows
531 unique PMIDs
530 usable unique abstracts
```

| Detector | Morphological Baseline | Real UMLS Pipeline |
|---|---:|---:|
| Biomedical entities | — | 98.7% |
| Warning cues | — | 31.3% |
| Syntactic complexity | 94.0% | 94.0% |
| Numerical expressions | — | 60.4% |
| Abstracts with at least one jargon match | 86.8% | **100.0%** |
| Three or more detectors firing | 90.8% | 98.1% |

The 100% UMLS result means every evaluated abstract contained at least one
accepted UMLS match. It does not mean every jargon term was detected or
that the detector has perfect precision or recall.

*Note on units:* the "31.3%" warning-cue figure above is measured at the
**abstract** level (at least one warning-bearing sentence per abstract).
At the **sentence** level (`results/classifier_features.csv`, n=9,085),
the warning-present rate is 5.7%. Both are correct — they're different
units of analysis — but any reuse of these numbers in the paper should
label the unit explicitly to avoid an apparent contradiction.

---

## Detector Independence

Pairwise phi correlations between the five detectors were close to zero.

The strongest observed correlation was approximately:

```text
phi = 0.078
```

This suggests the detectors capture mostly different kinds of complexity
rather than repeating the same signal.

---

## Operation Classification

The classifier predicts:

- **Substitution**
- **Explanation**
- **Generalization**

The current labels are generated using source-target length ratio and are
therefore weak supervision — see "Corrected Findings" above for the
measured mismatch against the real, human-annotated PLABA operation
distribution.

```text
Source sentence
        |
        v
TF-IDF (+ selected domain features) / BioBERT
        |
        v
Predicted operation
```

### Model comparison (val set, n=1,431)

| Model | Macro-F1 | Notes |
|---|---:|---|
| TF-IDF + Logistic Regression | 0.424 | Prelim baseline |
| DistilBERT | 0.392 | General contextual pretraining underperforms TF-IDF |
| **BioBERT** | **0.465** | +9.7% over TF-IDF; recommended final classifier |
| TF-IDF + UMLS + numerical (recommended feature set) | 0.445 | +2.1% over TF-IDF-only |
| TF-IDF + all 8 domain features | 0.449 | +2.5% over TF-IDF-only; marginal vs. above |

The macro-F1 range (0.42-0.47) is close to PLABA's own reported human
inter-annotator agreement (~0.46) and above prior baseline classifiers
reported around 0.33-0.39 — but given the pseudo-label distribution
mismatch documented above, this ceiling is at least partly a label-quality
ceiling, not purely a model-capacity one.

---

## Operation-Guided Generation

The predicted operation selects the generation strategy.

### Substitution

Replace difficult biomedical terms with simpler alternatives via CHV
lookup (`src/data/chv_lookup.py`, exact-match, case-insensitive, longest-
phrase-first to avoid partial matches).

### Explanation

Add a short explanation or definition, via a constrained FLAN-T5 prompt.

### Generalization

Rewrite the full sentence more broadly, via a constrained FLAN-T5 prompt.

Warning and numerical detector output is used as a generation-time
constraint (preserve warning meaning/negation; preserve exact numerical
values and units) — this is evaluated via the preservation metrics below,
not via the classifier's feature set.

**Status:** the pipeline (`src/pipeline/end_to_end_pipeline.py`) is
implemented end-to-end (classifier -> router -> CHV or constrained LLM),
and has been run against the 3 baselines for the paper's headline
comparison — see "The headline result — landed" above.

---

## Baseline Comparisons (val set, n=1,141 overall / n=73 warning-stratified)

| System | SARI (overall) | Warning Pres. (overall) | Warning Pres. (stratified) |
|---|---:|---:|---:|
| No simplification | 16.46 | 1.000 | 1.000 |
| Rule-based CHV | 19.06 | 1.000 | 1.000 |
| Direct LLM (FLAN-T5) | 24.37 | 0.959 | 0.959 |
| **Operation-aware (this paper)** | **20.22** | **1.000** | **1.000** |

Direct LLM has the highest SARI but is the only baseline to drop below
1.0 on warning and numerical preservation — the readability/safety
trade-off central to the paper's thesis, empirically visible for the
baselines. Operation-aware beats Rule-based CHV on SARI while matching its
perfect warning preservation, and narrows (but doesn't close) the SARI gap
to Direct-LLM — direct empirical support for the thesis, though not
outright dominance on every metric.

---

## Safety Evaluation

Readability alone is not enough.

Example:

```text
Original:
Do not exceed 2.5 mg daily.

Simplified:
Take the medicine daily.
```

The second sentence is simpler but loses the warning and dosage.

We therefore evaluate:

### Entity preservation

Checks whether biomedical entities remain in the output. Currently
string-match only (future work: CUI-based matching to credit valid
paraphrases like "hepatocellular carcinoma" -> "liver cancer").

### Numerical preservation

Checks whether important values and units remain correct.

### Warning preservation

Checks whether warnings, risk direction, and negation remain intact
(negation-aware as of this session's bug fix).

---

## Human Evaluation

A human-evaluation sample and rubric have been prepared.

The sample contains:

- 20 source sentences
- warning-stratified examples (10 warning-bearing and 10 non-warning
  source sentences)
- three simplification outputs per sentence

Reviewers evaluate: readability, meaning preservation, entity preservation,
numerical correctness, warning preservation, overall safety.

The evaluation is exploratory rather than a large-scale human study.

---

## Validation and Bug Fixes

Real bugs found and fixed via systematic audit (ongoing, not a one-time
pass):

### Warning negation

The warning detector could identify warning words without correctly
handling negation ("no evidence of X" was flagged the same as "X").

### Decimal extraction

The numerical extractor could truncate values such as `2.5`.

### Syntactic fallback

The syntactic detector could silently return unreliable values when
parsing failed. It now fails loudly instead of producing misleading
output.

### Track 2 ablation confound (this session)

`feature_augmented_classifier.py` compared models built with
`max_features=5000` against a baseline built with `max_features=50000`,
confounding the effect of adding domain features with a 10x vocabulary
shrink. Fixed by matching `max_features=50000` in both arms — see
"Corrected Findings" above.

### Untrained classifier in the generation pipeline (this session)

`biobert_classifier.py` never saved its trained weights;
`end_to_end_pipeline.py` was loading a fresh, randomly initialized
classification head instead of the trained model. Fixed: the classifier
now checkpoints its best epoch, and the pipeline now fails loudly instead
of silently substituting an untrained model.

### CHV placeholder (this session)

The generation pipeline's Substitution pathway defaulted to a pass-through
placeholder instead of the real CHV lookup, which would have made
Substitution-routed output look artificially safe. Fixed to default to the
real `chv_substitute()`.

---

## Reproducibility

Main scripts:

```text
infra/setup_quickumls.sh
infra/setup_ner_env.sh
src/evaluation/run_complexity_analysis.py
src/evaluation/extract_classifier_features.py
src/evaluation/entity_preservation.py
src/evaluation/numerical_preservation.py
src/evaluation/warning_preservation.py
src/evaluation/human_eval_sampling.py
src/classifier/tfidf_classifier.py
src/classifier/distilbert_classifier.py
src/classifier/biobert_classifier.py           # saves checkpoint to models/
src/classifier/feature_augmented_classifier.py
src/pipeline/end_to_end_pipeline.py            # requires biobert checkpoint
```

Supported UMLS backends:

```text
UMLS_BACKEND=morphological
UMLS_BACKEND=quickumls
```

---

## Repository Structure

```text
data/plaba/
infra/
notebooks/
src/
  complexity/
  classifier/
  pipeline/
  baselines/
  data/
  evaluation/
models/                  # BioBERT operation-classifier checkpoint (gitignored)
results/
paper/
team/
environment.yml
```

---

## Team Responsibilities

| Member | Main contribution | Pending work |
|---|---|---|
| Sruthilaya | Complexity detectors, UMLS pipeline, preservation metrics, validation, reproducibility, Track 1/2 correction + Baseline-4 bug fixes, operation-aware val + test-set runs, CHV-then-polish mechanism, full paper draft (this session) | Team review of drafted sections (Intro/Problem/Conclusion/Ethics — written this session, not yet reviewed by anyone else); final combine, trim to page limit |
| Sophakotra / Son | Pseudo-labeling, classifier (TF-IDF/DistilBERT/BioBERT), feature integration, generation pipeline | Confirm final classifier choice (BioBERT recommended); BioBERT+features experiment (paused, deferred to future work — not blocking) |
| Zihao | Full evaluation (SARI/FKGL/compression/preservation across 3 baselines) — **done, including the final test-set run** (now run by Sruthilaya using his harness pattern) | Review the final test-set numbers in `results/final_evaluation_testset.csv`; nothing blocking |
| Rishabh | Data pipeline, CHV vocabulary, EDA visualizations | Readability-vs-safety scatter plot — unblocked, real val + test numbers now exist; Related Work condensing and the bib-key fix are **done** (folded into this session's paper draft) |
| Whole team | — | review Introduction/Problem/Conclusion/Ethics drafts; final proofreading |

---

## Main Contributions

The project contributes:

1. Five interpretable biomedical complexity detectors
2. A real UMLS-based jargon pipeline (100% abstract-level coverage,
   validated)
3. Evidence that the complexity detectors capture mostly different signals
   (near-zero pairwise correlation)
4. A three-model classifier progression (TF-IDF -> DistilBERT -> BioBERT)
   isolating the effect of biomedical-domain pretraining
5. A controlled, ablation-validated integration of UMLS and numerical
   features into the classifier (+2.1-2.5% macro-F1)
6. Operation-guided generation (CHV substitution + constrained LLM
   prompting)
7. Entity, warning, and numerical preservation metrics
8. A human-evaluation protocol
9. A reproducible experimental pipeline, including two real bugs found and
   fixed this session (an uncontrolled ablation and an untrained
   classifier silently wired into the generation pipeline)

---

## Main Learning

The project started with the idea of combining all detector outputs with
the classifier, and separately, of treating "does adding a feature change
the F1 number" as sufficient evidence for whether that feature is useful.

Both assumptions needed correcting:

- Features should be selected based on the **role** they play, not added
  uniformly. UMLS features support operation prediction directly; warning
  features support safety-critical generation, not classification;
  numerical features help only in combination with UMLS. Treating all
  three as equally useful to the classifier would have hidden this.
- An empirical "features don't help" result is only as trustworthy as the
  ablation producing it. The single biggest correction this session was
  discovering that a change in TF-IDF vocabulary size, not the features
  themselves, explained most of an apparent negative result. **Before
  trusting a comparison, verify every other variable was actually held
  constant.**
- A pipeline that is architecturally complete is not the same as a
  pipeline that produces meaningful output. The generation pipeline "ran"
  and produced predictions long before anyone noticed those predictions
  came from an untrained model — a reminder to check not just that code
  executes, but that it executes on the thing you think it does.

---

## Limitations

- Operation labels are generated using a length-ratio heuristic, which
  measurably diverges from the real, human-annotated PLABA operation
  distribution (see Corrected Findings) — the classifier's macro-F1
  ceiling should be read as a joint function of model capacity and label
  quality, not model capacity alone.
- Only selected detector features (UMLS, numerical) are integrated into
  the classifier; warning features are deliberately excluded from
  classification and reserved for the safety-preservation metric.
- Syntactic and length-related features are excluded from the classifier
  experiment because of leakage risk with the length-ratio pseudo-labels.
- The UMLS result is abstract-level coverage, not perfect jargon
  detection; multi-word phrases can bypass the single-word frequency
  filter.
- Entity preservation is string-match only.
- Warning paraphrases are manually defined, not learned.
- Classifier progression (TF-IDF/DistilBERT/BioBERT) and the Track 2
  feature ablation were evaluated on val only, appropriate for the
  model-selection role they played; those specific numbers were not
  re-verified on test.
- The operation-aware-vs-3-baselines comparison **has** been run on the
  held-out test set (see "The headline result — landed" above) and is the
  number reported as the paper's headline result; the val-set run is kept
  as a secondary, consistent confirmation from development.
- Human evaluation and final generation comparison are still in progress.

---

## Final Paper Framing

The paper is framed as:

> An operation-aware biomedical simplification system that classifies the
> required edit operation (Substitution / Explanation / Generalization)
> before generating output, combining a validated UMLS-based complexity
> diagnostic with a controlled, ablation-verified feature-integration
> experiment, and evaluating the result against safety-preservation
> metrics that standard readability metrics (SARI, FKGL) do not capture.

The paper does not claim that all five detector signals are fused into one
model. It does now have direct empirical support for the operation-aware
system on the held-out test set (SARI 21.72 overall, warning preservation
1.000, beating Rule-based CHV and narrowing the gap to Direct-LLM without
its safety cost; on the warning-stratified subset, SARI 25.43 actually
exceeds Direct-LLM's 24.73 while holding perfect safety) — the claim is a
favorable safety/readability trade-off with outright dominance on the
safety-critical subset, not blanket dominance on every metric everywhere,
and this is now the frozen, final number reported at submission.

---

## Status

### Completed

- Five complexity detectors, real UMLS integration, validated
- Complexity analysis on 530 unique abstracts, detector correlation
  analysis
- Preservation metrics (entity, numerical, warning) and human-evaluation
  setup
- Classifier progression: TF-IDF (0.424) -> DistilBERT (0.392) -> BioBERT
  (0.465 macro-F1)
- Feature-integration ablation, corrected: UMLS + numerical features
  improve macro-F1 by +2.1-2.5%
- 3-baseline evaluation (no-simplification, rule-based CHV, direct LLM) on
  val, both overall (n=1,141) and warning-stratified (n=73) splits
- Three real bugs found and fixed this session: Track 2's uncontrolled
  ablation, the operation-aware pipeline's untrained-classifier /
  placeholder-CHV wiring, and a decoding-strategy mismatch vs. the
  Direct-LLM baseline
- BioBERT checkpoint trained and saved, reproduced across 3 independent
  runs (macro-F1 0.465, 0.461, 0.467)
- **Operation-aware system vs. 3-baselines comparison — the paper's
  central empirical result — run and landed on both val and the held-out
  test set.** Test set (the headline number): SARI 21.72 overall (beats
  Rule-based CHV's 21.12), warning preservation 1.000 (matches the safest
  baseline), narrows the SARI gap to Direct-LLM from 4.38 to 3.77 points;
  on the warning-stratified subset (n=49), SARI 25.43 exceeds Direct-LLM's
  24.73 while holding perfect warning preservation where Direct-LLM drops
  to 0.980
- CHV-then-polish generation step added: complexity detectors
  (numerical/warning/CHV-term detection) used as an explicit,
  post-hoc-verified guardrail on LLM generation, not just a classifier
  input — the first place detection directly feeds generation in this
  project
- Full paper draft written (`paper/final_paper/acl2023.tex`) — every
  section filled in with verified numbers, cross-checked digit-by-digit
  against the underlying results CSVs; bib duplicate-key issue fixed
- Reproducible infrastructure end-to-end

### In progress

- BioBERT + domain features experiment targeting the Generalization
  weakness (paused/deferred to future work — not blocking submission)
- Readability-vs-safety scatter plot (unblocked, real numbers now exist)
- Team review of newly-drafted paper sections; final assembly and
  page-limit trim
