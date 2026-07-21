# Project Status — Operation-Aware Biomedical Text Simplification
**Last updated:** 07/21/26 by Sruthilaya
**Deadline:** Team finishes remaining work by Thursday; final combine + submission after.

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

**Classifier (prelim, unchanged):** TF-IDF+LR, macro-F1 0.424 — within range
of PLABA's own reported human inter-annotator agreement (0.46) and prior
baseline classifiers (0.33-0.39).

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

## 5. What's pending — by owner

| Owner | Pending |
|---|---|
| Sophakotra | Decide on detector-feature integration question (Section 1); confirm `_levenshtein` dead-code question; generation pipeline; Main Idea section |
| Zihao | Full-corpus evaluation (currently n=530 verified split, not the earlier n=635 estimate — use the corrected split) |
| Rishabh | Readability-vs-safety scatter plot (can now use verified n=530 numbers); Related Work condensing |
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