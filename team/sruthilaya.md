# Sruthilaya — Complexity Detection & Safety Evaluation Lead

**Branch:** `main` (foundation branch — team builds on top of this)
**Role:** Stage 1 (Complexity Detection) + Safety Metrics + Repo/Paper Integration Owner

---

## STATUS

| Phase | Status |
|---|---|
| Prelim Report | ✅ Submitted — ACL format, 4 pages, all sections verified |
| Full Paper | 🔄 In progress |

---

## WHAT'S BUILT — Stage 1 Complete (Prelim Phase)

### 5 Complexity Detectors (`src/complexity/`)

| File | What it detects | NLP paradigm | Why this paradigm |
|---|---|---|---|
| `numerical_extractor.py` | Dosages, risk ratios, p-values, CIs | Regex | Numerical patterns are finite and rule-governed — no ML needed, 100% recall on known formats |
| `warning_lexicon.py` | Contraindications, risk alerts, adverse effects | Lexicon lookup | Warning phrases are clinically established — handbuilt gives full auditability, zero hallucination risk |
| `ner_detector.py` | Diseases and chemicals | Supervised NER (SciSpaCy BC5CDR) | General models miss "nephrotoxicity" — biomedical pretraining is non-negotiable for domain NLP |
| `syntactic_depth.py` | Structurally complex sentences | Dependency parsing | Length penalizes unfairly — depth captures genuine nesting complexity that vocabulary fixes miss |
| `umls_matcher.py` | Medical jargon not in common English | Morphological + knowledge graph | Suffix patterns catch drug naming conventions; QuickUMLS interface ready for GCP integration |

**Why 5 detectors instead of one:** each captures a fundamentally different complexity signal. No single paradigm catches everything — regex catches numbers but misses entities, lexicons catch warnings but miss structure, NER catches entities but misses jargon that isn't a named entity. This is the core architectural argument: complexity is multi-dimensional, detection must be too.

### 2 Baselines (`src/baselines/`)

**Baseline 1 — No simplification** — warning preservation 1.000 (safety ceiling). "Do nothing" is a legitimate clinical baseline — a system that simplifies but drops warnings is worse than no system. Regulators think this way.

**Baseline 2 — Rule-based CHV substitution** — warning preservation 0.994. Known limitation: context-unaware substitution creates grammatical artifacts ("is should not be used"). This is exactly why Stage 3 uses constrained LLM for Explanation/Generalization but keeps CHV for Substitution — auditability over fluency for safety-critical terms.

### Safety Metric (`src/evaluation/warning_preservation.py`)

Measures fraction of source warning phrases preserved after simplification, with paraphrase matching ("contraindicated" = "do not use" semantically). Why not ROUGE: ROUGE measures n-gram overlap and doesn't care if a warning was dropped — a system removing "contraindicated in patients with liver disease" can still score high ROUGE if the rest is preserved. This metric catches exactly what ROUGE misses.

Known limitation: the paraphrase map encodes annotator assumptions. Future work replaces it with BioBERT semantic similarity scoring — documented honestly in Limitations, not hidden.

---

## PRELIMINARY RESULTS (submitted)

**Complexity detection on PLABA training set (n=635 abstracts):**

| Detector | Abstracts flagged | % |
|---|---|---|
| UMLS jargon | 558 | 87.9% |
| Syntactic complexity | 525 | 82.7% |
| Numerical expressions | 387 | 60.9% |
| Warning cues | 210 | 33.1% |

**Complexity co-occurrence:**

| Detectors firing | Abstracts | % |
|---|---|---|
| 0 | 9 | 1.4% |
| 1 | 67 | 10.6% |
| 2 | 173 | 27.2% |
| 3 | 277 | 43.6% |
| 4 | 109 | 17.2% |

**Key finding:** 60.8% of abstracts exhibit 3+ simultaneous complexity types — biomedical complexity is not single-dimensional. This is the empirical justification for the multi-detector design.

**Baseline safety results (random 50-sentence sample, seed=42):**

| System | SARI↑ | FKGL↓ | Compression | Warn. Pres. (random, n=50) | Warn. Pres. (stress test, n=30) |
|---|---|---|---|---|---|
| No simplification | 16.82 | 15.31 | 1.00 | 1.000 | 1.000 |
| Rule-based CHV | 18.66 | 15.17 | 1.02 | 1.000 | 0.994 |
| Direct LLM (FLAN-T5) | 22.28 | 14.89 | 0.945 | 1.000 | 0.950 |

**Critical methodological finding:** random sampling alone (n=50) showed all baselines at perfect warning preservation — a misleading result caused by low statistical power on a rare event (33.1% base rate of warning-bearing sentences). Only a warning-**stratified** stress test (n=30, restricted to warning-bearing sentences) revealed that unguided LLM prompting is the only baseline that drops safety-critical content. This became a core methodological contribution of the paper: **safety-critical NLP systems require stratified evaluation, not random sampling, to reliably detect rare failure modes.**

**Classifier vs. literature:** our TF-IDF+LR operation classifier (0.424 macro-F1, pseudo-labeled) falls within and slightly above the range reported by PLABA's own creators for this exact task (Ondov et al. 2025: human inter-annotator agreement 0.46 F1, baseline classifiers 0.33–0.39 macro-F1) — a genuine external validation point, not just an internal majority-baseline comparison.

---

## BUG LOG — root-caused, not just patched (interview material)

### Bug 1 — numpy/thinc binary incompatibility
`ValueError: numpy.dtype size changed, may indicate binary incompatibility.` Caused by installing packages incrementally into a conda environment — a later `numpy` upgrade broke thinc's compiled Cython extensions (`numpy_ops.pyx`), which were compiled against a specific numpy ABI. Fixed by pinning `numpy<2.0`. **Lesson:** ML environments need atomic, versioned rebuilds — never `pip install` one package into a working environment without checking what it silently upgrades underneath. Root cause documented in `environment.yml`.

### Bug 2 — silent spaCy model/version mismatch
`syntactic_depth.py`'s dependency-parser fallback (`try: spacy.load(...) except Exception: USE_SPACY = False`) silently swallowed a `RegistryError` caused by a spaCy 3.8.0 model running under spaCy 3.6.1, degrading to a comma-counting heuristic that returned plausible-but-wrong depth scores (0 for real, structured sentences) with zero visible error. Diagnosed by running the parser directly and inspecting actual dependency arcs rather than trusting the printed output. Fixed by pinning `en_core_web_sm==3.6.0` to match the installed spaCy version, and by making the except block emit a visible `warnings.warn()` instead of failing silently. **Lesson:** every ML fallback path must announce itself — a silent fallback is more dangerous than a crash because it ships wrong numbers instead of stopping you.

### Bug 3 — abstract-level vs. sentence-level data granularity mismatch
`run_evaluation.py` read `val.csv`'s `input_text` column directly (abstract-level, mean 230 words, up to 822) and treated each row as a single sentence when evaluating baseline3 (FLAN-T5 generation). This produced a catastrophic-looking result (compression ratio 0.137, warning preservation 0.176) that looked like a genuine safety finding but was actually a data pipeline bug: the model was pattern-matching to title/abstract-generation instead of patient-facing sentence simplification, because it was given a 200-word block under a "simplify this sentence" prompt. Meanwhile `pseudo_labeler.py`'s `build_pairs_from_plaba_json()` had correctly built true sentence-level pairs from `data.json` all along — two files silently disagreed on what "one row" meant, and only the generation-sensitive baseline exposed it. Fixed by making `run_evaluation.py` reuse the same sentence-level builder as the pseudo-labeler, establishing one shared definition of "a validation sample" across the codebase. **Lesson:** never let two files independently define the same data contract — one shared builder, always; and verify data granularity assumptions explicitly and early, since they silently propagate through every downstream file that assumes them.

### Standing rule adopted from these three bugs
Every new script's `if __name__ == "__main__":` block must print input shape, input type, and one real worked example before any result is trusted for the paper. All three bugs above would have been caught in minutes at the point of introduction instead of hours later, buried under multiple layers of downstream code.

---

## KEY ENGINEERING DECISIONS

| Decision | Why |
|---|---|
| `.py` modules, not notebooks | Importable by teammates, version controlled, production-grade |
| Strategy pattern in UMLS matcher | QuickUMLS swaps in later without changing the pipeline interface |
| Word boundary regex in CHV substitution | Prevents substring matches ("intravenously" → "into the veinly") |
| Paraphrase map in warning-preservation metric | Catches semantic equivalents without ML — bounded, auditable bias, documented as a limitation |
| Bigram detection in UMLS heuristic | Multi-word concepts ("hepatocellular carcinoma") are missed by unigram matching alone |
| Stopword filter on bigrams | Prevents false positives at span boundaries ("and hepatotoxicity") |
| Two-table safety reporting → merged to one table with footnote (final) | Honest reporting of both random and stratified samples without doubling table overhead in a page-constrained ACL format |
| PLABA's own taxonomy (Attal et al.), not SALSA, cited for the 3-operation scope | Caught via close reading of Ondov et al. 2025 — SALSA is a general edit-evaluation framework, not PLABA's taxonomy source. Correcting this also surfaced a genuine external classifier benchmark to compare against |

---

## LITERATURE REVIEW — 6 Papers (Prelim, Fully Situated)

1. **Miyata et al. 2025 — EhiMeNLP (TSAR 2025 winner)** — generate-then-rerank across 28 model-prompt combinations; their own ablation shows gains come from diversity/selection, not operation-level understanding. Motivates our diagnostic framing as the actual missing ingredient.
2. **Alva-Manchego et al. 2025 — TSAR 2025 Shared Task Findings** — across 48 submissions/20 teams, none classified complexity type before generating; organizers explicitly call for expert-in-the-loop validation because no metric detects a fluent-but-unsafe output. Our preservation metrics directly answer this call.
3. **Papandreou et al. 2025 — Medical Jargon Detection** — jargon detectors transfer poorly across datasets with different annotation objectives (33.71% entity F1 MedReadMe→PLABA); their jargon-aware LLM prompts introduce factual errors a lookup table cannot produce — justifies CHV over jargon-surfacing prompts.
4. **Rahman & Lybarger 2025 — SPQA** — stylistic perturbations degrade LLM QA correctness/completeness even when fluency stays stable; external evidence that surface metrics miss real content degradation, same gap our safety metrics target.
5. **Ondov et al. 2025 — Lessons from the PLABA Track** — the dataset creators' own reflection; provided the correct taxonomy source (not SALSA), the real operation distribution (Substitution 64.7%/Explanation 19.3%/Generalization 6.3%/Omission 9.1%/Exemplification 0.6%), and a genuine external classifier benchmark (0.33–0.39 macro-F1) our own classifier beats.
6. **Neumann et al. 2019 — SciSpaCy** — justifies the NER detector's biomedical-pretraining requirement directly.

*(Also contributed but reassigned/condensed into team sections: Knappich et al. 2023 BoschAI, Cripwell et al. 2022, Devaraj et al. 2021, Xia et al. 2025 JEBS, Hayakawa et al. 2025 UoL-UPF — cited across Related Work as shared team citations.)*

---

## FULL PAPER — ACTIVE WORK

### My remaining tasks

1. **QuickUMLS full integration** — replace the morphological fallback with real UMLS lookup. Blocked on GCP credits (UMLS install is ~35GB; needs Cloud Storage + a small Compute Engine VM, not local disk — learned this the hard way after hitting `NoSpaceLeftError` twice this sprint on a ~2GB-free local drive).
2. **Entity preservation metric** (`src/evaluation/entity_preservation.py`) — same pattern as `warning_preservation.py`, built on top of `ner_detector.py`.
3. **Numerical preservation metric** (`src/evaluation/numerical_preservation.py`) — same pattern, built on top of `numerical_extractor.py`.
4. **Human evaluation coordination** — 30-50 sampled pairs, 5-point rubric (correctness, completeness, readability, actionability, safety). Given Bug 3's lesson, the sample must be explicitly warning-stratified, not purely random, to actually test the safety dimension.
5. **Infra documentation** — `environment.yml` gets `numpy<2.0` pinned explicitly; `plaba_loader.py` docstring gets an explicit note on abstract-level vs. sentence-level granularity, so Bug 3 cannot recur for a new contributor.
6. **Complexity co-occurrence deeper analysis** — extend beyond the 5-way count table into per-pair correlation (which two detectors co-occur most, and does that pairing predict a specific operation type).

### Team dependencies I'm tracking

- Sophakotra's operation router needs my entity/numerical preservation metrics finalized before full-system evaluation can run.
- Zihao's full test-set evaluation should use the same stratified-sampling discipline as my Bug 3 fix — flagged directly in his task brief.
- Rishabh's readability-vs-safety scatter plot is the single most important chart in the final paper; it needs Zihao's full-corpus numbers before it can be built for real (currently placeholder-able with prelim numbers).

### Standing process rule for full-paper phase
Daily incremental verification, not sprint-and-crash. Every script prints its sanity check before any number is trusted. Every cross-file data assumption (like "what counts as one row") gets written down in one place the first time it's decided, not rediscovered under deadline pressure.