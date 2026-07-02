# Sruthilaya — Stage 1 Complete

**Branch:** `main` (you set the foundation everyone builds on)  
**Status:** All prelim deliverables done

---

## What I Built — Stage 1 Complete

### 5 Complexity Detectors (`src/complexity/`)

| File | What it detects | NLP paradigm | Why this paradigm |
|---|---|---|---|
| `numerical_extractor.py` | Dosages, risk ratios, p-values, CIs | Regex | Numerical patterns are finite and rule-governed — no ML needed, 100% recall on known formats |
| `warning_lexicon.py` | Contraindications, risk alerts, adverse effects | Lexicon lookup | Warning phrases are clinically established — handbuilt gives full auditability, zero hallucination risk |
| `ner_detector.py` | Diseases and chemicals | Supervised NER (SciSpaCy BC5CDR) | General models miss "nephrotoxicity" — biomedical pretraining is non-negotiable for domain NLP |
| `syntactic_depth.py` | Structurally complex sentences | Dependency parsing | Length penalizes unfairly — depth captures genuine nesting complexity that vocabulary fixes miss |
| `umls_matcher.py` | Medical jargon not in common English | Morphological + knowledge graph | Suffix patterns catch drug naming conventions; QuickUMLS interface ready for GCP integration |

**Why 5 detectors instead of one?**
Each captures a fundamentally different complexity signal.
No single paradigm catches everything. Regex catches numbers 
but misses entities. Lexicons catch warnings but miss structure.
NER catches entities but misses jargon that isn't a named entity.
This is the core architectural argument — complexity is 
multi-dimensional, detection must be too.

### 2 Baselines (`src/baselines/`)

**Baseline 1 — No simplification** (`baseline1_no_simplification.py`)
- Warning preservation: 1.000 (confirmed safety ceiling)
- Purpose: upper bound anchor — no system should outscore this on safety
- Enterprise AI insight: "do nothing" is a legitimate clinical baseline.
  A system that simplifies but drops warnings is worse than no system.
  Regulators think this way. Now you do too.

**Baseline 2 — Rule-based CHV substitution** (`baseline2_rule_based_chv.py`)
- Warning preservation: 0.994
- Performs CHV term substitution + sentence splitting
- Known limitation: context-unaware substitution creates grammatical 
  artifacts ("is should not be used"). Documents why Stage 3 uses 
  constrained LLM for Explanation/Generalization but keeps CHV for 
  clean Substitution — auditability over fluency for safety-critical terms.

### Safety Metric (`src/evaluation/warning_preservation.py`)

Measures fraction of source warning phrases preserved after simplification.
Includes paraphrase matching — "contraindicated" = "do not use" semantically.

**Why not ROUGE?**
ROUGE measures n-gram overlap — doesn't care if a warning was dropped.
A system removing "contraindicated in patients with liver disease" can 
still score high ROUGE if the rest is preserved. 
This metric catches exactly what ROUGE misses: safety-critical information loss.

Known limitation: paraphrase map encodes annotator assumptions — 
future work replaces with BioBERT semantic similarity scoring.
Documented in paper as honest limitation, not hidden.

---

## Preliminary Results

**Complexity detection on PLABA train set (n=635):**

| Detector | Sentences flagged | % |
|---|---|---|
| UMLS jargon | 558 | 87.9% |
| Syntactic complexity | 525 | 82.7% |
| Numerical expressions | 387 | 60.9% |
| Warning cues | 210 | 33.1% |

**Complexity co-occurrence:**

| Detectors firing | Sentences | % |
|---|---|---|
| 0 | 9 | 1.4% |
| 1 | 67 | 10.6% |
| 2 | 173 | 27.2% |
| 3 | 277 | 43.6% |
| 4 | 109 | 17.2% |

**Key finding:** 43.6% of sentences exhibit 3+ simultaneous complexity 
types — biomedical complexity is not single-dimensional.
A system handling only jargon fails on structure and numbers simultaneously.
This is the empirical justification for the multi-detector design.

**Baseline safety results:**

| Baseline | Warning preservation |
|---|---|
| No simplification (ceiling) | 1.000 |
| Rule-based CHV | 0.994 |

---

## Paper Section — Complexity Detection

**What to write (~0.5 page in prelim report):**

Open with the motivation: complexity in biomedical text is not 
one problem, it's four overlapping problems simultaneously.
A jargon term needs substitution. A dense risk statement needs 
careful number-preserving treatment. A warning phrase needs to 
survive simplification verbatim or via clinically equivalent paraphrase.
A nested clause needs structural decomposition.

Present the five detectors as a deliberate design choice — 
each addresses a distinct complexity signal, each uses the 
paradigm most appropriate for that signal.
No ML where rules suffice. No hallucination risk where 
auditability is required.

Present the EDA numbers. The 43.6% three-detector co-occurrence 
is your strongest empirical argument — it shows complexity 
is genuinely multi-dimensional in this corpus, not just assumed.

Close with the safety baseline results — 1.000 vs 0.994 sets 
the safety anchor that all subsequent systems are measured against.

---

## My Lit Review Papers (6)

To fill with PDFs when we write the lit review section:
1. Attal et al. 2023 — PLABA dataset
2. HOPE Control Paper — rule-based beats zero-shot
3. Medical Jargon Paper — RoBERTa jargon detection on PLABA
4. TSAR Shared Task Findings 2025
5. Health QA SPQA — style perturbations reduce correctness
6. Neumann et al. 2019 — SciSpaCy

---

## Key Engineering Decisions Made

| Decision | Why |
|---|---|
| `.py` modules not notebooks | Importable by teammates, version controlled, production-grade |
| Strategy pattern in UMLS matcher | QuickUMLS swaps in without changing pipeline interface |
| Word boundary regex in CHV | Prevents substring matches ("intravenously" → "into the veinly") |
| Paraphrase map in warning metric | Catches semantic equivalents without ML — bounded, auditable bias |
| Bigram detection in UMLS heuristic | "hepatocellular carcinoma" is a multi-word concept — unigrams miss it |
| Stopword filter on bigrams | Prevents "and hepatotoxicity" false positives at span boundaries |

---

## FULL PAPER — My Remaining Work

- Complete lit review entries with PDF content (6 papers)
- Human evaluation coordination (30-50 sampled pairs, 5-point rubric)
- QuickUMLS full integration when GCP credits arrive
- Complexity Detection section finalized with full results
- Co-occurrence analysis visualization