# Sruthilaya — Complexity Detection & Safety Evaluation Lead

**Branch:** `main` (foundation branch — team builds on top of this)
**Role:** Stage 1 (Complexity Detection) + Safety Metrics + Repo/Paper Integration Owner

---

## PROGRESS CHECK — pending items (updated this session)

### My pending items

- [ ] Entity preservation metric (`src/evaluation/entity_preservation.py`)
- [ ] Numerical preservation metric (`src/evaluation/numerical_preservation.py`)
- [ ] Human evaluation coordination — warning-stratified sample, 5-point rubric
- [ ] `environment.yml` (`numpy<2.0` pin) + `plaba_loader.py` docstring (granularity note)
- [ ] Per-pair co-occurrence correlation (deeper analysis beyond the base table)
- [ ] Audit `warning_lexicon.py` / `numerical_extractor.py` for false positives (same rigor as UMLS/NER got)
- [ ] Test `syntactic_depth.py`'s heuristic fallback (`_approximate_depth`) logic itself — never independently verified
- [ ] Revoke `Storage Object Admin` IAM grant on compute service account
- [ ] Decide on / execute `umls-build-vm` teardown (index + data already backed up to GCS; setup scripts tested — safe to delete)
- [ ] Send team message: n=635 vs. n=530 training set discrepancy
- [ ] Send team message: possible silent-fallback bug in prelim's syntactic depth number (82.7%)
- [ ] Review/polish paper section drafts (Abstract, page-budget trim once teammates' sections land)

### Pending from teammates (for the shared ACL paper)

| Owner | Owns |
|---|---|
| Sophakotra | Operation classifier results, generation pipeline, Main Idea (top-level architecture) section |
| Zihao | Full-corpus evaluation, model inference results |
| Rishabh | Readability-vs-safety scatter plot, Related Work condensing help |
| Whole team | Introduction, Problem section, Conclusion, Ethics Statement |

---

## STATUS

| Phase | Status |
|---|---|
| Prelim Report | ✅ Submitted — ACL format, 4 pages, all sections verified |
| QuickUMLS Full Integration | ✅ Complete — real UMLS 2026AA indexed, validated, integrated |
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
| `umls_matcher.py` | Medical jargon not in common English | Real UMLS lookup via QuickUMLS (was: morphological fallback only) | Now fully wired to real UMLS 2026AA (10.7M concepts) — see full session log below |

**Why 5 detectors instead of one:** each captures a fundamentally different complexity signal. No single paradigm catches everything — regex catches numbers but misses entities, lexicons catch warnings but miss structure, NER catches entities but misses jargon that isn't a named entity. This is the core architectural argument: complexity is multi-dimensional, detection must be too.

### 2 Baselines (`src/baselines/`)

**Baseline 1 — No simplification** — warning preservation 1.000 (safety ceiling). "Do nothing" is a legitimate clinical baseline — a system that simplifies but drops warnings is worse than no system. Regulators think this way.

**Baseline 2 — Rule-based CHV substitution** — warning preservation 0.994. Known limitation: context-unaware substitution creates grammatical artifacts ("is should not be used"). This is exactly why Stage 3 uses constrained LLM for Explanation/Generalization but keeps CHV for Substitution — auditability over fluency for safety-critical terms.

### Safety Metric (`src/evaluation/warning_preservation.py`)

Measures fraction of source warning phrases preserved after simplification, with paraphrase matching ("contraindicated" = "do not use" semantically). Why not ROUGE: ROUGE measures n-gram overlap and doesn't care if a warning was dropped — a system removing "contraindicated in patients with liver disease" can still score high ROUGE if the rest is preserved. This metric catches exactly what ROUGE misses.

Known limitation: the paraphrase map encodes annotator assumptions. Future work replaces it with BioBERT semantic similarity scoring — documented honestly in Limitations, not hidden.

---

## PRELIMINARY RESULTS (submitted)

**Complexity detection on PLABA training set (n=635 abstracts, prelim split):**

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

## FULL PAPER RESULTS — real UMLS + full 5-detector pass (n=530 abstracts, current split)

**⚠️ Sample size note:** the current, verified training split is **530 abstracts**, not the prelim's 635 (see "Data granularity investigation" below for the full root-cause trace). Numbers below are not directly comparable to the prelim table above — different n, different (corrected) methodology.

**Detector flag rates:**

| Detector | Morphological (heuristic fallback) | Real UMLS (QuickUMLS, full pipeline) |
|---|---|---|
| NER | — | 98.7% |
| Warning cues | — | 32.1% |
| Syntactic complexity | 94.0%* | 94.0%* |
| Numerical expressions | — | 60.2% |
| UMLS/jargon | 86.8% | **100.0%** |

*Syntactic complexity required a bug fix mid-session (see bug log) — see caveat below.

**Co-occurrence (real UMLS backend, n=530):**

| Detectors firing | Abstracts | % |
|---|---|---|
| 2 | 10 | 1.9% |
| 3 | 154 | 29.1% |
| 4 | 272 | 51.3% |
| 5 | 94 | 17.7% |

**98.1% of abstracts have 3+ detectors firing** (vs. 90.8% under the morphological backend, vs. 60.8% in the prelim — see caveats below on why these aren't directly comparable).

**Headline empirical result: real UMLS lookup achieves 100% jargon coverage on the training set vs. 86.8% for the morphological heuristic fallback** — a concrete, validated demonstration of exactly the coverage-gap argument the paper is built around. Validated via:
1. Cherry-picked test sentences (hepatocellular carcinoma, sorafenib, nephrotoxicity, etc.) — all correctly matched with real CUIs
2. Random sample of 10 abstracts, full-text review — every flagged term was genuine biomedical jargon (implantable cardioverter-defibrillator, hyporeninemic hypoaldosteronism, Echinococcus granulosus, gabapentinoids, etc.), no residual false positives after filtering fixes

**Known remaining limitation (documented, not hidden):** "urine output" still gets flagged as jargon by the real-UMLS pipeline — a 2-word phrase that's borderline common English but not caught by the single-word frequency filter (see Engineering Decisions below). Judged acceptable given diminishing returns on further filtering; worth one line in the paper's Limitations section, same honest-disclosure pattern as the CHV paraphrase-map limitation.

---

## BUG LOG — root-caused, not just patched (interview material)

### Bug 1 — numpy/thinc binary incompatibility
`ValueError: numpy.dtype size changed, may indicate binary incompatibility.` Caused by installing packages incrementally into a conda environment — a later `numpy` upgrade broke thinc's compiled Cython extensions (`numpy_ops.pyx`), which were compiled against a specific numpy ABI. Fixed by pinning `numpy<2.0`. **Lesson:** ML environments need atomic, versioned rebuilds — never `pip install` one package into a working environment without checking what it silently upgrades underneath. Root cause documented in `environment.yml`.

### Bug 2 — silent spaCy model/version mismatch
`syntactic_depth.py`'s dependency-parser fallback (`try: spacy.load(...) except Exception: USE_SPACY = False`) silently swallowed a `RegistryError` caused by a spaCy 3.8.0 model running under spaCy 3.6.1, degrading to a comma-counting heuristic that returned plausible-but-wrong depth scores (0 for real, structured sentences) with zero visible error. Diagnosed by running the parser directly and inspecting actual dependency arcs rather than trusting the printed output. Fixed by pinning `en_core_web_sm==3.6.0` to match the installed spaCy version, and by making the except block emit a visible `warnings.warn()` instead of failing silently. **Lesson:** every ML fallback path must announce itself — a silent fallback is more dangerous than a crash because it ships wrong numbers instead of stopping you.

**Recurrence during full-paper QuickUMLS integration (see full session log below):** this exact same bug class recurred — same file, same bare `except Exception`, different trigger (running the orchestrator inside `ner_env`, which lacks `en_core_web_sm`). Produced a 10.8% syntactic complexity rate instead of the real 94.0% — an 8x understatement, silently. **This raises an open question about the prelim's own 82.7% syntactic number — flagged to the team, not yet resolved (see "Open items for team" below).**

### Bug 3 — abstract-level vs. sentence-level data granularity mismatch
`run_evaluation.py` read `val.csv`'s `input_text` column directly (abstract-level, mean 230 words, up to 822) and treated each row as a single sentence when evaluating baseline3 (FLAN-T5 generation). This produced a catastrophic-looking result (compression ratio 0.137, warning preservation 0.176) that looked like a genuine safety finding but was actually a data pipeline bug: the model was pattern-matching to title/abstract-generation instead of patient-facing sentence simplification, because it was given a 200-word block under a "simplify this sentence" prompt. Meanwhile `pseudo_labeler.py`'s `build_pairs_from_plaba_json()` had correctly built true sentence-level pairs from `data.json` all along — two files silently disagreed on what "one row" meant, and only the generation-sensitive baseline exposed it. Fixed by making `run_evaluation.py` reuse the same sentence-level builder as the pseudo-labeler, establishing one shared definition of "a validation sample" across the codebase. **Lesson:** never let two files independently define the same data contract — one shared builder, always; and verify data granularity assumptions explicitly and early, since they silently propagate through every downstream file that assumes them.

### Standing rule adopted from these three bugs
Every new script's `if __name__ == "__main__":` block must print input shape, input type, and one real worked example before any result is trusted for the paper. All three bugs above would have been caught in minutes at the point of introduction instead of hours later, buried under multiple layers of downstream code.

---

## FULL SESSION LOG — QuickUMLS Full Integration (this session)

This was a genuinely dense infra + ML-engineering debugging session — documenting the full chain since root causes matter more than patches, per the standing bug-log philosophy.

### Infra setup
- GCP project + Cloud Storage bucket (`nlp-text-simplification-umls`) provisioned as durable storage, decoupled from any single VM's disk
- Spot VM (`umls-build-vm`, e2-standard-4, 100GB standard persistent disk, asia-south1-c) provisioned for the UMLS index build — Spot pricing (~60-90% cheaper) appropriate since this is a short, one-off build job, not a persistent service
- **IAM/scopes lesson:** hit a real two-layer permissions model. VM **scopes** (`devstorage.read_only` by default) control what *categories* of API calls a VM is allowed to attempt; separately, the VM's **service account IAM role** controls what that identity is actually permitted to do. Had `devstorage.read_only` scope but needed `read_write` — fixed by editing VM scopes (requires VM stop/restart) to `cloud-platform` (full access), *and* separately granting the default compute service account (`146593091895-compute@developer.gserviceaccount.com`) the `Storage Object Admin` IAM role on the bucket. Neither layer alone was sufficient. **To do: revoke this IAM grant once VM is torn down (permission debt is a common real-world cloud security finding).**

### Bug 4 — `leveldb` package incompatible with Python 3.13
QuickUMLS's install pulled in the `leveldb` PyPI package as a hard dependency (`import leveldb` at module top-level in `toolbox.py`, unconditional even when using the `unqlite` backend). The package's C++ binding (`leveldb_object.cc`) calls `PyUnicode_AS_UNICODE`, a Python C-API function removed in Python 3.12+. Debian 13 (Trixie) ships Python 3.13 by default with no 3.11 package available via apt — fighting for an older Python wasn't viable.

**Fix:** QuickUMLS already supports `unqlite` as an alternative backend (confirmed by reading `toolbox.py`'s actual `__init__` branching logic, not just the README, which didn't match the installed version's constructor signature). Since `leveldb` is still imported unconditionally at module load even when unused, created an empty stub `leveldb.py` in site-packages purely to satisfy the import — the real `leveldb.LevelDB(...)` branch is never executed when `database_backend='unqlite'`.

### Bug 5 — `quickumls-simstring`'s SWIG loader uses removed `imp` module
Same Python-3.12+-removal pattern, different package: the SWIG-generated `_simstring` wrapper used `imp.find_module`/`imp.load_module` (removed in 3.12+). Confirmed via direct inspection of the generated file, then patched the legacy loader to a direct `from . import _simstring` (relative import — first attempt used a plain `import _simstring`, which failed because the compiled `.so` lives inside the package directory, not at top-level `site-packages`).

**Standing artifact from Bugs 4+5:** `infra/setup_quickumls.sh` — fully scripted, idempotent, glob-based version-path detection (won't break on future Python version bumps) so this entire chain doesn't need to be re-debugged on a fresh VM or by a teammate.

### QuickUMLS install validation (dry run before spending 3-day UMLS license wait)
No official QuickUMLS demo dataset exists in the pip package or GitHub repo (contrary to initial assumption). Built a small synthetic `MRCONSO.RRF`/`MRSTY.RRF` matching the real schema, ran the full install → matcher pipeline against it, confirmed the matcher correctly loads and returns real CUI matches before the actual UMLS data was even available — de-risked the real 3-day-later install.

### Real UMLS install
- Used NLM's API-based download endpoint (`uts-ws.nlm.nih.gov/releases` → `downloadUrl`, then `/download?url=...&apiKey=...`) directly from the VM, bypassing an unstable browser-based download that kept failing on the 5.4GB file
- **Correct `releaseType` value is `umls-full-release`**, not the more intuitive-seeming `umls-release` (404s) — worth noting for next time
- Unzipped nested/chunked `.gz` archives (`MRCONSO.RRF.aa.gz`, `.ab.gz`, `.ac.gz` — concatenate `.gz` streams *before* gunzip, not after, since gzip supports concatenated streams)
- Installed with `-L -U -E ENG -d unqlite` flags: lowercase + unicode normalization for recall, English-only to cut index size/build time, unqlite backend per Bug 4's fix
- **10,755,691 concepts indexed in ~19 minutes** on e2-standard-4
- Backed up to GCS bucket immediately (5.2GB, verified byte-for-byte size match between local and bucket copies)

### Bug 6 — QuickUMLS matcher: three layered false-positive mechanisms
This was the most substantial engineering problem of the session — validated against real data revealed three genuinely distinct failure modes, each needing a different, principled fix:

1. **Duplication:** `matcher.match(..., best_match=True)` still returns every candidate concept per span, not just the best one, despite the flag's name. Fixed by taking `max(match_group, key=similarity)` per span manually.
2. **Semtype-level false positives:** default UMLS lookup matches against all 127+ semantic types, including generic/administrative concepts ("adverse effects," "primary," "performed") that are technically UMLS concepts but not patient-facing jargon. Fixed via `accepted_semtypes` restricted to 8 clinically relevant types (Disease/Syndrome, Neoplastic Process, Pharmacologic Substance, etc.), plus `threshold=0.85` to cut weak/loose matches, plus a small manual `COMMON_ENGLISH_TERMS` extension for known specific offenders.
3. **Lexical-frequency false positives (the deepest one):** "control" matches a real UMLS chemical/substance concept (`T121`, similarity 1.0) — no semtype restriction can fix this, since `T121` is exactly the category needed to catch real drug names like "sorafenib." First attempt used NLTK's `words` corpus as a broad common-English filter — **this was itself wrong**, since NLTK's corpus is a raw dictionary, not a frequency list, and classified genuine jargon like "hyperglycemia" as valid English, silently filtering out real jargon. **Final fix:** `wordfreq` package's Zipf frequency score, empirically validated against 16 known common/jargon words to find a clean threshold (3.8) — separates "control" (5.40), "determine" (4.63) from "carcinoma" (3.15), "hyperglycemia" (2.31), applied only to single-word matches (multi-word phrases bypass this filter).

**Validation methodology:** didn't just trust the fix on the original 5 cherry-picked test sentences — re-validated against a random sample of 10 real training abstracts, full manual review of every flagged term, confirming genuine jargon throughout with no residual false positives.

### Bug 7 — spaCy version conflict: QuickUMLS vs. scispaCy NER, unresolvable in one environment
`en_ner_bc5cdr_md` (scispaCy's biomedical NER model) requires `spaCy >=3.4.1,<3.5.0`. QuickUMLS's `core.py` loads `spacy.load()` at match-time (not just install-time), and the existing `umls_env` already has spaCy 3.8.14 (a side-effect of QuickUMLS's own `en_core_web_sm` dependency). These are genuinely incompatible in one Python process — not fixable by patching, unlike Bugs 4/5.

**Fix — proper environment isolation, not another patch:** created a fully separate `ner_env` (Miniconda, Python 3.9 — needed since Debian 13 has no apt-installable Python <3.13, and pip couldn't compile `blis`/spaCy's Cython extensions from source on 3.13). Orchestrator bridges to it via `subprocess`, batched once for all ~5,300 sentences (not per-sentence, to avoid paying Python/spaCy startup cost repeatedly).

**Sub-bugs surfaced while making `setup_ner_env.sh` reproducible (worth noting — reproducibility scripting itself surfaced real issues):**
- Anaconda now requires explicit Terms of Service acceptance (`conda tos accept ...`) before channel use — a recent policy change, not documented anywhere obvious
- `scispacy` (unpinned) installs latest, which now requires `thinc>=8.3.12`, which requires Python ≥3.10 — incompatible with the Python 3.9 env needed for `blis` wheel availability. **Fixed by pinning `scispacy==0.5.4`** to match the model version.
- `spacy.load('en_ner_bc5cdr_md')` by name doesn't reliably work even when installed correctly — scispaCy models are proper Python packages with their own `.load()` function; **`import en_ner_bc5cdr_md; nlp = en_ner_bc5cdr_md.load()`** is the reliable pattern.
- Shell environment layering: `venv`-based activation (`umls_env`) and `conda activate` don't cleanly replace each other — both can stack in the prompt (`(ner_env) (umls_env) (base)`), and which Python actually runs depends on `$PATH` order, not which "looks" active. Diagnosed via `which -a python` showing all stacked interpreters; resolved by always starting fresh shell sessions when switching environments rather than trying to layer activations.

### Data granularity investigation — training set is 530 abstracts, not 635
While building the orchestrator script to reproduce the prelim's co-occurrence table, discovered `umls_matcher.py` was never actually wired into any pipeline — it was standalone, unused by any committed script. Building the real orchestrator required first resolving a genuine data-structure confusion:
- `data.json`'s top-level keys are NOT PMIDs — they're sequential wrapper indices (`"1"` through `"75"`), each containing multiple grouped PMIDs plus two metadata fields (`question`, `question_type`) as sibling keys
- After correctly excluding metadata keys: 749 real PMIDs in `data.json`
- `train.csv` (531 PMIDs) vs. derived (`all - val - test` = 530 PMIDs) — traced the single discrepancy to one blank-PMID row in `train.csv` (row 622, real text content about chronic cough, but missing PMID) — not a bug in either derivation method, just one incomplete source row
- **Neither 530 nor 531 matches the prelim's reported n=635** — flagged to team, unresolved (see Open Items below)

---

## OPEN ITEMS FOR TEAM (flagged, not yet resolved)

1. **n=635 vs. n=530 discrepancy** — prelim reported 635 training abstracts; current, carefully-verified derivation (both via `train.csv` and via `data.json` exclusion) gives 530. Possible causes: val/test split finalized after prelim, or looser earlier exclusion criterion. Message drafted, pending send.
2. **Possible silent-fallback bug affecting the prelim's syntactic depth number (82.7%)** — same bare-`except Exception` pattern that (in this session) caused an 8x understatement (10.8% vs. real 94.0%) when spaCy's model wasn't available. Worth checking whether whatever environment produced the prelim's 82.7% had `en_core_web_sm` correctly installed, or silently hit the heuristic fallback. Message drafted, pending send.
3. **`train.csv` has one row with a blank PMID** (row 622, chronic cough content) — minor data-quality gap, worth a one-line fix or note in `pseudo_labeler.py`'s docstring.

---

## KEY ENGINEERING DECISIONS

| Decision | Why |
|---|---|
| `.py` modules, not notebooks | Importable by teammates, version controlled, production-grade |
| Strategy pattern in UMLS matcher | QuickUMLS swaps in later without changing the pipeline interface — paid off directly this session, `UMLS_BACKEND` env var flip required zero pipeline code changes |
| Word boundary regex in CHV substitution | Prevents substring matches ("intravenously" → "into the veinly") |
| Paraphrase map in warning-preservation metric | Catches semantic equivalents without ML — bounded, auditable bias, documented as a limitation |
| Bigram detection in UMLS heuristic | Multi-word concepts ("hepatocellular carcinoma") are missed by unigram matching alone |
| Stopword filter on bigrams | Prevents false positives at span boundaries ("and hepatotoxicity") |
| Two-table safety reporting → merged to one table with footnote (final) | Honest reporting of both random and stratified samples without doubling table overhead in a page-constrained ACL format |
| PLABA's own taxonomy (Attal et al.), not SALSA, cited for the 3-operation scope | Caught via close reading of Ondov et al. 2025 — SALSA is a general edit-evaluation framework, not PLABA's taxonomy source. Correcting this also surfaced a genuine external classifier benchmark to compare against |
| `accepted_semtypes` restriction + `wordfreq` Zipf threshold (3.8) for QuickUMLS filtering | Semtype restriction alone can't separate "is a UMLS concept" from "is jargon" — some common English words (`control`) coincidentally match real, relevant-semtype UMLS concepts. Frequency-based filtering is the principled fix; NLTK's raw wordlist was tried first and rejected as too broad (misclassified real jargon like `hyperglycemia` as common) |
| Separate conda env (`ner_env`) + subprocess bridge for NER, rather than forcing one shared environment | spaCy version requirements for QuickUMLS (3.8.x, runtime dependency) and scispaCy's BC5CDR model (<3.5, hard pin) are genuinely incompatible in one process — isolation, not patching, is the correct fix, same category as needing Python 3.9 (via conda) vs. Debian 13's unavailable Python 3.11 |
| Batch NER subprocess call once for all sentences, not per-sentence | Avoids paying Python/spaCy startup cost (~1-2s) hundreds of times — amortized to a single call |

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

1. ~~**QuickUMLS full integration**~~ ✅ **Done this session** — real UMLS 2026AA indexed, matcher tuned and validated (3 layered filtering fixes), full 5-detector orchestrator built and run, results committed.
2. **Entity preservation metric** (`src/evaluation/entity_preservation.py`) — same pattern as `warning_preservation.py`, built on top of `ner_detector.py`. Not started.
3. **Numerical preservation metric** (`src/evaluation/numerical_preservation.py`) — same pattern, built on top of `numerical_extractor.py`. Not started.
4. **Human evaluation coordination** — 30-50 sampled pairs, 5-point rubric (correctness, completeness, readability, actionability, safety). Given Bug 3's lesson, the sample must be explicitly warning-stratified, not purely random, to actually test the safety dimension. Not started.
5. **Infra documentation** — `environment.yml` gets `numpy<2.0` pinned explicitly; `plaba_loader.py` docstring gets an explicit note on abstract-level vs. sentence-level granularity. **Partially done** — `infra/setup_quickumls.sh` and `infra/setup_ner_env.sh` now exist, tested, committed. `environment.yml` and `plaba_loader.py` docstring updates still pending.
6. **Complexity co-occurrence deeper analysis** — extend beyond the 5-way count table into per-pair correlation (which two detectors co-occur most, and does that pairing predict a specific operation type). **Base co-occurrence table done this session** (both morphological and real-UMLS backends); per-pair correlation breakdown not yet built.

### New/updated tasks from this session

7. **NOT YET AUDITED:** `warning_lexicon.py` and `numerical_extractor.py`'s outputs were trusted throughout this session's orchestrator runs but never independently spot-checked for false positives the way UMLS and NER were. Worth the same random-sample manual review before treating their flag rates as fully validated.
8. **NOT YET TESTED:** `syntactic_depth.py`'s heuristic fallback (`_approximate_depth`) itself — we validated the real spaCy path extensively, but never checked whether the fallback path's own logic is even correct, since we've been actively avoiding triggering it.
9. **GCP cleanup:** revoke `Storage Object Admin` IAM grant on `146593091895-compute@developer.gserviceaccount.com` once VM is torn down (flagged mid-session, action pending).
10. **VM teardown:** `umls-build-vm` to be deleted once this session's results are fully recorded — UMLS index + PLABA data both verified safely backed up to GCS bucket; setup scripts (`infra/setup_quickumls.sh`, `infra/setup_ner_env.sh`) both tested end-to-end, so rebuild is a known, fast path if ever needed.

### Team dependencies I'm tracking

- Sophakotra's operation router needs my entity/numerical preservation metrics finalized before full-system evaluation can run. **Now also unblocked more broadly** — real UMLS matcher is fully wired and validated, so any downstream component depending on `umls_matcher.py`'s output can now trust real jargon detection, not just the heuristic fallback.
- Zihao's full test-set evaluation should use the same stratified-sampling discipline as my Bug 3 fix — flagged directly in his task brief.
- Rishabh's readability-vs-safety scatter plot is the single most important chart in the final paper; it needs Zihao's full-corpus numbers before it can be built for real (currently placeholder-able with prelim numbers).

### Standing process rule for full-paper phase
Daily incremental verification, not sprint-and-crash. Every script prints its sanity check before any number is trusted. Every cross-file data assumption (like "what counts as one row") gets written down in one place the first time it's decided, not rediscovered under deadline pressure.