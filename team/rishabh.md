# Rishabh — Data Pipeline & Evaluation Infrastructure

**Repo:** https://github.com/ussruthilaya-lang/Operation-aware-biomedical-text-simplification
**Branch:** `feature/rishabh-data-pipeline` (merged to main)
**Role:** Data Loading, CHV Vocabulary, EDA Visualizations, Introduction/Data/Related Work Sections

---

## STATUS

| Phase | Status |
|---|---|
| Prelim Report | ✅ Delivered — reviewed, integrated into `paper/acl_latex.tex` |
| Full Paper | 🔄 In progress |

---

## WHAT'S BUILT — Prelim Phase Complete

### PLABA Data Loader (`src/data/plaba_loader.py`)

```python
def load_train()   # returns DataFrame
def load_val()     # returns DataFrame
def load_test()    # returns DataFrame
def load_all()     # combined with split column
```

Central loading point so encoding fixes and preprocessing happen once, not independently in every file. Uses `utf-8-sig` encoding to strip BOM characters from Windows-exported CSVs — a real, non-obvious bug source caught early. Path resolution is anchored to the file's own location (`os.path.dirname(os.path.abspath(__file__))`), so it works regardless of which directory a script is run from.

**One correction made during integration:** the loader's docstring did not originally flag that `train.csv`/`val.csv`'s `input_text` is **abstract-level** (mean 230 words, up to 822), not sentence-level. This ambiguity caused a real downstream bug (see Sruthilaya's Bug 3 log) when a different script assumed sentence-level granularity. Fixed by adding an explicit granularity note to the loader's docstring so this cannot silently recur.

### CHV Lookup Module (`src/data/chv_lookup.py`)

Extracted the Consumer Health Vocabulary dictionary out of `baseline2_rule_based_chv.py` into a standalone, importable module — single source of truth instead of a dictionary buried inside a baseline script.

```python
def lookup(term)         # returns plain English equivalent or None
def batch_lookup(terms)  # returns {term: equivalent}
def get_all_terms()      # returns a COPY of the full dictionary
```

**Key design choice:** `get_all_terms()` returns a copy, not the live dictionary — prevents any caller from accidentally mutating shared state. In healthcare NLP where a wrong substitution is a patient-safety issue, protecting the lookup table from silent modification is correct engineering, not paranoia.

Expanded from ~40 original terms to 80+, including adverb/morphological variants (`intravenous`/`intravenously`, `subcutaneous`/`subcutaneously`) that a naive lookup would otherwise miss. `baseline2_rule_based_chv.py` now imports from this module rather than maintaining its own copy — eliminates the class of bug where the two drift out of sync.

### EDA Visualizations (`notebooks/visualizations.ipynb`)

Three plots built from `results/complexity_prelim.csv`:
- Complexity signal bar chart (% abstracts per detector)
- Complexity co-occurrence distribution (0–4 detectors firing)
- Sentence length distribution (source vs. target word count)

Built defensively: an `as_bool()` helper normalizes True/False/1/0/"true" boolean representations that vary across export formats, and `os.makedirs(OUT, exist_ok=True)` prevents directory-not-found failures. Path resolution set to run correctly regardless of working directory (Colab vs. local).

### Paper Sections — Introduction, Data, Related Work

Introduction frames the diagnostic approach (classify first, then simplify) without over-claiming — refined during integration to explicitly point toward the stress-test evaluation (Section 5) rather than asserting "LLMs routinely drop warnings" as flat, unqualified fact.

Data section describes PLABA (750 abstracts, up to 4 references per sentence, 635/138/148 train/val/test split) and states the EDA findings. **Corrected during integration:** originally described the 635/138/148 split as "sentence pairs" — factually wrong, since these are abstract-level rows. Fixed to "abstract-level source-target pairs," with an added sentence clarifying that sentence-level analysis (pseudo-labeling, classifier training) uses a separately-built 6,307-pair sentence-level dataset from `data.json`. Same fix applied to the EDA co-occurrence table's caption and column header, which had independently made the same "sentences" vs. "abstracts" error — three separate instances of the same unresolved ambiguity, now fixed everywhere.

Related Work situates PLABA, BoschAI (Knappich et al. 2023), and BeeManc (Li et al. 2024) as prior systems and datasets, tightened during the trim pass from ~200 words per citation down to 2-3 sentences each to fit the 4-page ACL limit.

---

## LITERATURE REVIEW — Contributed Citations

1. **Attal et al. 2023 — PLABA dataset.** The dataset itself; also corrected mid-project to be the actual source of the 3-operation taxonomy (Substitution/Explanation/Generalization), not SALSA as originally cited elsewhere in the paper.
2. **Knappich et al. 2023 — BoschAI (PLABA 2023 winner).** Their finding that fine-tuned models copy input and edit conservatively directly motivates the diagnostic framing — a model that barely edits also barely distinguishes a safe substitution from a critical warning.
3. **Li et al. 2024 — BeeManc system.** LLMs with controllable attributes for the same track; source of the Lay-SciFive and BART-w-CTs checkpoints planned as full-paper neural baselines.
4. **Xu et al. 2016 — SARI metric.** Primary quality metric; rewards correct add/keep/delete against multiple references, unlike BLEU/ROUGE.
5. **Zhang et al. 2020 — BERTScore.** Complements SARI with contextual semantic similarity; neither metric captures whether a warning survived, which is the gap the team's custom preservation metrics fill.
6. **Berkman et al. 2011 — Health literacy and outcomes.** Grounds the practical clinical stakes of the whole project (low health literacy → worse outcomes), used in the Introduction's opening motivation.

---

## KEY ENGINEERING DECISIONS

| Decision | Why |
|---|---|
| `plaba_loader.py` as single import point | Eliminates N independent `pd.read_csv()` calls across the codebase — one place to fix encoding, one place to add preprocessing |
| `utf-8-sig` encoding | Windows CSV exports carry a BOM character that breaks naive UTF-8 reads; catching this early avoided a whole class of downstream string-matching bugs |
| `chv_lookup.py` extracted from baseline2 | Single source of truth — Sophakotra's operation router and baseline2 now both import the same dictionary instead of risking drift between two copies |
| `get_all_terms()` returns a copy | Prevents any caller from mutating shared vocabulary state — a correctness issue with real patient-safety consequences if it went unnoticed |
| Explicit granularity note added to loader docstring | Direct response to a real bug (abstract-level vs. sentence-level data) that cost hours to trace across three files — documenting the assumption once prevents it recurring |

---

## FULL PAPER — ACTIVE WORK

### Remaining tasks

1. **BERTScore evaluation module** (`src/evaluation/bertscore_eval.py`) — deferred from prelim. Use a biomedical-domain BERT variant so domain terms are matched fairly, not a general-purpose model.
2. **Readability-vs-safety scatter plot** — X axis SARI, Y axis warning preservation, one point per system (no-simplification, rule-based CHV, direct LLM, Lay-SciFive, BART-w-CTs, operation-aware). This is the single most important chart in the final paper — it's the visual proof of the whole thesis. Needs Zihao's full test-set numbers before it can be built for real; a placeholder version using prelim numbers can be drafted now.
3. **Per-operation safety preservation bar chart** — breaks preservation rate down by Substitution/Explanation/Generalization rather than reporting one aggregate number, showing where each system's safety risk actually concentrates.
4. **Span reassembly** — after Sophakotra's operation router processes each span independently (Substitution via CHV, Explanation/Generalization via constrained LLM), spans need to be reassembled back into a coherent output sentence. Needs to handle boundary cases where adjacent spans were processed by different pathways.
5. **Finalize Introduction and Data sections** with full test-set numbers once Zihao's complete evaluation lands, replacing the current 50/30-sample preliminary figures.

### Team dependencies I'm tracking

- The readability-vs-safety scatter plot cannot be finalized until Zihao's full-corpus (148 sentences) evaluation across all six systems is complete.
- Span reassembly is a hard dependency for Sophakotra's operation router — the router decides what to do per span, but reassembly is what turns that into a deliverable sentence.

### Standing rule adopted from prelim
Before treating any EDA number or table as final, explicitly state the unit of analysis (abstract vs. sentence vs. span) in the same sentence as the number — this was the single most common source of correction needed during the prelim integration pass, appearing independently in three separate places before being caught.