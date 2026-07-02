# Rishabh — Task Brief

**Repo:** https://github.com/ussruthilaya-lang/Operation-aware-biomedical-text-simplification  
**Your branch:** `feature/rishabh-data-pipeline`  
**Deadline:** Thursday EOD

---

## What Sruthilaya Already Built (Stage 1 complete)

- `data/plaba/` — train.csv (635), val.csv (138), test.csv (148) + data.json
- `src/complexity/` — all 5 detectors, importable
- `src/evaluation/warning_preservation.py` — safety metric
- `src/baselines/baseline1_no_simplification.py` — safety ceiling (1.0)
- `src/baselines/baseline2_rule_based_chv.py` — CHV substitution (0.994)
- `results/complexity_prelim.csv` — per-sentence complexity flags

**Key EDA numbers for your Introduction section:**
- 87.9% sentences contain medical jargon
- 82.7% are syntactically complex
- 60.9% carry numerical risk data
- 33.1% contain safety-critical warning language
- 43.6% exhibit 3+ simultaneous complexity types

---

## PRELIM — Due Thursday EOD

### 1. PLABA Data Loader — `src/data/plaba_loader.py`

```python
def load_train()   # returns DataFrame
def load_val()     # returns DataFrame
def load_test()    # returns DataFrame
def load_all()     # combined with split column
```

Why: everyone currently writes `pd.read_csv('data/plaba/train.csv')` 
independently. One central loader = one place to fix encoding issues, 
one place to add preprocessing. Production-grade data engineering.

### 2. CHV Lookup Module — `src/data/chv_lookup.py`

Extract the CHV dictionary from `src/baselines/baseline2_rule_based_chv.py`
into a standalone importable module:

```python
def lookup(term)         # returns plain English or None
def batch_lookup(terms)  # returns {term: equivalent}
def get_all_terms()      # returns full dictionary
```

Add 20-30 more terms from: https://www.nlm.nih.gov/research/umls/sourcereleasedocs/current/CHV/
Sophakotra's operation router needs this — don't leave it buried in baseline2.

### 3. Visualizations — `notebooks/visualizations.ipynb`

Three plots minimum, saved to `results/figures/` as PNG:
- Complexity signal bar chart (% sentences per detector)
- Complexity co-occurrence counts (0/1/2/3/4 detectors firing)
- Sentence length distribution (source vs target word count)

Load from `results/complexity_prelim.csv` — data is already there.

### 4. Paper Sections — `paper/prelim_report.tex`

**Introduction** (~0.5 page):
- Health literacy gap — patients can't read biomedical text
- Current systems treat all complexity the same
- Our diagnostic approach: classify first, simplify second
- Central research question

**Data section** (~0.3 page):
- PLABA description (750 abstracts, 4 references/sentence)
- Train/val/test split (635/138/148)
- Use EDA numbers above
- Note: span-level labels derived via pseudo-labeling

### 5. Literature Review — 5-6 papers

Your angle: datasets, evaluation metrics, prior PLABA systems.

Suggested:
1. Attal et al. 2023 — PLABA dataset
2. Knappich et al. 2023 — BoschAI PLABA winner
3. Li et al. 2024 — BeeManc system
4. Xu et al. 2016 — SARI metric
5. Zhang et al. 2020 — BERTScore
6. One on health literacy or readability

---

## FULL PAPER — Not Thursday, Final Submission

- BERTScore evaluation module `src/evaluation/bertscore_eval.py`
- Span reassembly for pipeline output
- Additional visualizations: readability vs safety scatter plot,
  per-operation safety preservation bar chart
- Finalize Introduction and Data sections with full results

---

## Repo Setup

```bash
git clone https://github.com/ussruthilaya-lang/Operation-aware-biomedical-text-simplification.git
cd Operation-aware-biomedical-text-simplification
conda create -n nlp-project python=3.10 -y
conda activate nlp-project
conda install -c conda-forge spacy pandas -y
pip install scispacy
```

Data: download from https://osf.io/rnpmf/ → put in `data/plaba/`

```bash
git checkout -b feature/rishabh-data-pipeline
```

PR to main by Thursday EOD.