# Zihao — Task Brief

**Repo:** https://github.com/ussruthilaya-lang/Operation-aware-biomedical-text-simplification  
**Your branch:** `feature/zihao-biobert-evaluation`  
**Deadline:** Thursday EOD

---

## What Sruthilaya Already Built (Stage 1 complete)

- `src/evaluation/warning_preservation.py` — safety metric ready
- `src/baselines/baseline1_no_simplification.py` — ceiling (1.000)
- `src/baselines/baseline2_rule_based_chv.py` — rule-based (0.994)
- `results/complexity_prelim.csv` — per-sentence complexity flags

**The evaluation story you're telling:**
SARI measures readability. Safety metrics measure what SARI misses.
The gap between them is our contribution.
Expected finding: Lay-SciFive scores highest SARI, lowest safety preservation.
If true, we've proven SARI alone is insufficient for healthcare NLP.

---

## PRELIM — Due Thursday EOD

### 1. Core Metrics — `src/evaluation/metrics.py`

```python
def compute_sari(source, hypothesis, references)
# pip install easse
# Primary metric — measures keep/add/delete vs 4 references

def compute_fkgl(text)
# Flesch-Kincaid Grade Level
# Target: below grade 8 for patient-facing text

def compute_compression_ratio(source, hypothesis)
# len(hypothesis.split()) / len(source.split())

def compute_all_metrics(source, hypothesis, references)
# Returns dict — all metrics in one call
```

### 2. Prelim Evaluation Run — `src/evaluation/run_evaluation.py`

Run on **50-sentence sample** from val set for prelim.
Even just baseline1 and baseline2 is enough — show the framework works.

```python
# Output: results/prelim_evaluation.csv
# Columns: baseline | SARI | FKGL | compression_ratio | warning_preservation
```

### 3. Paper Sections — `paper/prelim_report.tex`

**Results section** (~0.4 page):
- Prelim results table (50-sentence sample)
- Baseline 1 vs Baseline 2 comparison
- What full evaluation will add

**Conclusion** (~0.2 page):
- Restate diagnostic approach
- What prelim results suggest
- What final evaluation will confirm

### 4. Literature Review — 5-6 papers

Your angle: neural simplification models, evaluation, biomedical LMs.

Suggested:
1. Knappich et al. 2023 — BoschAI PLABA winner (your SARI target)
2. UoL-UPF 2nd Place — MBR decoding, metric criticism
3. Lewis et al. 2020 — BART
4. Readability Metrics Paper — LLM-as-judge vs traditional formulas
5. Difficulty Critics Paper — CEFR control limitations
6. Cochrane Multi-Agent — diagnostic framing comparison

---

## FULL PAPER — Not Thursday, Final Submission

- Lay-SciFive inference `src/models/layscifive_inference.py`
- BART-w-CTs inference `src/models/bart_inference.py`
- BioBERT fine-tuning `src/models/biobert_classifier.py`
  (use Google Colab free T4 GPU if no local GPU)
- Entity preservation metric `src/evaluation/entity_preservation.py`
- Numerical preservation metric `src/evaluation/numerical_preservation.py`
- Full evaluation table across all 6 systems on complete test set

---

## Repo Setup

```bash
git clone https://github.com/ussruthilaya-lang/Operation-aware-biomedical-text-simplification.git
cd Operation-aware-biomedical-text-simplification
conda create -n nlp-project python=3.10 -y
conda activate nlp-project
conda install -c conda-forge spacy pandas -y
pip install transformers torch easse sentence-transformers
```

Data: download from https://osf.io/rnpmf/ → put in `data/plaba/`

For BioBERT without local GPU → Google Colab, save checkpoint to Drive.

```bash
git checkout -b feature/zihao-biobert-evaluation
```

PR to main by Thursday EOD.