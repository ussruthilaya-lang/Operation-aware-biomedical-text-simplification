# Zihao — Task Brief

**From Sruthilaya (Stage 1 complete):**

The pipeline foundation is fully built. Your work is the final stage —
models, inference, and evaluation. Here's exactly what exists and 
how your work connects.

**What's ready for you:**
- `src/evaluation/warning_preservation.py` — safety metric ready
- `src/baselines/baseline1_no_simplification.py` — safety ceiling (1.0)
- `src/baselines/baseline2_rule_based_chv.py` — rule-based baseline (0.994)
- `results/complexity_prelim.csv` — per-sentence complexity flags
- Sophakotra's direct LLM baseline (baseline3) — your fourth comparison point

**The evaluation story you're telling:**
Every metric you implement answers one question:
does operation-aware simplification navigate the 
readability-safety tradeoff better than all baselines?
SARI measures readability. Your safety metrics measure 
what SARI misses. That gap is our contribution.

---

## Your Deliverables for Prelim (by Thursday EOD)

### 1. Lay-SciFive Inference — `src/models/layscifive_inference.py`

```python
# Load from BeeManc GitHub checkpoint
# https://github.com/HECTA-UoM/BeeManc
# Model: razent/SciFive-large-Pubmed_PMC fine-tuned on PLABA

def load_model()
def simplify(text)
def run_on_dataset(df, text_column='input_text')
```

Why Lay-SciFive matters: it's the best publicly available 
trained model for biomedical simplification. It will score 
highest on SARI — that's expected. But our hypothesis is it 
scores lowest on safety preservation. That contrast IS the paper.
If we're right, we've proven that SARI alone is insufficient 
for healthcare NLP evaluation.

### 2. BART-w-CTs Inference — `src/models/bart_inference.py`

```python
# Second BeeManc model — BART fine-tuned with clinical terms
def load_model()
def simplify(text)
def run_on_dataset(df, text_column='input_text')
```

### 3. Evaluation Metrics — `src/evaluation/metrics.py`

```python
def compute_sari(source, hypothesis, references)
# Use: pip install easse
# Why SARI: measures keep, add, delete operations separately
# against 4 references — only metric designed for simplification

def compute_fkgl(text)
# Flesch-Kincaid Grade Level — readability
# Lower = simpler. Target: below grade 8 for patient-facing text

def compute_sbert_similarity(source, hypothesis)
# Sentence-BERT cosine similarity — semantic preservation
# Use: sentence-transformers library
# Why SBERT over BERTScore: faster, better for sentence-level similarity

def compute_compression_ratio(source, hypothesis)
# len(hypothesis) / len(source) — how much was compressed

def compute_all_metrics(source, hypothesis, references)
# Returns dict of all metrics in one call
```

### 4. Entity Preservation Rate — `src/evaluation/entity_preservation.py`

```python
# Uses Sruthilaya's NER detector
from src.complexity.ner_detector import get_entity_spans

def compute_entity_preservation_rate(source, simplified):
    """
    What fraction of biomedical entities in source 
    appear in simplified output?
    
    Score 1.0 = all entities preserved (safe)
    Score 0.0 = all entities dropped (dangerous)
    """

def compute_corpus_entity_preservation(pairs)
```

Why this matters: entity preservation catches a different 
failure mode than warning preservation. A system can preserve 
"avoid taking" (warning) but drop "sorafenib" (entity) — 
the patient loses the drug name they need to tell their doctor.
Two separate metrics, two separate safety failure modes.

### 5. Numerical Preservation Rate — `src/evaluation/numerical_preservation.py`

```python
# Uses Sruthilaya's numerical extractor
from src.complexity.numerical_extractor import extract_numerical_expressions

def compute_numerical_preservation_rate(source, simplified):
    """
    What fraction of numerical expressions in source 
    appear in simplified output?
    Critical: dropped dosages and risk ratios are 
    safety failures, not just quality issues.
    """

def compute_corpus_numerical_preservation(pairs)
```

### 6. Results Table — `src/evaluation/run_evaluation.py`

Runs ALL metrics across ALL baselines and saves to 
`results/full_evaluation.csv`:

```python
# For each baseline/model:
# SARI | FKGL | SBERT | Compression | 
# Warning Pres. | Entity Pres. | Numerical Pres.

baselines = {
    'no_simplification': baseline1,
    'rule_based_chv': baseline2,
    'direct_llm': baseline3,      # from Sophakotra
    'lay_scifive': layscifive,    # yours
    'bart_w_cts': bart,           # yours
    'operation_aware': full_pipeline  # team's full system
}
```

For prelim: run on 50-sentence sample from val set.
Full eval on complete test set for final paper.

### 7. BioBERT Fine-tuning — `src/models/biobert_classifier.py`

```python
# dmis-lab/biobert-base-cased-v1.2
# Fine-tune on pseudo-labeled spans from Sophakotra
# learning_rate=2e-5, batch_size=16, epochs=3-5
# AdamW optimizer, class-weighted loss

def train(train_dataset, val_dataset)
def predict(text)
def evaluate(test_dataset)  # accuracy, macro-F1, confusion matrix
```

For prelim: if GPU not available locally, use Google Colab 
(free T4 GPU). Save checkpoint to `models/biobert_finetuned/`.
This is gitignored — checkpoint goes to GCP when credits arrive.

### 8. Paper Sections — `paper/prelim_report.tex`

Write Generation and Results sections plus Conclusion.

**Generation section:**
- How operation routing connects to generation
- Lay-SciFive and BART-w-CTs as trained model baselines
- Why loaded from checkpoint not prompted

**Results section:**
- Prelim results table (50-sentence sample)
- Readability vs safety tradeoff framing
- Expected finding: Lay-SciFive highest SARI, 
  lowest safety preservation

**Conclusion:**
- Restate the diagnostic approach
- What prelim results suggest
- What final evaluation will confirm

### 9. Literature Review — 5-6 papers

Your angle: neural text simplification, evaluation metrics, 
biomedical language models.

Suggested papers:
1. Knappich et al. 2023 — BoschAI PLABA winner
   (your SARI target to beat)
2. UoL-UPF 2nd Place — MBR decoding, strong metric criticism
3. Cochrane Multi-Agent — multi-agent diagnostic framing
4. Readability Metrics Paper — LLM-as-judge vs traditional formulas
5. Lewis et al. 2020 — BART (you're running BART-w-CTs)
6. Difficulty Critics Paper — CEFR control vs safety preservation

---

## How Your Work Connects to the Full Paper

```
Lay-SciFive + BART → trained model baselines → results table
Entity + numerical preservation → complete safety picture
  alongside Sruthilaya's warning preservation metric
BioBERT classifier → operation-aware pipeline's brain
Results table → paper's central empirical contribution
Conclusion → ties the diagnostic framing back to findings
```

You are the evidence layer. The paper's central claim —
operation-aware simplification navigates the 
readability-safety tradeoff better than all baselines —
is proven or disproven by your evaluation.

---

## Repo Instructions

```bash
git clone https://github.com/ussruthilaya-lang/Operation-aware-biomedical-text-simplification.git
cd Operation-aware-biomedical-text-simplification
conda create -n nlp-project python=3.10 -y
conda activate nlp-project
conda install -c conda-forge spacy pandas -y
pip install transformers torch easse sentence-transformers bert-score
```

Data files go in `data/plaba/` — download from https://osf.io/rnpmf/

For BioBERT fine-tuning without local GPU:
Use Google Colab — mount Drive, save checkpoint there.

Create your branch:
```bash
git checkout -b feature/zihao-biobert-evaluation
```

Push your work and open a PR to main by Thursday EOD.