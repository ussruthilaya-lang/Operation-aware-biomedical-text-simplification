# Sophakotra — Task Brief

**From Sruthilaya (Stage 1 complete):**

Stage 1 is fully built and ready for you to build on top of.
Here's exactly what exists and how your classifier connects to it.

**What's ready for you:**
- All 5 complexity detectors in `src/complexity/` — importable, tested
- `results/complexity_prelim.csv` — per-sentence complexity flags 
  for all 635 training sentences
- `src/baselines/baseline2_rule_based_chv.py` — CHV substitution 
  baseline already implemented (your direct LLM baseline competes with this)
- `src/evaluation/warning_preservation.py` — safety metric ready 
  to evaluate your system outputs

**Key architectural decision you need to know:**
My detectors output boolean flags per sentence. Your classifier 
takes it one level deeper — it predicts the OPERATION TYPE 
per flagged span, not just whether complexity exists.
That's the research contribution that separates us from every 
other PLABA system.

---

## Your Deliverables for Prelim (by Thursday EOD)

### 1. Pseudo-Label Pipeline — `src/data/pseudo_labeler.py`

Span-level operation labels are not publicly available 
(TREC participant login required). We derive them from 
parallel text using edit distance heuristics.

```python
def label_span(source_span, target_span):
    """
    Returns operation label based on edit distance ratio:
    - Substitution: similar length, high edit distance (word swap)
    - Explanation: target significantly longer than source (added context)
    - Generalization: target significantly shorter than source (compressed)
    """

def label_sentence_pair(source, target):
    """Aligns source and target spans, labels each."""

def label_dataset(df):
    """Runs pseudo-labeling on full DataFrame, returns labeled spans."""
```

Why edit distance? It's the simplest measurable signal for 
what TYPE of change was made. A word swap has high edit distance 
but similar length. An explanation adds words. A generalization 
removes them. This is weakly supervised NLP — standard practice 
when gold labels aren't available.

### 2. TF-IDF + Logistic Regression Classifier — `src/classifier/tfidf_classifier.py`

```python
# Key hyperparameters from proposal:
# ngram_range=(1,2), max_features=50000
# class_weight='balanced', solver='lbfgs'

def train(X_train, y_train)
def predict(X_test)
def evaluate(X_test, y_test)  # returns accuracy, macro-F1, per-class F1
```

Why TF-IDF+LR first? It's your classical NLP baseline — 
interpretable, GPU-free, fast to train. It tells you what 
n-gram patterns predict operation type without any contextual 
understanding. When BioBERT outperforms this, you know exactly 
what contextual embeddings add. That comparison IS a finding.

### 3. Operation Routing Logic — `src/pipeline/operation_router.py`

```python
def route(span, predicted_operation):
    """
    Substitution → CHV lookup (src/data/chv_lookup.py from Rishabh)
    Explanation  → LLM with explanation prompt
    Generalization → LLM with generalization prompt
    """
```

### 4. Direct LLM Baseline — `src/baselines/baseline3_direct_llm.py`

```python
# Prompt: "Simplify this biomedical sentence for a patient: {text}"
# No operation guidance — just raw prompting
# This is what we beat on safety metrics
def simplify(text, model="gpt-3.5-turbo")
def run_on_dataset(df)
```

### 5. Operation-Aware Prompting — `src/pipeline/constrained_prompts.py`

```python
EXPLANATION_PROMPT = """
You are simplifying biomedical text for patients.
The following span requires EXPLANATION — define the term 
and add context. Do not substitute with a simpler word.
Span: {span}
Output only the explanation, nothing else.
"""

GENERALIZATION_PROMPT = """
You are simplifying biomedical text for patients.
The following span requires GENERALIZATION — compress and 
simplify without losing the core meaning.
Span: {span}
Output only the generalized version, nothing else.
"""
```

Why constrained prompts matter: unconstrained LLM decides what 
to do. Constrained LLM executes what the classifier already decided. 
The classifier is in control, the LLM is the execution engine.
This is the architectural decision that makes our system auditable.
In regulated healthcare AI — FDA, HIPAA, clinical decision support — 
auditability is not optional. This framing matters for your paper 
and for enterprise AI careers.

### 6. Paper Sections — `paper/prelim_report.tex`

Write Classifier and Baselines sections. Key points:

**Classifier section:**
- Three-model progression rationale (TF-IDF → BERT → BioBERT)
- What each comparison isolates (contextual embeddings, 
  biomedical pretraining)
- Pseudo-labeling methodology and justification
- Prelim: TF-IDF results on val set (accuracy, macro-F1, 
  confusion matrix)

**Baselines section:**
- Baseline 1: no simplification (safety ceiling) — from Sruthilaya
- Baseline 2: rule-based CHV — from Sruthilaya
- Baseline 3: direct LLM prompt — yours
- Why these three form a meaningful comparison ladder

### 7. Literature Review — 5-6 papers

Your angle: text classification, operation taxonomies, 
constrained generation, LLM prompting for simplification.

Suggested papers:
1. Heineman et al. 2023 — SALSA edit taxonomy 
   (directly defines your 3 operation types)
2. TSAR Shared Task Findings — LLM dominance + safety gaps
3. EhiMeNLP Winner — best LLM ensemble, what you're competing with
4. Health QA SPQA — style perturbations drop correctness
5. Devlin et al. 2019 — BERT (classifier foundation)
6. Lee et al. 2020 — BioBERT (domain-specific classifier)

---

## How Your Work Connects to the Full Paper

```
Pseudo-labeler → trains your classifier → enables operation routing
TF-IDF classifier → prelim results → paper's classifier section
Direct LLM baseline → comparison point → proves operation-aware is better
Constrained prompts → Stage 3 generation → Zihao evaluates outputs
Operation router → connects Stage 1 (Sruthilaya) to Stage 3 (Zihao)
```

You are the research centerpiece. The classifier is what makes 
this project original. Everything else supports or evaluates it.

---

## Repo Instructions

```bash
git clone https://github.com/ussruthilaya-lang/Operation-aware-biomedical-text-simplification.git
cd Operation-aware-biomedical-text-simplification
conda create -n nlp-project python=3.10 -y
conda activate nlp-project
conda install -c conda-forge spacy pandas scikit-learn -y
pip install openai
```

Data files go in `data/plaba/` — download from https://osf.io/rnpmf/

Create your branch:
```bash
git checkout -b feature/sophakotra-classifier
```

Push your work and open a PR to main by Thursday EOD.