# Sophakotra — Task Brief

**Repo:** https://github.com/ussruthilaya-lang/Operation-aware-biomedical-text-simplification  
**Your branch:** `feature/sophakotra-classifier`  
**Deadline:** Thursday EOD

---

## What Sruthilaya Already Built (Stage 1 complete)

- All 5 complexity detectors in `src/complexity/` — importable, tested
- `results/complexity_prelim.csv` — per-sentence complexity flags
- `src/baselines/baseline2_rule_based_chv.py` — rule-based baseline
- `src/evaluation/warning_preservation.py` — safety metric ready

**Key architectural decision:**
My detectors flag THAT complexity exists.
Your classifier predicts WHAT OPERATION it needs.
That's the research contribution that separates us from every other PLABA system.

---

## PRELIM — Due Thursday EOD

### 1. Pseudo-Label Pipeline — `src/data/pseudo_labeler.py`

Span-level labels aren't publicly available — we derive them:

```python
def label_span(source_span, target_span):
    """
    Substitution: similar length, high character edit distance
    Explanation: target significantly longer than source
    Generalization: target significantly shorter than source
    """

def label_sentence_pair(source, target)
def label_dataset(df)  # runs on full DataFrame
```

Why edit distance: it's the most direct measurable signal for 
what TYPE of change was made. Standard weakly supervised NLP practice.

### 2. TF-IDF + Logistic Regression — `src/classifier/tfidf_classifier.py`

```python
# ngram_range=(1,2), max_features=50000
# class_weight='balanced', solver='lbfgs'

def train(X_train, y_train)
def predict(X_test)
def evaluate(X_test, y_test)
# returns: accuracy, macro-F1, per-class F1, confusion matrix
```

For prelim: train on pseudo-labeled spans, evaluate on val set.
Report numbers in your paper section.

### 3. Direct LLM Baseline — `src/baselines/baseline3_direct_llm.py`

```python
# Prompt: "Simplify this biomedical sentence for a patient: {text}"
# No operation guidance — raw prompting only
# Run on 20-30 sentence sample for prelim

def simplify(text, model="gpt-3.5-turbo")
def run_on_sample(df, n=30)
```

### 4. Paper Sections — `paper/prelim_report.tex`

**Classifier section** (~0.4 page):
- Three-model progression rationale: TF-IDF → BERT → BioBERT
- What each comparison isolates
- Pseudo-labeling methodology and justification
- Prelim TF-IDF results (accuracy, macro-F1, confusion matrix)

**Baselines section** (~0.3 page):
- Baseline 1: no simplification — from Sruthilaya (1.0 warning preservation)
- Baseline 2: rule-based CHV — from Sruthilaya (0.994)
- Baseline 3: direct LLM — yours
- Why these three form a meaningful comparison ladder

### 5. Literature Review — 5-6 papers

Your angle: operation taxonomies, text classification, constrained generation.

Suggested:
1. Heineman et al. 2023 — SALSA edit taxonomy
   (directly defines your 3 operation types)
2. TSAR Shared Task Findings — LLM dominance + safety gaps
3. EhiMeNLP Winner — best LLM ensemble
4. Devlin et al. 2019 — BERT
5. Lee et al. 2020 — BioBERT
6. Health QA SPQA — style perturbations drop correctness

---

## FULL PAPER — Not Thursday, Final Submission

- BERT-base fine-tuned classifier
- Operation routing logic `src/pipeline/operation_router.py`
- Operation-aware constrained prompts `src/pipeline/constrained_prompts.py`
- OpenAI integration for full pipeline
- Complete classifier comparison: TF-IDF vs BERT vs BioBERT

---

## Repo Setup

```bash
git clone https://github.com/ussruthilaya-lang/Operation-aware-biomedical-text-simplification.git
cd Operation-aware-biomedical-text-simplification
conda create -n nlp-project python=3.10 -y
conda activate nlp-project
conda install -c conda-forge spacy pandas scikit-learn -y
pip install openai
```

Data: download from https://osf.io/rnpmf/ → put in `data/plaba/`

```bash
git checkout -b feature/sophakotra-classifier
```

PR to main by Thursday EOD.