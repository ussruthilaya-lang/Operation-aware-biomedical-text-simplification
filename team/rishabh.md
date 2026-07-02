# Rishabh — Task Brief

**From Sruthilaya (Stage 1 complete):**

The repo is live and Stage 1 is done. Here's what I built that directly 
unblocks your work:

**What's ready for you:**
- `data/plaba/` — train.csv (635), val.csv (138), test.csv (148) + data.json
- `src/complexity/` — all 5 detectors running, importable
- `src/evaluation/warning_preservation.py` — safety metric ready
- `results/complexity_prelim.csv` — per-sentence complexity flags, 
   ready for your visualizations
- `src/baselines/baseline1_no_simplification.py` — safety ceiling done
- `src/baselines/baseline2_rule_based_chv.py` — CHV substitution done

**Key finding from my EDA you need to know:**
- 33.1% of sentences contain warning cues — this is our headline safety number
- 43.6% of sentences hit 3+ complexity types simultaneously
- These numbers go in the Introduction and Data sections you're writing

---

## Your Deliverables for Prelim (by Thursday EOD)

### 1. PLABA Data Pipeline — `src/data/plaba_loader.py`

Build a clean data loader the whole team imports. It should:

```python
load_train()  # returns DataFrame
load_val()    # returns DataFrame  
load_test()   # returns DataFrame
load_all()    # returns combined DataFrame with split column
```

Why this matters: right now everyone is writing 
`pd.read_csv('data/plaba/train.csv')` independently. 
One central loader means one place to fix encoding issues, 
one place to add preprocessing, one place to add the sample data path.
This is production-grade data engineering thinking.

### 2. CHV Lookup Module — `src/data/chv_lookup.py`

I have a starter CHV dictionary in `baseline2_rule_based_chv.py`.
Extract it into a standalone importable module:

```python
def lookup(term)        # returns plain English equivalent or None
def batch_lookup(terms) # returns dict of {term: equivalent}
def get_all_terms()     # returns full CHV dictionary
```

Why this matters: Sophakotra's operation routing needs CHV lookup 
for Substitution operations. If it lives in baseline2 she can't 
import it cleanly. Extract it, make it importable, add 50+ more 
terms from the real CHV vocabulary.
Real CHV: https://www.nlm.nih.gov/research/umls/sourcereleasedocs/current/CHV/

### 3. BERTScore Evaluation — `src/evaluation/bertscore_eval.py`

```python
def compute_bertscore(source, hypothesis, reference)
def compute_corpus_bertscore(pairs)  # list of (source, hyp, ref) tuples
```

Use `bert_score` library with BioBERT embeddings 
(`dmis-lab/biobert-base-cased-v1.2`) — domain-matched similarity 
matters here for the same reason our NER uses BC5CDR model.

### 4. Visualizations — `notebooks/visualizations.ipynb`

Four plots for the paper (load from `results/complexity_prelim.csv`):
- Complexity signal distribution bar chart (% sentences flagged per detector)
- Complexity co-occurrence heatmap (which detectors fire together)
- Sentence length distribution (source vs target)
- Warning preservation comparison (baseline 1 vs baseline 2)

Save all plots to `results/figures/` as PNG.

### 5. Paper Sections — `paper/prelim_report.tex`

Write Introduction and Data sections. Key points to hit:

**Introduction:**
- Health literacy gap — patients can't read biomedical text
- Current simplification systems treat all complexity the same
- Our diagnostic approach: classify first, simplify second
- Central research question (from proposal)

**Data section:**
- PLABA dataset description (750 abstracts, 4 references/sentence)
- Train/val/test split (635/138/148)
- Our EDA findings — use my numbers from complexity_prelim.csv
- Note: span-level operation labels derived via pseudo-labeling 
  (edit distance heuristics) since TREC annotations are behind login wall

### 6. Literature Review — 5-6 papers

Your angle: dataset construction, evaluation metrics, 
prior PLABA shared task systems.

Suggested papers:
1. Attal et al. 2023 — PLABA (you write the dataset section anyway)
2. Knappich et al. 2023 — BoschAI winner, motivates our approach
3. Li et al. 2024 — BeeManc system, your Lay-SciFive baseline source
4. Xu et al. 2016 — SARI metric (you're implementing BERTScore, 
   know why SARI exists)
5. Zhang et al. 2020 — BERTScore paper (you're implementing it)
6. One more on biomedical text readability or health literacy

---

## How Your Work Connects to the Full Paper

```
Your data loader → everyone imports it → no broken paths
Your CHV module → Sophakotra's operation router uses it
Your BERTScore → Zihao's results table uses it
Your visualizations → go directly into the paper figures
Your Introduction → frames the entire paper narrative
```

You are the team's data infrastructure. If your loader breaks, 
everyone's code breaks. Ship it first, ship it clean.

---

## Repo Instructions

```bash
git clone https://github.com/ussruthilaya-lang/Operation-aware-biomedical-text-simplification.git
cd Operation-aware-biomedical-text-simplification
conda create -n nlp-project python=3.10 -y
conda activate nlp-project
conda install -c conda-forge spacy pandas -y
pip install scispacy bert-score
```

Data files go in `data/plaba/` — download from https://osf.io/rnpmf/
(gitignored — don't commit the CSVs or data.json)

Create your branch:
```bash
git checkout -b feature/rishabh-data-pipeline
```

Push your work and open a PR to main by Thursday EOD.