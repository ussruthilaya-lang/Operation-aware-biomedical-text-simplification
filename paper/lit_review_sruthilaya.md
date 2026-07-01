# Literature Review — Sruthilaya

## Papers to cover
1. Attal et al. (2023) — PLABA dataset
2. HOPE Control Paper — rule-based baseline beats zero-shot
3. Medical Jargon Paper — RoBERTa jargon detection on PLABA
4. TSAR Shared Task Findings (2025)
5. Health QA SPQA — style perturbations reduce correctness
6. Neumann et al. (2019) — SciSpaCy

## Full entries — to fill with PDFs

## 1. Attal et al. (2023) — PLABA Dataset
**Citation:** Attal, K., Ondov, B., & Demner-Fushman, D. (2023). A dataset for 
plain language adaptation of biomedical abstracts. Scientific Data, 10(1), 8.

**What it is:** Introduces PLABA — 750 PubMed abstracts paired with 
professionally written plain-language adaptations, sentence-aligned with 
4 references per sentence for SARI evaluation.

**Relation to our work:** This is our primary dataset. The sentence-level 
alignment structure directly enables our span-level complexity detection. 
Unlike prior work that uses single references, PLABA's 4-reference design 
reduces evaluation variance — critical for our safety metric comparisons.

**Different from us:** Attal et al. focus on dataset construction and 
benchmark baselines. We treat the simplification as a diagnostic task — 
classifying complexity type before generating output.

---

## 2. Heineman et al. (2023) — SALSA Edit Taxonomy
**Citation:** Heineman, B., et al. (2023). SALSA: Semantic Aware 
Lexical Simplification Access. arXiv:2305.14458.

**What it is:** 

**Relation to our work:** 

**Different from us:** 

---

## 3. Neumann et al. (2019) — SciSpaCy
**Citation:** Neumann, M., King, D., Beltagy, I., & Ammar, W. (2019). 
SciSpaCy: Fast and Robust Models for Biomedical NLP. arXiv:1902.07669.

**What it is:** 

**Relation to our work:** 

**Different from us:** 

---

## 4. Soldaini & Goharian (2016) — QuickUMLS
**Citation:** Soldaini, L., & Goharian, N. (2016). QuickUMLS: a fast, 
unsupervised approach for medical concept extraction. MedIR Workshop.

**What it is:** 

**Relation to our work:** 

**Different from us:** 

---

## 5. [Readability in clinical text]
**Citation:** 

**What it is:** 

**Relation to our work:** 

**Different from us:** 

---

## 6. [Safety / information loss in medical NLP]
**Citation:** 

**What it is:** 

**Relation to our work:** 

**Different from us:**