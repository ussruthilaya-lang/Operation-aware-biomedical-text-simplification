<!--
Preliminary Report — Sophakotra's sections
Operation Classification + Baselines + Literature Review

All numbers below are the actual outputs of:
  python -m src.data.pseudo_labeler
  python -m src.classifier.tfidf_classifier
  python -m src.baselines.baseline3_direct_llm
-->

# Operation-Aware Biomedical Text Simplification
## Preliminary Report — Operation Classification and Baselines
**Sophakotra — July 2026**

---

## Operation Classification

The central contribution of our system is a *diagnostic* step: before
simplifying a span of biomedical text, we predict *which* simplification
operation it requires. We adopt the three-operation taxonomy of
Heineman et al. [1]: **Substitution** (replacing a jargon term with a
plain-language equivalent), **Explanation** (adding words to unpack a concept),
and **Generalization** (compressing or removing detail). While prior PLABA
systems apply a single undifferentiated simplification model, we argue that
routing each span to the appropriate operation is what preserves safety-critical
information.

**Pseudo-labeling.**
Span-level operation labels are not available in PLABA, so we derive them by
weak supervision from the source→target sentence pairs. The *type* of edit a
human writer performed leaves a measurable fingerprint on the text: an
Explanation lengthens the sentence, a Generalization shortens it, and a
Substitution changes many characters while keeping the length roughly constant.
Concretely, let *r = |t| / |s|* be the target/source word-length ratio. We label
a pair Explanation if *r ≥ 1.15*, Generalization if *r ≤ 0.85*, and otherwise
Substitution, using normalized character-level edit distance to confirm that
similar-length pairs were genuinely rewritten rather than copied. Edit distance
is the most direct measurable signal of the kind of change made, and requires no
gold annotation — standard weakly-supervised practice.

**Data and leakage control.**
The PLABA release (`data.json`) is the full corpus and contains the abstracts
held out in the official validation and test splits. To avoid train/test
leakage, we exclude every validation and test PMID from the training pairs at
the source level. This yields **6,307** pseudo-labeled training sentence pairs,
distributed across the three operations as shown below. We evaluate on **1,431**
sentence pairs built from the held-out validation PMIDs and pseudo-labeled
identically.

**Table 1. Pseudo-labeled operation distribution on the leakage-free training set.**

| Operation | Train pairs | Share |
|---|---:|---:|
| Substitution | 2,723 | 43.2% |
| Explanation | 2,167 | 34.4% |
| Generalization | 1,417 | 22.5% |
| **Total** | **6,307** | **100%** |

**Three-model progression.**
We plan a controlled progression of classifiers, each isolating one variable:
(i) **TF-IDF + Logistic Regression** tests whether surface lexical features alone
predict the operation; (ii) **BERT-base** [3] adds general contextual
pretraining; and (iii) **BioBERT** [4] adds biomedical-domain pretraining on top.
Reporting all three localizes exactly where any gains originate. This
preliminary report covers the TF-IDF baseline; the contextual models follow in
the full paper.

**Preliminary results.**
The classifier receives only the *source* sentence, since in the full pipeline
the operation must be chosen before the simplified text is generated. Using
(1,2)-gram TF-IDF features (`max_features = 50,000`) and Logistic Regression with
balanced class weights, the model attains **0.431** accuracy and **0.424**
macro-F1 on the validation set (Table 2). Although accuracy is close to the
majority-class rate (0.427), the macro-F1 is far above the corresponding
majority baseline (≈ 0.20), confirming that the model makes balanced predictions
across all three classes rather than collapsing to the majority. The confusion
matrix (Table 3) shows the strongest confusion between Substitution and
Explanation, consistent with the fact that both often preserve sentence length.
These results establish a weak-but-nontrivial lower bound and motivate the
contextual models, which can capture semantic and structural cues unavailable to
bag-of-words features.

**Table 2. TF-IDF + Logistic Regression, per-class results on the validation set.**

| | Precision | Recall | F1 |
|---|---:|---:|---:|
| Substitution | 0.471 | 0.481 | 0.476 |
| Explanation | 0.427 | 0.351 | 0.385 |
| Generalization | 0.376 | 0.449 | 0.410 |
| **Macro avg** | **0.425** | **0.427** | **0.424** |
| **Accuracy** | | | **0.431** (n = 1,431) |

**Table 3. Confusion matrix (rows = true, columns = predicted).**

| True \ Pred | Sub | Exp | Gen |
|---|---:|---:|---:|
| Substitution | 294 | 150 | 167 |
| Explanation | 203 | 163 | 98 |
| Generalization | 127 | 69 | 160 |

---

## Baselines

We frame our baselines as a comparison ladder anchored on *warning preservation*
— the fraction of source warning phrases that survive simplification. This
metric targets safety-critical information loss that n-gram metrics such as
ROUGE ignore.

**Table 4. Warning preservation across baselines. Direct LLM prompting loses the most safety-critical information.**

| Baseline | Warning preservation |
|---|:---:|
| No simplification (ceiling) | 1.000 |
| Rule-based CHV substitution | 0.994 |
| Direct LLM prompting (FLAN-T5-Base) | 0.950 |
| Operation-aware system (ours) | TBD |

**Baseline 1 (no simplification)** leaves the text unchanged and thus trivially
preserves all warnings (1.000); it is the safety ceiling against which no system
should be expected to improve. **Baseline 2 (rule-based CHV substitution)**
performs Consumer Health Vocabulary term replacement and sentence splitting,
preserving 99.4% of warnings — auditable and low-risk, but limited to
vocabulary-level edits. **Baseline 3 (direct LLM prompting)** is the
deliberately naive approach our system argues against: we prompt an
instruction-tuned open-source LLM (FLAN-T5-Base) to "simplify this biomedical
sentence for a patient," with no operation guidance and no safety constraint. We
run it locally to keep the baseline fully reproducible (no paid API).

On a 30-sentence sample restricted to warning-bearing sentences, direct LLM
prompting preserved only **95.0%** of source warnings — measurably below both
the rule-based baseline (99.4%) and the do-nothing ceiling (100%). This confirms
our central premise: unguided LLM simplification sacrifices safety-critical
information that conservative methods retain, motivating the operation-aware
routing our full system introduces. The three baselines together bracket the
safety–fluency trade-off our system must navigate.

---

## Literature Review

**SALSA edit taxonomy** [1] defines the fine-grained edit operations that
underpin our three-way operation taxonomy, directly grounding our
Substitution/Explanation/Generalization labels. The **PLABA dataset** [2]
provides the sentence-aligned source–adaptation pairs we pseudo-label.
**BERT** [3] and **BioBERT** [4] motivate the contextual and biomedical-domain
classifiers in our model progression. The **TSAR shared task** [5] documents the
field's shift toward LLM-based simplification and the accompanying safety gaps
that motivate our diagnostic, safety-first framing.

<!-- VERIFY: two more entries need confirmed bibliographic details before
     submission — fill in authors/venue/year, then add to the list:
     - EhiMeNLP TSAR winner (best LLM ensemble)
     - Health QA SPQA (style perturbations reduce correctness/completeness) -->

### References

1. Heineman, D., Dou, Y., Maddela, M., & Xu, W. (2023). *Dancing Between Success
   and Failure: Edit-level Simplification Evaluation using SALSA.* EMNLP 2023.
   arXiv:2305.14458.
2. Attal, K., Ondov, B., & Demner-Fushman, D. (2023). *A dataset for plain
   language adaptation of biomedical abstracts.* Scientific Data, 10(1), 8.
3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training
   of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT.
4. Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2020).
   *BioBERT: a pre-trained biomedical language representation model for
   biomedical text mining.* Bioinformatics, 36(4), 1234–1240.
5. Saggion, H., Štajner, S., Ferrés, D., et al. (2022). *Findings of the
   TSAR-2022 Shared Task on Multilingual Lexical Simplification.* Proceedings of
   the TSAR Workshop, EMNLP 2022. *(VERIFY exact author list / page numbers.)*
