# Operation-Aware Biomedical Text Simplification — Status Update

**Complexity-aware routing with safety preservation** | Team: Sruthilaya, Sophakotra, Zihao, Rishabh (+ Son)

---

### What's proven

- **Real UMLS jargon detection**: 100% abstract coverage (vs. 86.8% morphological baseline), validated on random samples
- **BioBERT beats TF-IDF by +9.7% macro-F1** (0.465 vs 0.424) for operation classification; general-domain DistilBERT *underperforms* TF-IDF (-7.5%) — biomedical pretraining matters, generic contextual pretraining doesn't
- **Domain features (UMLS + numerical) improve classification by +2.1–2.5% macro-F1** — verified via controlled ablation and model-coefficient inspection, not just an F1 delta
- **Random sampling hides rare safety failures**: a random 50-sentence sample showed perfect warning preservation across all baselines; a warning-*stratified* stress test caught a real 5% drop — methodological finding, not noise

### Two real bugs found & fixed this week

- An earlier "domain features don't help" result was caused by an **uncontrolled experiment** (mismatched TF-IDF vocab size between arms) — corrected, features do help
- The end-to-end generation pipeline was **silently running on an untrained classifier** (weights were never saved) — fixed; pipeline now fails loudly instead of guessing, checkpoint re-training in progress

### What's left before submission

- **The paper's central result — operation-aware system vs. 3 baselines — has not been run yet.** This is the top priority, in progress now.
- One final **test-set** run (all numbers so far are on val, correctly used for model selection)
- Readability-vs-safety scatter plot, human evaluation, final paper assembly

**Bottom line:** diagnostic components (detection, classification) are validated and outperforming baselines; the generation comparison that proves the paper's thesis is the one remaining experiment, on track before the deadline.
