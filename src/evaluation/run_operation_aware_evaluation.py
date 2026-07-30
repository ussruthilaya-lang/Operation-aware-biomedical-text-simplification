# Operation-aware system vs. the 3 baselines — the paper's headline comparison.
#
# Reuses the exact same val set, split logic, and metric suite as
# run_evaluation.py (same n=1,141 overall / n=73 warning-stratified sentences)
# so the new row is directly comparable to the existing baseline rows in
# results/final_evaluation.csv, not a different sample.
#
# Requires a trained BioBERT checkpoint at
# models/biobert_operation_classifier/ (see src/classifier/biobert_classifier.py)
# and the decoding-matched OperationAwarePipeline (see end_to_end_pipeline.py —
# generate() now uses the same num_beams=4, max_new_tokens=256 beam search as
# baseline3_direct_llm.py, so this comparison isolates the effect of
# classify-then-route rather than a confounded difference in decoding strategy).

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.evaluation.run_evaluation import load_full_val_set, evaluate_baseline
from src.complexity.warning_lexicon import flag_sentence as flag_warning
from src.pipeline.end_to_end_pipeline import OperationAwarePipeline

OUT_PATH = 'results/final_evaluation.csv'
SYSTEM_NAME = 'baseline4_operation_aware'  # matches baseline1/2/3 naming convention


def main():
    print("Loading full val set (same split as run_evaluation.py)...")
    sources, references_list = load_full_val_set()
    print(f"  total unique source sentences: {len(sources)}")

    strat_indices = [i for i, s in enumerate(sources) if flag_warning(s)]
    strat_sources = [sources[i] for i in strat_indices]
    strat_refs = [references_list[i] for i in strat_indices]
    print(f"Warning-stratified stress-test subset: "
          f"{len(strat_sources)} / {len(sources)} sentences")
    print()

    print("Loading OperationAwarePipeline (BioBERT + CHV + constrained FLAN-T5)...")
    pipeline = OperationAwarePipeline(load_llm=True)

    op_counts = {}

    def simplify_fn(sentence):
        result = pipeline.simplify(sentence)
        op_counts[result['operation']] = op_counts.get(result['operation'], 0) + 1
        return result['output']

    results = []

    print("\n=== OVERALL SPLIT (full val set) ===")
    row = evaluate_baseline(SYSTEM_NAME, simplify_fn, sources, references_list, 'overall')
    results.append(row)
    print(f"  {row}")
    print(f"  Operation distribution (overall): {op_counts}")

    op_counts_strat = dict(op_counts)  # snapshot before stratified pass overwrites it
    op_counts.clear()

    print("\n=== WARNING-STRATIFIED SPLIT (stress test) ===")
    row = evaluate_baseline(SYSTEM_NAME, simplify_fn, strat_sources, strat_refs, 'warning_stratified')
    results.append(row)
    print(f"  {row}")
    print(f"  Operation distribution (warning-stratified): {op_counts}")

    # Append to the existing baseline table rather than overwriting it —
    # results/final_evaluation.csv stays the single shared record for all
    # systems on this val split.
    existing = pd.read_csv(OUT_PATH)
    existing = existing[existing['baseline'] != SYSTEM_NAME]  # avoid dup rows on re-run
    combined = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} (added '{SYSTEM_NAME}' rows)")


if __name__ == "__main__":
    main()
