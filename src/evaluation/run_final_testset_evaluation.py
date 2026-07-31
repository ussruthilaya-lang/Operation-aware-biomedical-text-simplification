# Final test-set evaluation — all 4 systems, frozen configuration.
#
# Everything reported so far this session (Track 1, Track 2, Zihao's
# 3-baseline harness, the operation-aware pipeline result) was measured on
# the VAL split — correct for model selection during development, but not
# valid as the paper's headline numbers, since val was used to pick the
# final configuration. This script re-runs the exact same metric suite on
# the true 148-row / 110-PMID TEST split instead, with the configuration
# frozen as-is (no re-tuning against test results — that would just move
# the same leakage problem up one level).
#
# Writes to a NEW file (results/final_evaluation_testset.csv), not
# overwriting results/final_evaluation.csv, which stays the val-set record.
#
# Caveat worth stating alongside these numbers, not hiding: the test set is
# small (148 abstract-level rows), and the warning-stratified subset of
# that will be smaller still (extrapolating from val's ~6.4% warning rate,
# roughly 10-15 sentences) — at that size a couple of sentences either way
# can swing preservation numbers by several points. Report the stratified
# test numbers with n stated prominently; treat them as indicative, not a
# precise estimate.

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.baselines.baseline1_no_simplification import simplify as b1_simplify
from src.baselines.baseline2_rule_based_chv import simplify as b2_simplify
from src.baselines.baseline3_direct_llm import simplify as b3_simplify
from src.pipeline.end_to_end_pipeline import OperationAwarePipeline

from src.evaluation.run_evaluation import evaluate_baseline
from src.complexity.warning_lexicon import flag_sentence as flag_warning
from src.data.pseudo_labeler import build_pairs_from_plaba_json, _DEFAULT_JSON, _TEST_CSV

OUT_PATH = 'results/final_evaluation_testset.csv'

BASELINES = [
    ('baseline1_no_simplification', b1_simplify),
    ('baseline2_rule_based_chv', b2_simplify),
    ('baseline3_direct_llm', b3_simplify),
]


def load_full_test_set():
    """
    Same construction as run_evaluation.py's load_full_val_set(), but
    against _TEST_CSV instead of _VAL_CSV — sentence-level pairs from
    data.json (not test.csv's abstract-level input_text), restricted to
    test PMIDs, grouped by source so multiple adaptation-version
    references become a reference list.
    """
    test_pmids = set(str(p) for p in pd.read_csv(_TEST_CSV)['pmid'].unique())
    df = build_pairs_from_plaba_json(_DEFAULT_JSON)
    df = df[df['pmid'].astype(str).isin(test_pmids)].reset_index(drop=True)

    grouped = df.groupby('source')['target'].apply(list).reset_index()
    sources = grouped['source'].tolist()
    references_list = grouped['target'].tolist()
    return sources, references_list


def main():
    print("Loading full TEST set (sentence-level, from data.json)...")
    sources, references_list = load_full_test_set()
    print(f"  total unique source sentences: {len(sources)}")

    strat_indices = [i for i, s in enumerate(sources) if flag_warning(s)]
    strat_sources = [sources[i] for i in strat_indices]
    strat_refs = [references_list[i] for i in strat_indices]
    print(f"Warning-stratified stress-test subset: "
          f"{len(strat_sources)} / {len(sources)} sentences "
          f"({100 * len(strat_sources) / len(sources):.1f}%)")
    print("NOTE: this subset is small — treat as indicative, not precise "
          "(see module docstring).")
    print()

    results = []

    print("=== OVERALL SPLIT (full test set) ===")
    for name, fn in BASELINES:
        print(f"Running {name}...")
        row = evaluate_baseline(name, fn, sources, references_list, 'overall')
        results.append(row)
        print(f"  {row}")

    print("=== WARNING-STRATIFIED SPLIT (stress test) ===")
    for name, fn in BASELINES:
        print(f"Running {name}...")
        row = evaluate_baseline(name, fn, strat_sources, strat_refs, 'warning_stratified')
        results.append(row)
        print(f"  {row}")

    print("\nLoading OperationAwarePipeline (BioBERT + CHV + constrained FLAN-T5)...")
    pipeline = OperationAwarePipeline(load_llm=True)

    def op_aware_simplify(sentence):
        return pipeline.simplify(sentence)['output']

    print("\n=== OPERATION-AWARE: OVERALL SPLIT ===")
    row = evaluate_baseline('baseline4_operation_aware', op_aware_simplify,
                            sources, references_list, 'overall')
    results.append(row)
    print(f"  {row}")

    print("=== OPERATION-AWARE: WARNING-STRATIFIED SPLIT ===")
    row = evaluate_baseline('baseline4_operation_aware', op_aware_simplify,
                            strat_sources, strat_refs, 'warning_stratified')
    results.append(row)
    print(f"  {row}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
