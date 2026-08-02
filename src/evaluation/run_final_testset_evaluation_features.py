# Same test-set evaluation as run_final_testset_evaluation.py, but for the
# BioBERT+UMLS-features pipeline variant (src/pipeline/end_to_end_pipeline_features.py)
# instead of plain BioBERT.
#
# Baselines 1-3 (no-simplification, rule-based CHV, direct LLM) don't depend
# on the classifier at all, so their numbers are identical to
# results/final_evaluation_testset.csv -- copied over rather than rerun, to
# save the ~time cost of re-running FLAN-T5 baseline3 for no reason. Only
# the operation-aware row is new here (baseline4_operation_aware_features),
# so the two CSVs can be diffed directly on that one row to isolate the
# effect of the classifier swap.

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.pipeline.end_to_end_pipeline_features import OperationAwarePipelineWithFeatures

from src.evaluation.run_evaluation import evaluate_baseline
from src.complexity.warning_lexicon import flag_sentence as flag_warning
from src.data.pseudo_labeler import build_pairs_from_plaba_json, _DEFAULT_JSON, _TEST_CSV

OUT_PATH = 'results/final_evaluation_testset_features.csv'
BASELINE_SOURCE_CSV = 'results/final_evaluation_testset.csv'


def load_full_test_set():
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

    print("\nCopying baselines 1-3 from the existing test-set evaluation "
          f"({BASELINE_SOURCE_CSV}) -- unaffected by the classifier swap...")
    existing = pd.read_csv(BASELINE_SOURCE_CSV)
    carried_over = existing[existing['baseline'] != 'baseline4_operation_aware'].copy()
    results = carried_over.to_dict('records')

    print("\nLoading OperationAwarePipelineWithFeatures "
          "(BioBERT+UMLS-features + CHV + constrained FLAN-T5)...")
    pipeline = OperationAwarePipelineWithFeatures(load_llm=True)

    def op_aware_simplify(sentence):
        return pipeline.simplify(sentence)['output']

    print("\n=== OPERATION-AWARE (FEATURES): OVERALL SPLIT ===")
    row = evaluate_baseline('baseline4_operation_aware_features', op_aware_simplify,
                            sources, references_list, 'overall')
    results.append(row)
    print(f"  {row}")

    print("=== OPERATION-AWARE (FEATURES): WARNING-STRATIFIED SPLIT ===")
    row = evaluate_baseline('baseline4_operation_aware_features', op_aware_simplify,
                            strat_sources, strat_refs, 'warning_stratified')
    results.append(row)
    print(f"  {row}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
