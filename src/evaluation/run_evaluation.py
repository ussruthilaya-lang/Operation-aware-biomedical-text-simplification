# Full-corpus final evaluation across all baselines.
#
# What changed from prelim:
#   - Runs on the FULL val set (all sentence-level pairs), not a 50-sample.
#   - Adds entity_preservation and numerical_preservation to the metric suite,
#     alongside the existing SARI / FKGL / compression / warning_preservation.
#   - Outputs TWO tables per metric run:
#       (a) overall  -- every val sentence
#       (b) warning-stratified stress test -- only warning-bearing sentences,
#           per the prelim methodological finding that random sampling has
#           insufficient power to detect rare safety failures.
#   - Writes to results/final_evaluation.csv (does NOT overwrite the prelim
#     numbers in results/prelim_evaluation.csv).
#
# Same registry-style BASELINES list as prelim: adding a new system is one
# line, not five edits across the file.

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.baselines.baseline1_no_simplification import simplify as b1_simplify
from src.baselines.baseline2_rule_based_chv import simplify as b2_simplify
from src.baselines.baseline3_direct_llm import simplify as b3_simplify

from src.evaluation.metrics import (
    compute_corpus_sari,
    compute_fkgl,
    compute_compression_ratio,
)
from src.evaluation.warning_preservation import compute_corpus_warning_preservation
from src.evaluation.entity_preservation import compute_corpus_entity_preservation
from src.evaluation.numerical_preservation import compute_corpus_numerical_preservation

from src.data.pseudo_labeler import build_pairs_from_plaba_json, _DEFAULT_JSON, _VAL_CSV
from src.complexity.warning_lexicon import flag_sentence as flag_warning
from src.pipeline.end_to_end_pipeline import OperationAwarePipeline


OUT_PATH = 'results/final_evaluation.csv'

def b4_simplify(sentence):
    global _b4_pipeline
    if _b4_pipeline is None:
        _b4_pipeline = OperationAwarePipeline(load_llm=True)
    return _b4_pipeline.simplify(sentence)['output']

BASELINES = [
    ('baseline1_no_simplification', b1_simplify),
    ('baseline2_rule_based_chv', b2_simplify),
    ('baseline3_direct_llm', b3_simplify),
    ('baseline4_operation_aware', b4_simplify),
]


def load_full_val_set():
    """
    Build sentence-level val pairs from data.json (not val.csv's abstract-
    level input_text — the granularity bug from prelim), restricted to val
    PMIDs. Groups by source so multiple adaptation-version references
    become a reference list.

    Returns:
        sources:         list[str]
        references_list: list[list[str]]
    """
    val_pmids = set(str(p) for p in pd.read_csv(_VAL_CSV)['pmid'].unique())
    df = build_pairs_from_plaba_json(_DEFAULT_JSON)
    df = df[df['pmid'].astype(str).isin(val_pmids)].reset_index(drop=True)

    grouped = df.groupby('source')['target'].apply(list).reset_index()
    sources = grouped['source'].tolist()
    references_list = grouped['target'].tolist()
    return sources, references_list


def evaluate_baseline(name, simplify_fn, sources, references_list, split_label):
    """
    Runs one baseline's simplify_fn across all sources and computes the
    full metric suite. `split_label` is just a tag for the output row
    (e.g. 'overall' or 'warning_stratified').
    """
    hypotheses = [simplify_fn(s) for s in sources]

    sari = compute_corpus_sari(sources, hypotheses, references_list)
    fkgl = sum(compute_fkgl(h) for h in hypotheses) / len(hypotheses)
    cr = sum(compute_compression_ratio(s, h)
             for s, h in zip(sources, hypotheses)) / len(hypotheses)

    pairs = list(zip(sources, hypotheses))
    wp = compute_corpus_warning_preservation(pairs)['mean_score']
    ep = None  # skipped: NER env is a separate conda env (Sruthilaya's setup)
    np_ = compute_corpus_numerical_preservation(pairs)['mean_score']

    return {
        'split': split_label,
        'baseline': name,
        'n_sentences': len(sources),
        'SARI': round(sari, 3),
        'FKGL': round(fkgl, 3),
        'compression_ratio': round(cr, 3),
        'warning_preservation': round(wp, 3),
        'entity_preservation': round(ep, 3) if ep is not None else 'N/A',
        'numerical_preservation': round(np_, 3),
    }


def main():
    print("Loading full val set (sentence-level, from data.json)...")
    sources, references_list = load_full_val_set()
    print(f"  total unique source sentences: {len(sources)}")
    print(f"  reference counts per source: "
          f"min={min(len(r) for r in references_list)}, "
          f"max={max(len(r) for r in references_list)}")
    print()

    # Warning-stratified subset: only sentences flagged by warning_lexicon.
    # Per prelim methodological finding: random sampling has too little
    # statistical power to detect rare safety failures, so we report
    # both the overall table and this stress-test subset.
    strat_indices = [i for i, s in enumerate(sources) if flag_warning(s)]
    strat_sources = [sources[i] for i in strat_indices]
    strat_refs = [references_list[i] for i in strat_indices]
    print(f"Warning-stratified stress-test subset: "
          f"{len(strat_sources)} / {len(sources)} sentences "
          f"({100 * len(strat_sources) / len(sources):.1f}%)")
    print()

    results = []

    print("=== OVERALL SPLIT (full val set) ===")
    for name, fn in BASELINES:
        print(f"Running {name}...")
        row = evaluate_baseline(name, fn, sources, references_list, 'overall')
        results.append(row)
        print(f"  {row}")
        print()

    print("=== WARNING-STRATIFIED SPLIT (stress test) ===")
    for name, fn in BASELINES:
        print(f"Running {name}...")
        row = evaluate_baseline(name, fn, strat_sources, strat_refs,
                                'warning_stratified')
        results.append(row)
        print(f"  {row}")
        print()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()