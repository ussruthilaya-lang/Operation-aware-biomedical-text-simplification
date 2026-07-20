"""
Generates a warning-stratified sample for human evaluation.

WHY STRATIFIED, NOT RANDOM (per Bug 3 / prelim methodological finding):
warning-bearing sentences occur at only ~33% base rate. A purely random
sample of 30-50 sentences risks drawing too few (or zero) warning-bearing
cases to meaningfully test the safety dimension of the rubric below. We
therefore explicitly oversample warning-bearing sentences to guarantee
adequate representation of this rare-but-safety-critical case, mirroring
the same stratified-sampling discipline used for the warning_preservation
stress test in the prelim.

WHY VAL SET, NOT TRAIN: training data is used to build/tune detectors and
baselines; human evaluation should test system behavior on held-out data
the system was never fit against, matching standard ML evaluation practice.

Output: a single CSV, one row per (sentence, baseline) pair, with five
blank rating columns for annotators to fill in on a 1-5 scale.
"""

import os
import sys
import json
import random
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.data.pseudo_labeler import build_pairs_from_plaba_json, _DEFAULT_JSON, _VAL_CSV
from src.complexity.warning_lexicon import flag_sentence as flag_warning
from src.baselines.baseline1_no_simplification import simplify as b1_simplify
from src.baselines.baseline2_rule_based_chv import simplify as b2_simplify
from src.baselines.baseline3_direct_llm import simplify as b3_simplify

RANDOM_SEED = 42
N_WARNING_BEARING = 20   # oversampled — target ~50% of the sample
N_NON_WARNING = 20       # remainder, so the sample isn't ONLY worst-case sentences
OUT_PATH = "results/human_eval_sample.csv"

BASELINES = [
    ("baseline1_no_simplification", b1_simplify),
    ("baseline2_rule_based_chv", b2_simplify),
    ("baseline3_direct_llm", b3_simplify),
]

RATING_DIMENSIONS = [
    "correctness", "completeness", "readability", "actionability", "safety"
]


def load_val_sentences():
    """
    Builds sentence-level pairs from data.json, restricted to val PMIDs only.
    Deduplicates by source sentence (multiple target references can share
    the same source).

    Returns: list of unique source sentences from the held-out val set.
    """
    val_pmids = set(str(p) for p in pd.read_csv(_VAL_CSV)['pmid'].unique())
    df = build_pairs_from_plaba_json(_DEFAULT_JSON)
    df['pmid'] = df['pmid'].astype(str)
    val_df = df[df['pmid'].isin(val_pmids)].reset_index(drop=True)

    unique_sentences = val_df['source'].drop_duplicates().tolist()
    return unique_sentences


def build_stratified_sample(sentences):
    """
    Splits sentences into warning-bearing vs. non-warning-bearing, then
    samples N_WARNING_BEARING and N_NON_WARNING from each pool respectively.

    Returns: list of (sentence, is_warning_bearing) tuples.
    """
    random.seed(RANDOM_SEED)

    warning_bearing = [s for s in sentences if flag_warning(s)]
    non_warning = [s for s in sentences if not flag_warning(s)]

    print(f"  warning-bearing pool: {len(warning_bearing)} sentences")
    print(f"  non-warning pool: {len(non_warning)} sentences")

    n_warn = min(N_WARNING_BEARING, len(warning_bearing))
    n_non = min(N_NON_WARNING, len(non_warning))

    sampled_warning = random.sample(warning_bearing, n_warn)
    sampled_non = random.sample(non_warning, n_non)

    sample = [(s, True) for s in sampled_warning] + [(s, False) for s in sampled_non]
    random.shuffle(sample)  # so annotators don't see all warning cases grouped together
    return sample


def build_annotation_rows(sample):
    """
    For each (sentence, is_warning_bearing) pair, generates output from all
    3 baselines and produces one row per (sentence, baseline) combination,
    with blank rating columns for annotators.
    """
    rows = []
    row_id = 1
    for sentence, is_warning in sample:
        for baseline_name, simplify_fn in BASELINES:
            output = simplify_fn(sentence)
            row = {
                'id': row_id,
                'source': sentence,
                'baseline': baseline_name,
                'system_output': output,
                'warning_bearing': is_warning,
            }
            for dim in RATING_DIMENSIONS:
                row[dim] = ""  # blank for annotator to fill, 1-5 scale
            rows.append(row)
            row_id += 1
    return rows


def main():
    print("Loading val set sentences...")
    sentences = load_val_sentences()
    print(f"  shape: {len(sentences)} unique sentences")
    print(f"  worked example: {sentences[0][:80]}...")

    print("\nBuilding warning-stratified sample...")
    sample = build_stratified_sample(sentences)
    print(f"  total sampled sentences: {len(sample)} "
          f"({sum(1 for _, w in sample if w)} warning-bearing, "
          f"{sum(1 for _, w in sample if not w)} non-warning)")

    print("\nGenerating baseline outputs for each sampled sentence...")
    rows = build_annotation_rows(sample)
    print(f"  total annotation rows: {len(rows)} "
          f"({len(sample)} sentences x {len(BASELINES)} baselines)")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()