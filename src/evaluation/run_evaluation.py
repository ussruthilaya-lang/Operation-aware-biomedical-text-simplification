# Runs prelim evaluation over baseline1 (no simplification) and
# baseline2 (rule-based CHV) on a 50-sentence sample from the val set.
#
# Output: results/prelim_evaluation.csv
# Columns: baseline | SARI | FKGL | compression_ratio | warning_preservation
#
# Why 50 sentences: the PRELIM brief only requires showing the framework
# works. Full corpus numbers are for the final paper.

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.baselines.baseline1_no_simplification import simplify as b1_simplify
from src.baselines.baseline2_rule_based_chv import simplify as b2_simplify
from src.evaluation.metrics import (
    compute_corpus_sari,
    compute_fkgl,
    compute_compression_ratio,
)
from src.evaluation.warning_preservation import compute_corpus_warning_preservation


SAMPLE_SIZE = 50
RANDOM_SEED = 42
VAL_PATH = 'data/plaba/val.csv'
OUT_PATH = 'results/prelim_evaluation.csv'


def load_val_sample():
    """
    Group val rows by input_text so multiple Adaptation_Versions become
    a reference list. Then take a fixed-seed random sample of SAMPLE_SIZE
    unique sources.
    """
    df = pd.read_csv(VAL_PATH)
    grouped = (
        df.groupby('input_text')['target_text']
          .apply(list)
          .reset_index()
    )
    n = min(SAMPLE_SIZE, len(grouped))
    sample = grouped.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    sources = sample['input_text'].tolist()
    references_list = sample['target_text'].tolist()
    return sources, references_list


def evaluate_baseline(name, simplify_fn, sources, references_list):
    hypotheses = [simplify_fn(s) for s in sources]

    sari = compute_corpus_sari(sources, hypotheses, references_list)

    # Mean per-sentence FKGL and compression ratio
    fkgl = sum(compute_fkgl(h) for h in hypotheses) / len(hypotheses)
    cr = sum(compute_compression_ratio(s, h)
             for s, h in zip(sources, hypotheses)) / len(hypotheses)

    pairs = list(zip(sources, hypotheses))
    wp = compute_corpus_warning_preservation(pairs)['mean_score']

    return {
        'baseline': name,
        'SARI': round(sari, 3),
        'FKGL': round(fkgl, 3),
        'compression_ratio': round(cr, 3),
        'warning_preservation': round(wp, 3),
    }


def main():
    print(f"Loading {SAMPLE_SIZE}-sentence val sample (seed={RANDOM_SEED})...")
    sources, references_list = load_val_sample()
    print(f"  loaded {len(sources)} unique sources")
    print(f"  reference counts per source: "
          f"min={min(len(r) for r in references_list)}, "
          f"max={max(len(r) for r in references_list)}")
    print()

    results = []
    for name, fn in [
        ('baseline1_no_simplification', b1_simplify),
        ('baseline2_rule_based_chv', b2_simplify),
    ]:
        print(f"Running {name}...")
        row = evaluate_baseline(name, fn, sources, references_list)
        results.append(row)
        print(f"  {row}")
        print()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()