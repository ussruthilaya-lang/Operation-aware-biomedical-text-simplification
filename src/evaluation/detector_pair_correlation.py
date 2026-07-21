"""
Per-pair detector co-occurrence analysis.

Extends the base co-occurrence table (how many detectors fire per abstract)
with pairwise analysis: which two detectors most often fire together, and
whether any pairing is unusually strong or weak relative to chance.

Uses phi coefficient (binary-variable equivalent of Pearson correlation) to
quantify pairwise association strength, not just raw co-occurrence counts,
since raw counts alone are misleading when base rates differ a lot (e.g.
UMLS fires on 100% of abstracts, so it "co-occurs" with everything trivially
often — phi coefficient corrects for this by comparing against what
independence would predict).

NOTE: this analyzes detector-level co-occurrence only. Whether a given
detector PAIR predicts a specific PLABA operation type (Substitution/
Explanation/Generalization) requires Sophakotra's operation classifier
output, which this script does not depend on — flagged as a natural
follow-up once that's available, not built here.
"""

import sys
import os
import itertools
import pandas as pd
import numpy as np

IN_PATH = "results/complexity_cooccurrence_quickumls.csv"
OUT_PATH = "results/detector_pair_correlation.csv"

DETECTOR_COLS = ['ner', 'warning', 'syntactic', 'numerical', 'umls']


def phi_coefficient(col_a, col_b):
    """
    Phi coefficient for two binary columns — equivalent to Pearson
    correlation for binary data. Ranges -1 to 1; 0 means the two
    detectors fire independently of each other (no more or less often
    together than chance would predict given their individual rates).
    """
    a = col_a.astype(int)
    b = col_b.astype(int)
    n11 = ((a == 1) & (b == 1)).sum()
    n10 = ((a == 1) & (b == 0)).sum()
    n01 = ((a == 0) & (b == 1)).sum()
    n00 = ((a == 0) & (b == 0)).sum()
    n = n11 + n10 + n01 + n00

    numerator = (n11 * n00) - (n10 * n01)
    denominator = np.sqrt(
        (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    )
    if denominator == 0:
        return 0.0  # undefined (one detector has zero variance) — treat as no correlation
    return numerator / denominator


def main():
    print(f"Loading {IN_PATH}...")
    df = pd.read_csv(IN_PATH)
    print(f"  shape: {df.shape[0]} abstracts, columns: {list(df.columns)}")

    missing = [c for c in DETECTOR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected detector columns missing from CSV: {missing}")

    print("\n=== Individual detector flag rates (sanity check vs. known values) ===")
    for col in DETECTOR_COLS:
        rate = df[col].sum() / len(df)
        print(f"  {col:12s}: {100*rate:.1f}%")

    print("\n=== Pairwise co-occurrence (raw counts + phi coefficient) ===")
    results = []
    for a, b in itertools.combinations(DETECTOR_COLS, 2):
        both = ((df[a] == True) & (df[b] == True)).sum()
        only_a = ((df[a] == True) & (df[b] == False)).sum()
        only_b = ((df[a] == False) & (df[b] == True)).sum()
        neither = ((df[a] == False) & (df[b] == False)).sum()
        phi = phi_coefficient(df[a], df[b])

        results.append({
            'detector_a': a,
            'detector_b': b,
            'both_fire': both,
            'both_fire_pct': round(100 * both / len(df), 1),
            'only_a': only_a,
            'only_b': only_b,
            'neither': neither,
            'phi_coefficient': round(phi, 3),
        })

        print(f"  {a:12s} + {b:12s}: both={both} ({100*both/len(df):.1f}%), "
              f"phi={phi:.3f}")

    results_df = pd.DataFrame(results).sort_values('phi_coefficient', ascending=False)

    print("\n=== Strongest pairwise associations (by phi coefficient) ===")
    print(results_df[['detector_a', 'detector_b', 'phi_coefficient', 'both_fire_pct']]
          .head(5).to_string(index=False))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    results_df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")

    print("\nNOTE: UMLS fires on 100% of abstracts in this dataset, making its "
          "phi coefficient with any other detector mathematically undefined "
          "or trivially near-zero regardless of true association, since it has "
          "no variance to correlate against. Interpret UMLS-paired rows with "
          "this in mind — the more informative pairs here are among the other "
          "four detectors.")


if __name__ == "__main__":
    main()