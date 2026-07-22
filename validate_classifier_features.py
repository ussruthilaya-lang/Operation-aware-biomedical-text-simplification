"""
Validates results/classifier_features.csv before handoff.

Checks:
1. Shape/schema sanity (per the standing rule — print shape/type/example
   before trusting any output)
2. Known-case correctness — re-checks specific sentences we already
   manually validated earlier this session, to confirm the feature
   extraction script computed them the same way (not a regression)
3. Aggregate sanity — feature present-rates should be in a plausible
   range, not 0% or 100% across the board (which would indicate the
   detector didn't actually run, e.g. wrong backend or silent failure)
"""

import pandas as pd

PATH = "results/classifier_features.csv"

df = pd.read_csv(PATH)

print("=== 1. Shape / schema check ===")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Unique sentences (source): {df['source'].nunique()}")
print()
print("Worked example (first row):")
print(df.iloc[0][['pmid', 'source', 'operation', 'umls_jargon_present',
                   'warning_present', 'numerical_present']])
print()

print("=== 2. Known-case spot checks ===")
# These are sentences (or close paraphrases) we manually validated earlier
# this session — confirming the extraction script agrees with what we
# already know to be correct.
checks = [
    ("hepatocellular carcinoma", "umls_jargon_present", True,
     "Known jargon term — validated earlier via random-sample review"),
    ("no adverse events", "warning_present", False,
     "Negation case — should NOT be flagged after this session's negation fix"),
    ("16.8 years", "numerical_present", True,
     "Decimal case — should be flagged correctly after decimal-truncation fix"),
]

for substring, col, expected, note in checks:
    matches = df[df['source'].str.contains(substring, case=False, na=False)]
    if len(matches) == 0:
        print(f"  [SKIP] No sentence containing '{substring}' found in this dataset")
        continue
    actual = matches.iloc[0][col]
    status = "PASS" if actual == expected else "FAIL"
    print(f"  [{status}] '{substring}' -> {col}={actual} (expected {expected}) — {note}")
print()

print("=== 3. Aggregate sanity check ===")
for col in ['umls_jargon_present', 'warning_present', 'numerical_present']:
    rate = df[col].mean()
    flag = "⚠️  SUSPICIOUS" if rate == 0.0 or rate == 1.0 else "OK"
    print(f"  {col:25s}: {100*rate:.1f}% present  [{flag}]")

print()
print("If any spot-check FAILed, or any aggregate rate shows SUSPICIOUS,")
print("do NOT hand off this file yet — investigate before committing.")