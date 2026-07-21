"""
PLABA data loader.

Central place to read the dataset. Everyone was calling
pd.read_csv('data/plaba/train.csv') on their own, so encoding fixes and
preprocessing had to be repeated in every file. This module is the one place
to load a split, so any fix happens once here.

Usage:
    from src.data.plaba_loader import load_train, load_val, load_test, load_all

    train = load_train()          # DataFrame
    val   = load_val()            # DataFrame
    test  = load_test()           # DataFrame
    df    = load_all()            # all three, with a 'split' column

CSV files expected in data/plaba/:
    train.csv (635 rows)
    val.csv   (138 rows)
    test.csv  (148 rows)

IMPORTANT — ROWS ARE NOT ABSTRACTS. READ BEFORE COMPUTING ANY ABSTRACT-LEVEL
STATISTIC (e.g. "N abstracts flagged by detector X").

train.csv has 635 rows but only 531 UNIQUE PMIDs (530 after excluding one
row with a blank/missing pmid — see below). The extra rows are not extra
abstracts: 104 PMIDs each have two rows because PLABA provides two human
adaptation versions for those source sentences (427 PMIDs x 1 row + 104
PMIDs x 2 rows = 635). Confirmed via:

    df['pmid'].value_counts().value_counts()
    # -> {1: 427, 2: 104}   (427 + 208 = 635)

If you treat each row as one abstract, PMIDs with two adaptation rows get
counted TWICE — this is exactly what produced an earlier internal report
of "635 training abstracts" for what is actually 530-531 unique abstracts.
Any abstract-level analysis (complexity detection rates, co-occurrence
tables, etc.) must deduplicate on 'pmid' first. See
src/evaluation/run_complexity_analysis.py's load_training_abstracts() for
the correct pattern: group by pmid, treat an abstract as "flagged" if ANY
of its rows/sentences trigger a detector, and never just count rows.

One row in train.csv also has a blank pmid (empty string, not NaN — use
df['pmid'].astype(str).str.strip() == '' to detect it, not .isna()). It
has real source/target text (about chronic cough) but can't be traced to
a PMID, so it can't be grouped into any abstract. Excluded from the
530-abstract figure above; included in the raw 635-row / 531-PMID counts.
"""

import os

import pandas as pd


DATA_DIR = "data/plaba"


def _load_csv(split, data_dir):
    """Read one split CSV. utf-8-sig strips the BOM some exports add."""
    path = os.path.join(data_dir, split + ".csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download PLABA from https://osf.io/rnpmf/ "
            f"and put the CSVs in {data_dir}/."
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def load_train(data_dir=DATA_DIR):
    """
    Load the training split as a DataFrame.

    Returns 635 ROWS, not 635 abstracts — see module docstring's
    "ROWS ARE NOT ABSTRACTS" section. Use df['pmid'].nunique() (or
    dedupe on 'pmid' first) for any abstract-level count or analysis.
    """
    return _load_csv("train", data_dir)


def load_val(data_dir=DATA_DIR):
    """Load the validation split as a DataFrame."""
    return _load_csv("val", data_dir)


def load_test(data_dir=DATA_DIR):
    """Load the test split as a DataFrame."""
    return _load_csv("test", data_dir)


def load_all(data_dir=DATA_DIR):
    """
    Load all three splits into one DataFrame with a 'split' column
    marking each row as 'train', 'val', or 'test'.
    """
    frames = []
    for split in ("train", "val", "test"):
        df = _load_csv(split, data_dir)
        df = df.copy()
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    # Sanity check. Run: python -m src.data.plaba_loader
    for name, loader in (("train", load_train), ("val", load_val),
                         ("test", load_test)):
        df = loader()
        n_unique_pmid = df["pmid"].nunique() if "pmid" in df.columns else "n/a"
        print(f"{name:5s}  rows={len(df):4d}  unique_pmids={n_unique_pmid}  "
              f"cols={list(df.columns)}")
    combined = load_all()
    print(f"all    rows={len(combined):4d}  splits={combined['split'].value_counts().to_dict()}")