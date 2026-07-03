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
    """Load the training split as a DataFrame."""
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
        print(f"{name:5s}  rows={len(df):4d}  cols={list(df.columns)}")
    combined = load_all()
    print(f"all    rows={len(combined):4d}  splits={combined['split'].value_counts().to_dict()}")
