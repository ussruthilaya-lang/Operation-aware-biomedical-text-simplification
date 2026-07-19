"""
Runs all 5 complexity detectors over the full PLABA training set and
produces a co-occurrence table matching the prelim's format.

GRANULARITY (explicit, per Bug 3 lesson): PLABA's data.json is sentence-level
(source/target pairs). We group sentences by pmid to reconstruct abstracts,
then an abstract is "flagged" by a detector if ANY of its sentences trigger
that detector — matching the prelim's "Abstracts flagged" methodology.

Run with either backend:
    UMLS_BACKEND=morphological python run_complexity_analysis.py
    UMLS_BACKEND=quickumls python run_complexity_analysis.py

Output: results/complexity_cooccurrence_{backend}.csv
"""

import os
import sys
import json
import subprocess
import tempfile
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.data.pseudo_labeler import (
    build_pairs_from_plaba_json, _DEFAULT_JSON, _VAL_CSV, _TEST_CSV
)
from src.complexity.warning_lexicon import flag_sentence as flag_warning
from src.complexity.syntactic_depth import flag_sentence as flag_syntactic
from src.complexity.numerical_extractor import flag_sentence as flag_numerical
from src.complexity.umls_matcher import (
    flag_sentence as flag_umls, UMLS_BACKEND, QUICKUMLS_AVAILABLE,
    QUICKUMLS_INDEX_PATH
)

OUT_PATH = f"results/complexity_cooccurrence_{UMLS_BACKEND}.csv"

NER_ENV_PYTHON = os.path.expanduser("~/miniconda3/envs/ner_env/bin/python")
NER_RUNNER_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "complexity", "ner_runner_subprocess.py"
)


def run_ner_batch(all_sentences):
    """
    Runs NER flagging for ALL sentences in one subprocess call (not per-sentence,
    not per-abstract) — avoids paying Python/spaCy startup cost hundreds of times.

    Returns: dict {sentence: bool} for O(1) lookup during per-abstract aggregation.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_in:
        json.dump({"sentences": all_sentences}, f_in)
        input_path = f_in.name

    output_path = input_path.replace('.json', '_out.json')

    print(f"  calling ner_env subprocess for {len(all_sentences)} sentences...")
    result = subprocess.run(
        [NER_ENV_PYTHON, NER_RUNNER_SCRIPT, input_path, output_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("NER subprocess failed:", result.stderr)
        raise RuntimeError("ner_runner_subprocess.py failed")

    with open(output_path) as f_out:
        flags = json.load(f_out)["flags"]

    os.unlink(input_path)
    os.unlink(output_path)

    return dict(zip(all_sentences, flags))


def load_training_abstracts():
    """
    Builds sentence-level pairs from data.json, excludes val/test PMIDs,
    groups remaining sentences by pmid to reconstruct abstracts.

    Returns: dict {pmid: [sentence1, sentence2, ...]}
    """
    val_pmids = set(str(p) for p in pd.read_csv(_VAL_CSV)['pmid'].unique())
    test_pmids = set(str(p) for p in pd.read_csv(_TEST_CSV)['pmid'].unique())
    excluded = val_pmids | test_pmids

    df = build_pairs_from_plaba_json(_DEFAULT_JSON)
    df['pmid'] = df['pmid'].astype(str)
    train_df = df[~df['pmid'].isin(excluded)].reset_index(drop=True)

    # Dedup: multiple target references can share the same source sentence
    unique_sentences = train_df[['pmid', 'source']].drop_duplicates()

    abstracts = unique_sentences.groupby('pmid')['source'].apply(list).to_dict()
    return abstracts


def get_umls_matcher():
    """Instantiates the raw QuickUMLS matcher if backend=quickumls, else None."""
    if UMLS_BACKEND == "quickumls":
        if not QUICKUMLS_AVAILABLE:
            raise ImportError("UMLS_BACKEND=quickumls but quickumls package not installed.")
        from quickumls import QuickUMLS
        # Match the same filtering as QuickUMLSStrategy for consistency
        from src.complexity.umls_matcher import QuickUMLSStrategy
        return QuickUMLSStrategy().matcher
    return None


def analyze_abstracts(abstracts, umls_matcher, ner_flags):
    """
    Runs 4 detectors live per-sentence, uses pre-computed ner_flags (dict)
    for NER since that runs in a separate subprocess/environment.
    """
    rows = []
    for pmid, sentences in abstracts.items():
        row = {
            'pmid': pmid,
            'n_sentences': len(sentences),
            'ner': any(ner_flags[s] for s in sentences),
            'warning': any(flag_warning(s) for s in sentences),
            'syntactic': any(flag_syntactic(s) for s in sentences),
            'numerical': any(flag_numerical(s) for s in sentences),
            'umls': any(flag_umls(s, umls_matcher) for s in sentences),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(df):
    detector_cols = ['ner', 'warning', 'syntactic', 'numerical', 'umls']
    n = len(df)

    print(f"\n=== Detector flag rates (n={n} abstracts) ===")
    for col in detector_cols:
        flagged = df[col].sum()
        print(f"  {col:12s}: {flagged:4d} ({100*flagged/n:.1f}%)")

    df['n_detectors_firing'] = df[detector_cols].sum(axis=1)
    print(f"\n=== Co-occurrence: detectors firing per abstract ===")
    counts = df['n_detectors_firing'].value_counts().sort_index()
    for k, v in counts.items():
        print(f"  {k}: {v:4d} ({100*v/n:.1f}%)")

    three_plus = (df['n_detectors_firing'] >= 3).sum()
    print(f"\n  {three_plus}/{n} ({100*three_plus/n:.1f}%) abstracts have 3+ detectors firing")


def main():
    print(f"UMLS_BACKEND: {UMLS_BACKEND}")
    print(f"QuickUMLS available: {QUICKUMLS_AVAILABLE}")

    print("\nLoading training abstracts (excluding val/test PMIDs)...")
    abstracts = load_training_abstracts()
    print(f"  shape: {len(abstracts)} abstracts")
    example_pmid = next(iter(abstracts))
    print(f"  worked example — pmid={example_pmid}, "
          f"{len(abstracts[example_pmid])} sentences, "
          f"first sentence: {abstracts[example_pmid][0][:80]}...")

    print("\nInitializing UMLS matcher...")
    umls_matcher = get_umls_matcher()
    print(f"  matcher: {'QuickUMLS instance' if umls_matcher else 'None (using heuristic fallback)'}")

    print("\nRunning NER via subprocess (ner_env)...")
    all_sentences = [s for sents in abstracts.values() for s in sents]
    ner_flags = run_ner_batch(all_sentences)

    print("\nRunning remaining detectors over each abstract...")
    df = analyze_abstracts(abstracts, umls_matcher, ner_flags)

    print_summary(df)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()