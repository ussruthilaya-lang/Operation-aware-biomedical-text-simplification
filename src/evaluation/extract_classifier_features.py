"""
Extracts UMLS jargon, warning, and numerical features per sentence, for
Sophakotra's classifier feature-integration experiment (Model 2 / Model 3).

WHY A HANDOFF CSV, NOT SHARED GCP ACCESS: these features are deterministic
given the fixed PLABA sentences already in the repo. Computing them once
here (with the validated real-UMLS backend + all 3 bug fixes applied) and
handing off a CSV is faster and more reliable than onboarding a teammate to
the QuickUMLS/UMLS-index infra just to read feature values — no VM, no GCP
IAM, no environment setup needed on his end.

Deliberately EXCLUDES syntactic depth / length-based features — per the
scoping decision, these risk leaking the length-ratio pseudo-labeling rule
itself (see paper draft's Main Idea / status doc).

Output: results/classifier_features.csv, one row per (pmid, adaptation,
sent_id) — same keys as pseudo_labeler.py's build_pairs_from_plaba_json(),
so Sophakotra can join directly on those columns.

Usage (must run with UMLS_BACKEND=quickumls for the validated real jargon
detector, not the morphological fallback):
    UMLS_BACKEND=quickumls python src/evaluation/extract_classifier_features.py
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.data.pseudo_labeler import (
    build_pairs_from_plaba_json, label_dataset, _DEFAULT_JSON
)
from src.complexity.umls_matcher import detect_umls_terms, UMLS_BACKEND, QUICKUMLS_AVAILABLE
from src.complexity.warning_lexicon import detect_warnings
from src.complexity.numerical_extractor import extract_numerical_expressions

OUT_PATH = "results/classifier_features.csv"


def get_umls_matcher_instance():
    """Instantiate the real QuickUMLS matcher if backend=quickumls."""
    if UMLS_BACKEND == "quickumls":
        if not QUICKUMLS_AVAILABLE:
            raise ImportError(
                "UMLS_BACKEND=quickumls but quickumls package not installed. "
                "This script should be run in umls_env on a machine/VM with "
                "the real UMLS index — see infra/setup_quickumls.sh."
            )
        from src.complexity.umls_matcher import QuickUMLSStrategy
        return QuickUMLSStrategy().matcher
    return None


def extract_features_for_sentence(text, umls_matcher):
    """
    Returns a dict of the 3 selected feature groups for one sentence.
    Deliberately no syntactic/length features (leakage risk — see module docstring).
    """
    word_count = max(len(text.split()), 1)  # avoid div-by-zero on empty strings

    # --- UMLS jargon features ---
    umls_terms = detect_umls_terms(text, umls_matcher)
    umls_jargon_count = len(umls_terms)
    umls_jargon_present = umls_jargon_count > 0
    umls_jargon_density = umls_jargon_count / word_count

    # --- Warning features (negation-aware, per this session's bug fix) ---
    warnings = detect_warnings(text)
    warning_present = len(warnings) > 0
    warning_count = len(warnings)
    # Categories present, useful if Sophakotra wants finer-grained features
    # later than a single binary flag
    warning_categories = sorted(set(w['category'] for w in warnings))

    # --- Numerical features (decimal-truncation bug fixed this session) ---
    numerical_matches = extract_numerical_expressions(text)
    numerical_present = len(numerical_matches) > 0
    numerical_count = len(numerical_matches)
    numerical_density = numerical_count / word_count

    return {
        'umls_jargon_present': umls_jargon_present,
        'umls_jargon_count': umls_jargon_count,
        'umls_jargon_density': round(umls_jargon_density, 4),
        'warning_present': warning_present,
        'warning_count': warning_count,
        'warning_categories': ';'.join(warning_categories),
        'numerical_present': numerical_present,
        'numerical_count': numerical_count,
        'numerical_density': round(numerical_density, 4),
    }


def main():
    print(f"UMLS_BACKEND: {UMLS_BACKEND}")
    if UMLS_BACKEND != "quickumls":
        print("WARNING: not using quickumls backend — UMLS jargon features "
              "will be computed with the morphological fallback, NOT the "
              "validated real-UMLS detector. Re-run with "
              "UMLS_BACKEND=quickumls before handing off to Sophakotra.")

    print("\nBuilding full sentence-pair dataset (all splits, pseudo-labeled)...")
    # No PMID exclusion here — we want features for train+val+test so
    # Sophakotra can join against whichever split he's evaluating.
    df = build_pairs_from_plaba_json(_DEFAULT_JSON)
    df = label_dataset(df)
    print(f"  shape: {len(df)} sentence pairs")
    print(f"  worked example — pmid={df.iloc[0]['pmid']}, "
          f"source: {df.iloc[0]['source'][:80]}...")

    print("\nInitializing UMLS matcher...")
    umls_matcher = get_umls_matcher_instance()
    print(f"  matcher: {'QuickUMLS instance' if umls_matcher else 'None (morphological fallback)'}")

    print(f"\nExtracting features for {len(df)} unique source sentences...")
    # Dedup on source text — feature values only depend on the source
    # sentence, so no need to recompute for every (adaptation, target) row
    unique_sources = df['source'].drop_duplicates().tolist()
    feature_rows = []
    for i, source in enumerate(unique_sources):
        if i % 200 == 0:
            print(f"  {i}/{len(unique_sources)}...")
        feats = extract_features_for_sentence(source, umls_matcher)
        feats['source'] = source
        feature_rows.append(feats)

    features_df = pd.DataFrame(feature_rows)
    print(f"  extracted features for {len(features_df)} unique sentences")

    # Join back onto the full pmid/adaptation/sent_id/operation table so
    # Sophakotra has everything in one file, keyed the same way as his
    # existing pseudo-labeled training data.
    merged = df.merge(features_df, on='source', how='left')

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(f"Columns: {list(merged.columns)}")
    print(f"\nFeature summary:")
    print(f"  UMLS jargon present: {features_df['umls_jargon_present'].sum()} / {len(features_df)}")
    print(f"  Warning present: {features_df['warning_present'].sum()} / {len(features_df)}")
    print(f"  Numerical present: {features_df['numerical_present'].sum()} / {len(features_df)}")


if __name__ == "__main__":
    main()