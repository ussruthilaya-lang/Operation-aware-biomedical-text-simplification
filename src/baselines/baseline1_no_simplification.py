# Baseline 1: No simplification — safety ceiling anchor
# 
# Why this baseline exists:
# Before measuring any system, you need to know the upper bound
# on safety preservation. If you don't simplify at all, you preserve
# 100% of warnings, entities, and numbers by definition.
# This is the SAFETY CEILING — no system should score higher on
# safety metrics than this. If one does, your safety metric is broken.
#
# Enterprise AI insight: In healthcare NLP, "do nothing" is a legitimate
# and important baseline. A system that simplifies but drops critical
# information is WORSE than no system at all. Regulators and clinical
# informaticists think this way. You should too.

def simplify(text):
    """Returns text unchanged — the no-simplification baseline."""
    return text

def run_on_dataset(df, text_column='input_text'):
    """
    Runs no-simplification baseline on full dataset.
    
    Args:
        df: pandas DataFrame with PLABA data
        text_column: column containing source text
    
    Returns:
        list of (source, output) pairs
    """
    pairs = []
    for _, row in df.iterrows():
        source = row[text_column]
        pairs.append((source, simplify(source)))
    return pairs

if __name__ == "__main__":
    import pandas as pd
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.evaluation.warning_preservation import compute_corpus_warning_preservation

    df = pd.read_csv('data/plaba/train.csv')
    pairs = run_on_dataset(df)
    result = compute_corpus_warning_preservation(pairs)

    print("=== Baseline 1: No Simplification ===")
    print(f"Warning preservation rate: {result['mean_score']}")
    print(f"Warning sentences: {result['warning_sentences']}/{result['total_sentences']}")
    print("Expected: ~1.0 (safety ceiling)")