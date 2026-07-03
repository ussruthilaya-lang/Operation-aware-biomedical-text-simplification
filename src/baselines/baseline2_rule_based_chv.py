# Baseline 2: Rule-based CHV substitution + sentence splitting
#
# Why this baseline exists:
# Consumer Health Vocabulary (CHV) is a manually curated mapping of
# medical jargon to plain English. "Myocardial infarction" -> "heart attack".
# This baseline tests: how far can pure rule-based substitution get us
# WITHOUT any ML, any LLM, any learned representation?
#
# Enterprise AI insight: Rule-based systems are auditable, explainable,
# and have zero hallucination risk. In regulated healthcare environments
# (FDA-cleared software, clinical decision support), auditability often
# matters more than accuracy. A rule-based baseline that performs
# reasonably well is a strong argument for conservative deployment.
# This is why Stage 3 routes SUBSTITUTION through CHV, not LLM.

# The CHV dictionary now lives in src/data/chv_lookup.py so the operation
# router and this baseline share one source of truth. get_all_terms() returns
# the full mapping (all of the original baseline2 terms plus the added ones).
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.chv_lookup import get_all_terms

CHV_LOOKUP = get_all_terms()


def chv_substitute(text):
    """
    Replaces medical jargon with CHV plain-language equivalents.
    Case-insensitive exact phrase matching.

    Why exact matching, not fuzzy?
    In clinical NLP, false substitutions are dangerous.
    "Hepatic failure" -> "liver failure" is safe.
    "Hepatic fibrosis" -> "liver failure" (fuzzy) is a medical error.
    Exact matching trades recall for precision, the right call for healthcare.
    """
    result = text
    # Sort by length descending so longer phrases match first.
    # Prevents "carcinoma" matching inside "hepatocellular carcinoma".
    sorted_terms = sorted(CHV_LOOKUP.keys(), key=len, reverse=True)

    for term in sorted_terms:
        if term.lower() in result.lower():
            # Preserve original casing of surrounding text
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            result = pattern.sub(CHV_LOOKUP[term], result)

    return result


def split_long_sentence(text, max_words=35):
    """
    Splits sentences longer than max_words at natural boundaries.
    Targets conjunctions and semicolons, which are syntactically safe points.

    Why 35 words?
    FDA plain language guidelines recommend max 20 words per sentence
    for patient-facing content. 35 is generous for biomedical text
    where some technical context must be preserved.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    # Find split points, conjunctions and semicolons
    split_signals = [
        '; ', ', however, ', ', although ', ', which ',
        ', but ', ', and ', ', therefore ', ', thus '
    ]

    for signal in split_signals:
        if signal in text:
            parts = text.split(signal, 1)
            if len(parts[0].split()) >= 10:  # Don't split too early
                return [parts[0].strip() + '.', parts[1].strip()]

    return [text]  # Can't find clean split point, return as-is


def simplify(text):
    """Full baseline 2 pipeline: CHV substitution + sentence splitting."""
    substituted = chv_substitute(text)
    sentences = split_long_sentence(substituted)
    return ' '.join(sentences)


def run_on_dataset(df, text_column='input_text'):
    pairs = []
    for _, row in df.iterrows():
        source = row[text_column]
        pairs.append((source, simplify(source)))
    return pairs


if __name__ == "__main__":
    import pandas as pd
    import sys, os
    #sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.evaluation.warning_preservation import compute_corpus_warning_preservation

    # Test on examples
    test_sentences = [
        "The patient was diagnosed with myocardial infarction and hypertension.",
        "Angioplasty was performed intravenously to treat renal failure.",
        "This anticoagulant is contraindicated in patients with epistaxis.",
    ]
    print("=== Baseline 2: Rule-based CHV Substitution ===\n")
    for s in test_sentences:
        output = simplify(s)
        print(f"Source:     {s}")
        print(f"Simplified: {output}\n")

    # Run on dataset
    df = pd.read_csv('data/plaba/train.csv')
    pairs = run_on_dataset(df)
    result = compute_corpus_warning_preservation(pairs)
    print(f"Warning preservation rate: {result['mean_score']}")
    print(f"Warning sentences: {result['warning_sentences']}/{result['total_sentences']}")
