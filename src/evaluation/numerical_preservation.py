# Why a custom metric instead of ROUGE/BLEU?
# Same rationale as warning_preservation.py and entity_preservation.py:
# ROUGE measures n-gram overlap and doesn't care whether a specific
# dosage, risk ratio, or p-value survived simplification with its exact
# value intact. This metric checks numerical survival directly.
#
# IMPORTANT DEVIATION FROM THE SIBLING METRICS' MATCHING LOGIC:
# warning_preservation.py and entity_preservation.py both use partial-word
# matching (any word >4 chars shared counts as "preserved"), because a
# paraphrase ("contraindicated" -> "do not use", "hepatocellular
# carcinoma" -> "liver cancer") is often an acceptable, even correct,
# simplification.
#
# Numbers are different: "500mg" substituted with "50mg" is not a
# paraphrase, it is a dangerous factual error. There is no valid
# "paraphrase" of a dosage or p-value the way there is for a warning
# phrase or a disease name. Partial-word matching would also be actively
# wrong here for a different reason: matching only the unit word ("mg")
# without the number would falsely credit "500mg" as preserved even if
# the output said "5mg" — the unit matches, the critical value doesn't.
#
# For this reason, numerical preservation requires an EXACT match (after
# light whitespace normalization only, e.g. "500 mg" == "500mg") of the
# full matched expression. This is a deliberate, documented deviation
# from the partial-match pattern used elsewhere, not an inconsistency.

import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.complexity.numerical_extractor import extract_numerical_expressions


def _normalize(expr):
    """Collapse internal whitespace so '500 mg' and '500mg' compare equal."""
    return re.sub(r'\s+', '', expr.lower())


def compute_numerical_preservation_rate(source, simplified):
    """
    Measures what fraction of source numerical expressions survive
    simplification with their exact value intact.

    Why exact match only (see module docstring for full rationale):
    a substituted or altered number is a factual/safety error, not an
    acceptable simplification, so no paraphrase map or partial-word
    credit is given here, unlike warning_preservation.py and
    entity_preservation.py.

    Formula: |numerical expressions preserved exactly| / |expressions in source|
    Score of 1.0 = all numbers preserved exactly (safe)
    Score of 0.0 = all numbers dropped or altered (safety failure)

    Args:
        source: original biomedical sentence
        simplified: system output after simplification

    Returns:
        dict: {score, preserved, dropped, total}
    """
    source_expressions = extract_numerical_expressions(source)

    if not source_expressions:
        return {
            'score': 1.0,  # No numerical expressions to preserve — trivially safe
            'preserved': [],
            'dropped': [],
            'total': 0
        }

    simplified_normalized = _normalize(simplified)
    preserved = []
    dropped = []

    for expr in source_expressions:
        expr_normalized = _normalize(expr['match'])
        if expr_normalized in simplified_normalized:
            preserved.append(expr)
        else:
            dropped.append(expr)

    score = len(preserved) / len(source_expressions)

    return {
        'score': round(score, 3),
        'preserved': preserved,
        'dropped': dropped,
        'total': len(source_expressions)
    }


def compute_corpus_numerical_preservation(pairs):
    """
    Computes average numerical preservation rate across a corpus.

    Args:
        pairs: list of (source, simplified) tuples

    Returns:
        dict: {mean_score, numerical_sentences, total_sentences, per_pair_scores}
    """
    scores = []
    numerical_count = 0

    for source, simplified in pairs:
        result = compute_numerical_preservation_rate(source, simplified)
        if result['total'] > 0:
            scores.append(result['score'])
            numerical_count += 1

    mean_score = round(sum(scores) / len(scores), 3) if scores else 1.0

    return {
        'mean_score': mean_score,
        'numerical_sentences': numerical_count,
        'total_sentences': len(pairs),
        'per_pair_scores': scores
    }


if __name__ == "__main__":
    test_pairs = [
        (
            "The drug showed an OR of 2.3 (95% CI 1.4-3.8, p<0.05).",
            "The drug showed an odds ratio of 2.3 (95% CI 1.4-3.8, p<0.05)."
            # All numbers preserved exactly; "OR" -> "odds ratio" is fine,
            # since we only check the NUMBERS, not surrounding wording
        ),
        (
            "Patients received 500mg twice daily for 12 weeks.",
            "Patients received 50mg twice daily for 12 weeks."
            # DANGEROUS: dosage altered (500mg -> 50mg) — must score as dropped,
            # not partially credited, since this is exactly the failure mode
            # this metric exists to catch
        ),
        (
            "Muscle cramps are a common problem.",
            "Muscle cramps are common."
            # No numerical expression — trivially safe
        ),
        (
            "The RR was 0.75 with 40 patients in each group.",
            "The risk was lower, based on a study of patients."
            # Both numbers dropped entirely during simplification
        ),
    ]

    print("=== Per-pair numerical preservation ===\n")
    for source, simplified in test_pairs:
        result = compute_numerical_preservation_rate(source, simplified)
        print(f"Source:     {source}")
        print(f"Simplified: {simplified}")
        print(f"Score: {result['score']} | Preserved: {[e['match'] for e in result['preserved']]} "
              f"| Dropped: {[e['match'] for e in result['dropped']]}")
        print()

    print("=== Corpus-level score ===")
    corpus = compute_corpus_numerical_preservation(test_pairs)
    print(f"Mean preservation rate: {corpus['mean_score']}")
    print(f"Numerical sentences: {corpus['numerical_sentences']}/{corpus['total_sentences']}")