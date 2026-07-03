# Why a custom metric instead of ROUGE/BLEU?
# ROUGE measures n-gram overlap — it doesn't care if a warning phrase
# was dropped. A system that removes "contraindicated in patients with
# liver disease" can still score high ROUGE if the rest of the sentence
# is preserved. This metric catches exactly what ROUGE misses:
# safety-critical information loss.
# This is our paper's primary differentiating evaluation contribution.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.complexity.warning_lexicon import get_warning_phrases

def compute_warning_preservation_rate(source, simplified):
    """
    Measures what fraction of source warning phrases survive simplification.

    Why this matters: our central claim is that operation-aware simplification
    preserves safety-critical language better than direct LLM prompting.
    This metric is how we prove that claim quantitatively.

    Formula: |warnings preserved in output| / |warnings in source|
    Score of 1.0 = all warnings preserved (perfect safety)
    Score of 0.0 = all warnings dropped (safety failure)

    Args:
        source: original biomedical sentence
        simplified: system output after simplification

    Returns:
        dict: {score, preserved, dropped, total}
    """
    source_warnings = get_warning_phrases(source)

    if not source_warnings:
        return {
            'score': 1.0,  # No warnings to preserve — trivially safe
            'preserved': [],
            'dropped': [],
            'total': 0
        }

    simplified_lower = simplified.lower()
    preserved = []
    dropped = []

# Semantic paraphrase map — plain English equivalents of warning phrases
    PARAPHRASE_MAP = {
        'contraindicated': ['do not use', 'must not use', 'cannot be used', 'should not use'],
        'should avoid': ['do not', 'must not', 'avoid'],
        'increased risk': ['higher chance', 'more likely', 'greater risk'],
        'adverse effect': ['side effect', 'unwanted effect', 'harmful effect'],
        'adverse event': ['harmful side effect', 'harmful effect', 'bad reaction'],
        'adverse reaction': ['bad reaction', 'harmful reaction'],
    }

    for phrase in source_warnings:
        phrase_lower = phrase.lower()
        simplified_lower_check = simplified.lower()
        
        # Exact match
        if phrase_lower in simplified_lower_check:
            preserved.append(phrase)
        # Paraphrase match
        elif phrase_lower in PARAPHRASE_MAP:
            if any(p in simplified_lower_check for p in PARAPHRASE_MAP[phrase_lower]):
                preserved.append(phrase)
        # Partial word match
        elif any(word in simplified_lower_check for word in phrase_lower.split()
                 if len(word) > 4):
            preserved.append(phrase)
        else:
            dropped.append(phrase)
    score = len(preserved) / len(source_warnings)

    return {
        'score': round(score, 3),
        'preserved': preserved,
        'dropped': dropped,
        'total': len(source_warnings)
    }


def compute_corpus_warning_preservation(pairs):
    """
    Computes average warning preservation rate across a corpus.

    Args:
        pairs: list of (source, simplified) tuples

    Returns:
        dict: {mean_score, warning_sentences, total_sentences, per_pair_scores}
    """
    scores = []
    warning_count = 0

    for source, simplified in pairs:
        result = compute_warning_preservation_rate(source, simplified)
        if result['total'] > 0:
            scores.append(result['score'])
            warning_count += 1

    mean_score = round(sum(scores) / len(scores), 3) if scores else 1.0

    return {
        'mean_score': mean_score,
        'warning_sentences': warning_count,
        'total_sentences': len(pairs),
        'per_pair_scores': scores
    }


if __name__ == "__main__":
    test_pairs = [
        (
            "This drug is contraindicated in patients with liver disease.",
            "Do not use this drug if you have liver disease."
        ),
        (
            "There is an increased risk of bleeding when combined with aspirin.",
            "This drug works well for pain relief."  # Warning dropped!
        ),
        (
            "Patients should avoid taking this medication with grapefruit juice.",
            "Patients should avoid taking this medication with grapefruit juice."
        ),
        (
            "Muscle cramps are a common problem.",
            "Muscle cramps are common."  # No warning — trivially safe
        ),
    ]

    print("=== Per-pair warning preservation ===\n")
    for source, simplified in test_pairs:
        result = compute_warning_preservation_rate(source, simplified)
        print(f"Source:     {source}")
        print(f"Simplified: {simplified}")
        print(f"Score: {result['score']} | Preserved: {result['preserved']} | Dropped: {result['dropped']}")
        print()

    print("=== Corpus-level score ===")
    corpus = compute_corpus_warning_preservation(test_pairs)
    print(f"Mean preservation rate: {corpus['mean_score']}")
    print(f"Warning sentences: {corpus['warning_sentences']}/{corpus['total_sentences']}")