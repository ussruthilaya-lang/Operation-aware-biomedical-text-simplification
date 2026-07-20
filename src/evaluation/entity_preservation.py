# Why a custom metric instead of ROUGE/BLEU?
# Same rationale as warning_preservation.py: ROUGE measures n-gram overlap
# and doesn't care whether a specific named entity survived simplification.
# A system that drops "hepatocellular carcinoma" entirely can still score
# high ROUGE if the rest of the sentence is preserved. This metric checks
# entity-level survival directly — did the disease/chemical the source
# mentioned actually make it into the output in some recognizable form,
# or was it silently dropped (as opposed to substituted with a plain-
# English equivalent, which the Substitution operation is supposed to do)?
#
# Mirrors src/evaluation/warning_preservation.py's structure and
# conventions exactly, per team pattern for evaluation metrics.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.complexity.ner_detector import detect_biomedical_entities


def compute_entity_preservation_rate(source, simplified):
    """
    Measures what fraction of source biomedical entities survive
    simplification, in some recognizable form, in the output.

    Why this matters: entities are the primary targets for Substitution
    operations (see ner_detector.py docstring). A system that silently
    drops "nephrotoxicity" rather than substituting a plain-English
    equivalent ("kidney damage") has failed the operation, even if the
    rest of the sentence reads fluently. This metric distinguishes
    "substituted" from "dropped" the same way warning_preservation.py
    distinguishes "preserved" from "silently removed".

    NOTE — string-based matching, not CUI-based: this checks whether the
    entity's surface text (or a partial/substring match) appears in the
    output, not whether the *same UMLS concept* survived under a different
    surface form (e.g., "hepatocellular carcinoma" -> "liver cancer" would
    count as DROPPED here, not "substituted", since no shared substring
    exists). A CUI-based check via umls_matcher.py would catch true
    semantic substitutions and is a natural extension — documented here
    as a known limitation/future work, matching the paraphrase-map
    limitation already documented in warning_preservation.py, rather than
    built now, to keep this consistent with the team's existing metric
    pattern and ship a working version first.

    Formula: |entities preserved in output| / |entities in source|
    Score of 1.0 = all entities preserved (substituted or kept verbatim)
    Score of 0.0 = all entities dropped

    Args:
        source: original biomedical sentence
        simplified: system output after simplification

    Returns:
        dict: {score, preserved, dropped, total}
    """
    source_entities = detect_biomedical_entities(source)

    if not source_entities:
        return {
            'score': 1.0,  # No entities to preserve — trivially safe
            'preserved': [],
            'dropped': [],
            'total': 0
        }

    simplified_lower = simplified.lower()
    preserved = []
    dropped = []

    for entity in source_entities:
        entity_text = entity['text']
        entity_lower = entity_text.lower()

        # Exact match
        if entity_lower in simplified_lower:
            preserved.append(entity)
        # Partial word match — same convention as warning_preservation.py:
        # require words longer than 4 chars to avoid false "preserved"
        # credit from short, generic words (e.g., "acute", "with")
        elif any(word in simplified_lower for word in entity_lower.split()
                 if len(word) > 4):
            preserved.append(entity)
        else:
            dropped.append(entity)

    score = len(preserved) / len(source_entities)

    return {
        'score': round(score, 3),
        'preserved': preserved,
        'dropped': dropped,
        'total': len(source_entities)
    }


def compute_corpus_entity_preservation(pairs):
    """
    Computes average entity preservation rate across a corpus.

    Args:
        pairs: list of (source, simplified) tuples

    Returns:
        dict: {mean_score, entity_sentences, total_sentences, per_pair_scores}
    """
    scores = []
    entity_count = 0

    for source, simplified in pairs:
        result = compute_entity_preservation_rate(source, simplified)
        if result['total'] > 0:
            scores.append(result['score'])
            entity_count += 1

    mean_score = round(sum(scores) / len(scores), 3) if scores else 1.0

    return {
        'mean_score': mean_score,
        'entity_sentences': entity_count,
        'total_sentences': len(pairs),
        'per_pair_scores': scores
    }


if __name__ == "__main__":
    test_pairs = [
        (
            "Patients with hepatocellular carcinoma received sorafenib.",
            "Patients with liver cancer received sorafenib."
            # "hepatocellular carcinoma" dropped as exact/partial match
            # (would only count as preserved under a future CUI-based check)
        ),
        (
            "Nephrotoxicity and hepatotoxicity were the primary adverse effects.",
            "Kidney and liver damage were the main side effects."
            # Both entities dropped under string matching — realistic
            # example of why CUI-based matching is flagged as future work
        ),
        (
            "The patient developed deep vein thrombosis post-surgery.",
            "The patient developed deep vein thrombosis after surgery."
            # Entity preserved verbatim
        ),
        (
            "Muscle cramps are a common problem.",
            "Muscle cramps are common."
            # No biomedical entity detected — trivially safe
        ),
    ]

    print("=== Per-pair entity preservation ===\n")
    for source, simplified in test_pairs:
        result = compute_entity_preservation_rate(source, simplified)
        print(f"Source:     {source}")
        print(f"Simplified: {simplified}")
        print(f"Score: {result['score']} | Preserved: {[e['text'] for e in result['preserved']]} "
              f"| Dropped: {[e['text'] for e in result['dropped']]}")
        print()

    print("=== Corpus-level score ===")
    corpus = compute_corpus_entity_preservation(test_pairs)
    print(f"Mean preservation rate: {corpus['mean_score']}")
    print(f"Entity sentences: {corpus['entity_sentences']}/{corpus['total_sentences']}")