# Why syntactic depth?
# Biomedical sentences are structurally complex — deeply nested 
# subordinate clauses, long noun phrases, passive constructions.
# A sentence can have simple vocabulary but be unreadable due to 
# structure alone. Syntactic depth catches this case that 
# lexical detectors miss entirely.
# We use dependency parse tree depth — the longest path from 
# root to any leaf node. Higher depth = more complex structure.

import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    import spacy
    nlp_dep = spacy.load('en_core_web_sm')
    USE_SPACY = True
except Exception:
    USE_SPACY = False

def get_dependency_depth(text):
    """
    Computes maximum dependency tree depth.
    
    Why dependency depth over sentence length?
    Length penalizes long but simple sentences unfairly.
    Depth captures genuine structural complexity — a 10-word 
    sentence with 4 levels of nesting is harder than a 
    20-word list of simple facts.

    Returns:
        int: max depth of dependency parse tree
    """
    if not USE_SPACY:
        # Fallback: approximate depth by clause count
        return _approximate_depth(text)
    
    doc = nlp_dep(text)
    
    def token_depth(token):
        depth = 0
        current = token
        while current.head != current:
            current = current.head
            depth += 1
        return depth
    
    depths = [token_depth(token) for token in doc]
    return max(depths) if depths else 0

def _approximate_depth(text):
    """
    Fallback when spacy unavailable.
    Approximates complexity by counting subordinating conjunctions
    and punctuation as structural complexity signals.
    """
    complexity_signals = [
        'which', 'that', 'although', 'whereas', 'whereby',
        'wherein', 'however', 'nevertheless', 'furthermore',
        ';', ','
    ]
    text_lower = text.lower()
    count = sum(text_lower.count(signal) for signal in complexity_signals)
    return count

DEPTH_THRESHOLD = 7  # Tuned on PLABA — sentences above this need structural simplification

def flag_sentence(text):
    """
    Returns True if sentence has deep syntactic structure.
    Threshold of 7 calibrated to catch biomedical run-on sentences
    while ignoring simple compound sentences.
    """
    return get_dependency_depth(text) > DEPTH_THRESHOLD

def get_depth_score(text):
    """Returns raw depth score for EDA analysis."""
    return get_dependency_depth(text)


if __name__ == "__main__":
    test_sentences = [
        # Simple
        "Muscle cramps are a common problem.",
        # Medium
        "Patients received 500mg twice daily for 12 weeks.",
        # Complex — nested clauses
        "Although the neuromuscular hypothesis, which suggests that altered neuromuscular control due to muscle fatigue is the primary mechanism, has gained traction, the dehydration hypothesis remains contested.",
        # Very complex — typical biomedical abstract sentence
        "The aim of this review is to examine the recent literature, in terms of physiological mechanisms of EAMC, which has been associated with electrolyte imbalance, peripheral fatigue, and altered spinal reflex activity.",
    ]
    for s in test_sentences:
        depth = get_depth_score(s)
        flagged = flag_sentence(s)
        print(f"\nSentence: {s[:80]}...")
        print(f"Depth: {depth} | Flagged: {flagged}")