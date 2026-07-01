# Why a handbuilt lexicon instead of ML?
# Warning phrases are finite, well-defined, and clinically established.
# A lexicon gives 100% recall on known phrases, zero hallucination,
# and full auditability — critical for healthcare NLP.
# This is also the foundation of our warning PRESERVATION metric later.

WARNING_CUES = {
    "contraindication": [
        "contraindicated", "contraindication", "must not be used",
        "should not be taken", "do not use", "not recommended"
    ],
    "risk_alert": [
        "risk of", "increased risk", "higher risk", "danger of",
        "potentially fatal", "life-threatening", "serious adverse",
        "severe side effect", "black box warning", "boxed warning"
    ],
    "avoidance": [
        "should avoid", "must avoid", "avoid taking", "avoid use",
        "refrain from", "do not take", "stop taking immediately",
        "discontinue if"
    ],
    "interaction": [
        "drug interaction", "interacts with", "do not combine",
        "dangerous combination", "may interact"
    ],
    "overdose": [
        "overdose", "toxic dose", "lethal dose", "maximum dose",
        "do not exceed"
    ],
    "side_effect": [
        "side effect", "adverse effect", "adverse event",
        "adverse reaction", "unwanted effect", "complication"
    ]
}

def detect_warnings(text):
    """
    Detects safety-critical warning language in biomedical text.

    Why this matters: LLM simplification routinely drops or softens
    warning language. This detector flags sentences that MUST preserve
    their warning content through simplification.

    Returns:
        list of dicts: [{match, category, start, end}]
    """
    text_lower = text.lower()
    results = []

    for category, phrases in WARNING_CUES.items():
        for phrase in phrases:
            start = 0
            while True:
                idx = text_lower.find(phrase, start)
                if idx == -1:
                    break
                results.append({
                    'match': text[idx:idx + len(phrase)],
                    'category': category,
                    'start': idx,
                    'end': idx + len(phrase)
                })
                start = idx + 1

    # Sort by position in text
    results = sorted(results, key=lambda x: x['start'])
    return results

def flag_sentence(text):
    """
    Returns True if sentence contains safety-critical warning language.
    This is the interface the pipeline uses.
    """
    return len(detect_warnings(text)) > 0

def get_warning_phrases(text):
    """
    Returns just the matched phrases — used by warning preservation metric.
    """
    return [r['match'] for r in detect_warnings(text)]


if __name__ == "__main__":
    test_sentences = [
        "This drug is contraindicated in patients with liver disease.",
        "There is an increased risk of bleeding when combined with aspirin.",
        "Patients should avoid taking this medication with grapefruit juice.",
        "The treatment showed improved outcomes over 12 weeks.",
        "Serious adverse effects include kidney failure and cardiac arrest.",
        "Do not exceed 4g of acetaminophen per day to avoid toxic dose."
    ]
    for s in test_sentences:
        warnings = detect_warnings(s)
        print(f"\nSentence: {s}")
        print(f"Flagged: {flag_sentence(s)}")
        print(f"Warnings: {[(w['match'], w['category']) for w in warnings]}")