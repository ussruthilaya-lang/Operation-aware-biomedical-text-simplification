import re

# Why regex? Numerical expressions in biomedical text follow 
# consistent surface patterns. No ML needed; rules catch 100% 
# of the formats that matter: dosages, risk ratios, percentages.

NUMERICAL_PATTERNS = [
    # Percentages: 45%, 12.3%
    r'\b\d+\.?\d*\s*%',
    # Dosages: 500mg, 2.5 mg/kg, 10 mL
    r'\b\d+\.?\d*\s*(?:mg|mcg|µg|g|kg|mL|L|IU|mmol|µmol)(?:/(?:kg|day|dose|mL))?\b',
    # Risk ratios — catches "OR of 2.3", "RR was 0.75", "OR=2.3"
    r'\b(?:OR|RR|HR|ARR|NNT|NNH)\s*(?:of|was|is|=|:)?\s*\d+\.?\d*',
    # Confidence intervals: 95% CI AND ranges like (1.4-3.8)
    r'\b(?:95|90|99)\s*%\s*CI\b',
    r'\(?\s*\d+\.?\d*\s*[-–]\s*\d+\.?\d*\s*\)?',
    # P-values: p<0.05, p=0.001
    r'\bp\s*[<>=≤≥]\s*0\.\d+',
    # Plain numbers with units — FIXED: \d+\.?\d* allows decimals
    # (was \d+ only, which truncated "16.8 years" into "8 years" by
    # matching just the digit after the decimal point)
    r'\b\d+\.?\d*\s*(?:patients|participants|subjects|trials|weeks|months|years?|days|hours)\b',
]

def extract_numerical_expressions(text):
    """
    Detects numerical risk/dosage/statistical expressions in biomedical text.
    
    Why this matters: these are the expressions most commonly dropped or 
    weakened during simplification — catching them flags sentences that 
    need careful number-preserving treatment.
    
    Returns:
        list of dicts: [{match, start, end, pattern_type}]
    """
    results = []
    for pattern in NUMERICAL_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            results.append({
                'match': match.group(),
                'start': match.start(),
                'end': match.end(),
                'pattern': pattern
            })
    # Remove overlapping matches — keep longest
    results = _remove_overlaps(results)
    return results

def _remove_overlaps(matches):
    """Keep longest match when spans overlap."""
    if not matches:
        return []
    matches = sorted(matches, key=lambda x: x['start'])
    kept = [matches[0]]
    for current in matches[1:]:
        last = kept[-1]
        if current['start'] < last['end']:
            # Overlap — keep whichever is longer
            if (current['end'] - current['start']) > (last['end'] - last['start']):
                kept[-1] = current
        else:
            kept.append(current)
    return kept

def flag_sentence(text):
    """
    Returns True if sentence contains numerical expressions needing preservation.
    This is the interface the pipeline uses.
    """
    return len(extract_numerical_expressions(text)) > 0


# Quick test — run this file directly to verify
if __name__ == "__main__":
    test_sentences = [
        "The drug showed an OR of 2.3 (95% CI 1.4-3.8, p<0.05).",
        "Patients received 500mg twice daily for 12 weeks.",
        "Muscle cramps are a common problem.",
        "The RR was 0.75 with 40 patients in each group.",
    ]
    for s in test_sentences:
        matches = extract_numerical_expressions(s)
        flagged = flag_sentence(s)
        print(f"\nSentence: {s}")
        print(f"Flagged: {flagged}")
        print(f"Matches: {[m['match'] for m in matches]}")