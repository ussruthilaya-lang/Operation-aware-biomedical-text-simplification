# Why UMLS + QuickUMLS?
# UMLS (Unified Medical Language System) is the most comprehensive
# biomedical vocabulary — 3.5M+ medical concepts, 200+ source vocabularies
# (SNOMED CT, MeSH, ICD-10, RxNorm all unified).
# QuickUMLS maps free text to UMLS concept IDs in milliseconds using
# approximate string matching — no exact match needed.
#
# Why this matters for simplification:
# A term is "complex" if it exists in UMLS but NOT in common English.
# "Nephrotoxicity" → in UMLS, not common English → needs simplification.
# "Kidney damage" → in UMLS AND common English → acceptable in output.
# This detector flags the first case — terms that NEED simplification.
#
# Architecture note: QuickUMLS requires a local UMLS installation (~35GB).
# Interface is fully implemented; UMLS data integration for final submission.

import os

# UMLS_BACKEND selects the matching strategy without touching code:
#   morphological (default) — suffix/pattern heuristics, no external data
#   quickumls               — real UMLS lookup via a local QuickUMLS index
UMLS_BACKEND = os.environ.get("UMLS_BACKEND", "morphological").lower()
QUICKUMLS_INDEX_PATH = os.environ.get(
    "QUICKUMLS_INDEX_PATH", os.path.expanduser("~/umls_index")
)

# Common English vocabulary — terms that don't need simplification
# even if they appear in UMLS
COMMON_ENGLISH_TERMS = {
    'pain', 'fever', 'cough', 'cold', 'flu', 'cancer', 'diabetes',
    'heart', 'blood', 'brain', 'lung', 'liver', 'kidney', 'bone',
    'muscle', 'skin', 'eye', 'ear', 'nose', 'throat', 'stomach',
    'infection', 'inflammation', 'surgery', 'treatment', 'drug',
    'medicine', 'dose', 'patient', 'doctor', 'hospital', 'clinic',
    'symptoms', 'diagnosis', 'therapy', 'recovery', 'disease',
    'vitamin', 'protein', 'fat', 'sugar', 'salt', 'water', 'oxygen',
    # Added: general clinical/administrative vocabulary that happens to
    # exist as formal UMLS concepts but is already plain English —
    # false positives caught during QuickUMLS integration testing.
    'adverse effects', 'side effects', 'primary', 'performed',
    'secondary', 'chronic', 'acute',
}

# Biomedical jargon patterns — terms almost never in common English
# These approximate what QuickUMLS would flag via UMLS lookup
JARGON_SIGNALS = [
    # existing signals...
    'itis', 'emia', 'osis', 'ectomy', 'plasty', 'scopy',
    'graphy', 'pathy', 'toxicity', 'carcinoma', 'adenoma',
    'stenosis', 'infarction', 'hemorrhage',
    # Drug name suffixes — systematic pharmaceutical naming conventions
    'mab',   # monoclonal antibodies: rituximab, trastuzumab
    'nib',   # kinase inhibitors: sorafenib, imatinib
    'ide',   # many drug classes: furosemide, chloride
    'ine',   # alkaloids and many drugs: morphine, codeine
    'ol',    # beta blockers: metoprolol, atenolol
    'pril',  # ACE inhibitors: lisinopril, enalapril
    'sartan', # ARBs: losartan, valsartan
    'statin', # statins: atorvastatin, rosuvastatin
]

try:
    from quickumls import QuickUMLS
    QUICKUMLS_AVAILABLE = True
except ImportError:
    QUICKUMLS_AVAILABLE = False


def _quickumls_match(text, matcher):
    """Full UMLS lookup when QuickUMLS is available."""
    matches = matcher.match(text, best_match=True, ignore_syntax=False)
    seen_spans = {}
    for match_group in matches:
        # Each match_group is a list of candidate matches for one span;
        # keep only the highest-similarity candidate, not all of them.
        best = max(match_group, key=lambda m: m['similarity'])
        term = best['ngram']
        if term.lower() not in COMMON_ENGLISH_TERMS:
            span_key = (best['start'], best['end'])
            seen_spans[span_key] = {
                'term': term,
                'cui': best['cui'],
                'similarity': best['similarity'],
                'start': best['start'],
                'end': best['end'],
                'source': 'quickumls'
            }
    return list(seen_spans.values())


def _heuristic_match(text):
    """
    Fallback when QuickUMLS unavailable.
    Checks both individual words AND bigrams for biomedical signals.
    """
    results = []
    text_lower = text.lower()
    
    # Check individual words
    words = text_lower.split()
    for word in words:
        clean_word = word.strip('.,;:()')
        if clean_word in COMMON_ENGLISH_TERMS:
            continue
        for signal in JARGON_SIGNALS:
            if clean_word.endswith(signal) and len(clean_word) > len(signal) + 2:
                start = text_lower.find(clean_word)
                results.append({
                    'term': clean_word,
                    'cui': None,
                    'similarity': 1.0,
                    'start': start,
                    'end': start + len(clean_word),
                    'source': 'heuristic'
                })
                break
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
        'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are',
        'was', 'were', 'be', 'been', 'has', 'have', 'had',
        'received', 'performed', 'developed', 'showed', 'used'
    }

    # Check bigrams — catches "hepatocellular carcinoma", "renal failure"
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    for bigram in bigrams:
        clean = bigram.strip('.,;:()')
        first_word = clean.split()[0]
        # Skip if first word is a stopword
        if first_word in STOPWORDS:
            continue
        for signal in JARGON_SIGNALS:
            if clean.endswith(signal) and clean not in COMMON_ENGLISH_TERMS:
                start = text_lower.find(clean)
                results.append({
                    'term': clean,
                    'cui': None,
                    'similarity': 1.0,
                    'start': start,
                    'end': start + len(clean),
                    'source': 'heuristic'
                })
                break

    return results


class MorphologicalStrategy:
    """Default backend — suffix/pattern heuristics, no external data required."""

    def match(self, text):
        return _heuristic_match(text)


class QuickUMLSStrategy:
    """Real UMLS lookup via a local QuickUMLS index.

    Requires the `quickumls` package and a local UMLS installation
    (see QuickUMLS setup docs). Point `index_path` at wherever that
    index lives on the machine running the pipeline.
    """
    RELEVANT_SEMTYPES = {
        'T047',  # Disease or Syndrome
        'T191',  # Neoplastic Process
        'T037',  # Injury or Poisoning
        'T121',  # Pharmacologic Substance
        'T061',  # Therapeutic or Preventive Procedure
        'T060',  # Diagnostic Procedure
        'T184',  # Sign or Symptom
        'T046',  # Pathologic Function
    }

    def __init__(self, index_path=None):
        if not QUICKUMLS_AVAILABLE:
            raise ImportError(
                "quickumls is not installed. Install it or use UMLS_BACKEND=morphological."
            )
        self.index_path = index_path or QUICKUMLS_INDEX_PATH
        self.matcher = QuickUMLS(
            self.index_path,
            accepted_semtypes=self.RELEVANT_SEMTYPES,
            threshold=0.85  # stricter than default 0.7 — cuts weak/loose matches
        )

    def match(self, text):
        return _quickumls_match(text, self.matcher)


def get_strategy(backend=None):
    """Factory: returns the matching strategy selected by UMLS_BACKEND."""
    backend = (backend or UMLS_BACKEND).lower()
    if backend == "quickumls":
        return QuickUMLSStrategy()
    return MorphologicalStrategy()


def detect_umls_terms(text, matcher=None):
    """
    Detects medical jargon terms not in common English vocabulary.

    Why this matters: these are the exact terms that need Substitution
    operation — CHV lookup will find a plain-language equivalent.
    The classifier uses this signal to route spans to CHV substitution
    rather than LLM explanation.

    Args:
        text: input biomedical sentence
        matcher: QuickUMLS matcher instance (None = use heuristic fallback)

    Returns:
        list of dicts: [{term, cui, similarity, start, end, source}]
    """
    if QUICKUMLS_AVAILABLE and matcher is not None:
        return _quickumls_match(text, matcher)
    return _heuristic_match(text)


def flag_sentence(text, matcher=None):
    """Returns True if sentence contains UMLS medical jargon terms."""
    return len(detect_umls_terms(text, matcher)) > 0


def get_jargon_terms(text, matcher=None):
    """Returns just the matched terms — used by substitution routing."""
    return [r['term'] for r in detect_umls_terms(text, matcher)]


if __name__ == "__main__":
    test_sentences = [
        "Patients with hepatocellular carcinoma received sorafenib.",
        "Nephrotoxicity and hepatotoxicity were the primary adverse effects.",
        "Muscle cramps are a common problem.",
        "The patient developed deep vein thrombosis post-surgery.",
        "Angioplasty was performed to treat coronary stenosis.",
    ]

    strategy = get_strategy()
    print(f"UMLS_BACKEND: {UMLS_BACKEND}")
    print(f"QuickUMLS available: {QUICKUMLS_AVAILABLE}")
    print(f"Using: {type(strategy).__name__}\n")

    for s in test_sentences:
        terms = strategy.match(s)
        print(f"Sentence: {s}")
        print(f"Flagged: {len(terms) > 0}")
        print(f"Terms: {[t['term'] for t in terms]}")
        print()