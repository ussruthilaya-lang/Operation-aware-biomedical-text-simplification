import sys
sys.path.insert(0, '.')
import warnings

# Simulate spaCy being unavailable
import src.complexity.syntactic_depth as sd
sd.USE_SPACY = False

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        sd.flag_sentence("Test sentence.")
        print("ERROR: should have raised!")
    except RuntimeError as e:
        print(f"Correctly raised: {e}")