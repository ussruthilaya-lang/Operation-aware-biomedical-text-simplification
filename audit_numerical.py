import sys, random
sys.path.insert(0, '.')
from src.evaluation.run_complexity_analysis import load_training_abstracts
from src.complexity.numerical_extractor import extract_numerical_expressions

test_sentences = [
    "RESULTS: The estimated BMD testing interval was 16.8 years (95% confidence interval [CI], 11.5 to 24.3 years).",
    "His 29 year old daughter also had seizures.",
    "His family doctor had prescribed 40 mg furosemide and 25 mg spironolactone.",
    "Multiple regression analysis revealed p<0.0001 as significant.",
]
for s in test_sentences:
    matches = extract_numerical_expressions(s)
    print(f"{s[:60]}...")
    print(f"  -> {[m['match'] for m in matches]}")
    print()
