import pandas as pd
df = pd.read_csv("results/classifier_features.csv")

keyword_pattern = "adverse|contraindicated|risk of|toxic|overdose"
suspects = df[
    df['source'].str.contains(keyword_pattern, case=False, na=False) &
    (df['warning_present'] == False)
]
print(f"Sentences with warning keywords but warning_present=False: {len(suspects)}")
for s in suspects['source'].head(10):
    print(f"  - {s[:100]}")