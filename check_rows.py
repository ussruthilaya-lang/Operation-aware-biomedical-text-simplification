import pandas as pd

df = pd.read_csv('data/plaba/train.csv')
print(f'Total rows: {len(df)}')
print(f'Unique PMIDs: {df["pmid"].nunique()}')
print(f'Value counts of rows-per-PMID:')
print(df['pmid'].value_counts().value_counts().sort_index())