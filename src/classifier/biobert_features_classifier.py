# BioBERT + domain features operation classifier.
#
# Track 1 (BioBERT alone, biobert_classifier.py) and Track 2 (features +
# TF-IDF only, feature_augmented_classifier.py) were never combined. This
# tests whether the engineered features that helped TF-IDF (+2.1-2.5%
# macro-F1) also help BioBERT, specifically targeting Generalization,
# BioBERT's weakest class (F1 0.365, recall 32% in the feature-less run).
#
# Runs TWO configs rather than assuming one, because the TF-IDF ablation
# gave a genuinely mixed signal on the numerical feature: alone, it hurt
# Generalization F1 (0.410->0.387), but BioBERT's contextual embeddings are
# a different kind of signal than TF-IDF's bag-of-words, so the same
# feature could behave differently here. Rather than guess, we run:
#   Model A: BioBERT + UMLS only (3 features)      -- the TF-IDF-safe bet
#   Model B: BioBERT + UMLS + Numerical (6 features) -- the riskier bet
# and report both, same ablation discipline used for every other result
# this session.
#
# Warning features are excluded from both configs, matching the TF-IDF
# finding that they contribute ~0 to classification (their value is in the
# safety-preservation metric, not routing).
#
# Saves to SEPARATE checkpoint directories from biobert_classifier.py's
# models/biobert_operation_classifier/ — that checkpoint is depended on by
# the operation-aware pipeline and must not be overwritten by this script.

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW

from src.data.pseudo_labeler import (
    build_pairs_from_plaba_json,
    label_dataset,
    _DEFAULT_JSON,
    _DEFAULT_OUT,
    _VAL_CSV,
)

LABELS = ["Substitution", "Explanation", "Generalization"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}

UMLS_COLS = ["umls_jargon_present", "umls_jargon_count", "umls_jargon_density"]
NUMERICAL_COLS = ["numerical_present", "numerical_count", "numerical_density"]

CONFIGS = {
    "biobert_umls_only": UMLS_COLS,
    "biobert_umls_numerical": UMLS_COLS + NUMERICAL_COLS,
}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_CSV = os.path.join(_REPO_ROOT, "results", "classifier_features.csv")
OUT_RESULTS_PATH = os.path.join(_REPO_ROOT, "results", "biobert_features_classifier.txt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 3
MAX_LENGTH = 128
BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.2"


class BioBERTWithFeatures(nn.Module):
    """BioBERT's pooled [CLS] representation concatenated with the
    engineered domain features, feeding a linear classification head."""

    def __init__(self, model_name, num_features, num_labels=3, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size + num_features, num_labels)

    def forward(self, input_ids, attention_mask, features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        combined = torch.cat([pooled, features], dim=1)
        return self.classifier(combined)

    def save(self, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.bert.save_pretrained(checkpoint_dir)
        torch.save(self.classifier.state_dict(), os.path.join(checkpoint_dir, "classifier_head.pt"))


class OperationFeatureDataset(Dataset):
    """Same as OperationDataset (biobert_classifier.py) plus a scaled
    feature vector per sentence."""

    def __init__(self, texts, labels, features, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts.reset_index(drop=True)
        self.labels = [LABEL2ID[label] for label in labels]
        self.features = features  # numpy array, already scaled
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long),
        }


def compute_class_weights(labels):
    unique, counts = np.unique(labels, return_counts=True)
    class_weights = len(labels) / (len(unique) * counts)
    return torch.tensor(class_weights, dtype=torch.float).to(DEVICE)


def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        features = batch['features'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, eval_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            features = batch['features'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", labels=[0, 1, 2]),
        "per_class_f1": dict(
            zip(LABELS, f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2]))
        ),
        "report": classification_report(all_labels, all_preds, labels=[0, 1, 2],
                                       target_names=LABELS, digits=3),
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]),
    }


def load_training_set(csv_path=_DEFAULT_OUT):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run `python -m src.data.pseudo_labeler` first.")
    return pd.read_csv(csv_path)


def build_val_set(json_path=_DEFAULT_JSON, val_csv=_VAL_CSV):
    val_pmids = set(str(p) for p in pd.read_csv(val_csv)["pmid"].unique())
    df = build_pairs_from_plaba_json(json_path)
    df = df[df["pmid"].astype(str).isin(val_pmids)].reset_index(drop=True)
    return label_dataset(df)


def attach_features(df, features_df):
    """Join classifier_features.csv onto df by pmid/adaptation/sent_id,
    same keys as feature_augmented_classifier.py."""
    df = df.copy()
    features_df = features_df.copy()
    for d in (df, features_df):
        d["pmid"] = d["pmid"].astype(str)
        d["sent_id"] = d["sent_id"].astype(str)
    all_feature_cols = UMLS_COLS + NUMERICAL_COLS
    subset = features_df[["pmid", "adaptation", "sent_id"] + all_feature_cols]
    merged = df.merge(subset, on=["pmid", "adaptation", "sent_id"], how="left")
    return merged


def run_config(config_name, feature_cols, train_df, val_df, tokenizer):
    print("\n" + "=" * 80)
    print(f"CONFIG: {config_name}  ({len(feature_cols)} features: {feature_cols})")
    print("=" * 80)

    checkpoint_dir = os.path.join(_REPO_ROOT, "models", config_name)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[feature_cols].fillna(0))
    val_features = scaler.transform(val_df[feature_cols].fillna(0))

    model = BioBERTWithFeatures(BIOBERT_MODEL, num_features=len(feature_cols)).to(DEVICE)

    train_dataset = OperationFeatureDataset(train_df["source"], train_df["operation"], train_features, tokenizer)
    val_dataset = OperationFeatureDataset(val_df["source"], val_df["operation"], val_features, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    train_labels = [LABEL2ID[label] for label in train_df["operation"]]
    class_weights = compute_class_weights(train_labels)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = 0
    best_results = None
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_results = evaluate(model, val_loader)

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val Accuracy : {val_results['accuracy']:.3f}")
        print(f"  Val Macro-F1 : {val_results['macro_f1']:.3f}")

        if val_results['macro_f1'] > best_val_f1:
            best_val_f1 = val_results['macro_f1']
            best_results = val_results
            model.save(checkpoint_dir)
            print(f"  New best val macro-F1 ({best_val_f1:.3f}) — saved checkpoint to {checkpoint_dir}")

    return best_results


if __name__ == "__main__":
    print("=== Loading data ===")
    train_df = load_training_set()
    val_df = build_val_set()
    features_df = pd.read_csv(FEATURES_CSV)
    print(f"Train pairs: {len(train_df)}   Val pairs: {len(val_df)}")

    train_df = attach_features(train_df, features_df)
    val_df = attach_features(val_df, features_df)

    tokenizer = BertTokenizer.from_pretrained(BIOBERT_MODEL)

    all_results = {}
    for config_name, feature_cols in CONFIGS.items():
        all_results[config_name] = run_config(config_name, feature_cols, train_df, val_df, tokenizer)

    print("\n\n" + "=" * 80)
    print("SUMMARY: BioBERT + Domain Features")
    print("=" * 80)
    print(f"BioBERT alone (reference):        macro-F1 0.461-0.465, Generalization F1 0.365")
    print(f"TF-IDF baseline (reference):      macro-F1 0.424")

    lines = [
        "BioBERT + Domain Features — Operation Classifier\n",
        "Reference: BioBERT alone macro-F1 0.461-0.465, Generalization F1 0.365\n",
        "Reference: TF-IDF baseline macro-F1 0.424\n\n",
    ]
    for config_name, results in all_results.items():
        summary_line = (
            f"{config_name:25s} macro-F1 {results['macro_f1']:.3f}  "
            f"(Sub {results['per_class_f1']['Substitution']:.3f} / "
            f"Exp {results['per_class_f1']['Explanation']:.3f} / "
            f"Gen {results['per_class_f1']['Generalization']:.3f})"
        )
        print(summary_line)
        lines.append(summary_line + "\n")
        lines.append("\n" + results["report"] + "\n")
        lines.append(f"Confusion matrix ({config_name}, rows=true, cols=pred):\n")
        lines.append(f"labels: {LABELS}\n")
        lines.append(str(results["confusion_matrix"]) + "\n\n")

    os.makedirs(os.path.dirname(OUT_RESULTS_PATH), exist_ok=True)
    with open(OUT_RESULTS_PATH, "w") as f:
        f.writelines(lines)
    print(f"\nWrote {OUT_RESULTS_PATH}")
