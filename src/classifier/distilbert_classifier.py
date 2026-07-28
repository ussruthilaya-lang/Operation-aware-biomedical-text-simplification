# BERT-base Operation Classifier (Model 2 in progression)
#
# Second rung of three-model progression: TF-IDF -> BERT-base -> BioBERT.
# Question it answers: does contextual pretraining (knowing word context) help
# predict operations better than bag-of-words (TF-IDF)?
#
# Input  = SOURCE sentence only (same as TF-IDF).
# Output = operation label.

import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
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
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 3
MAX_LENGTH = 128


class OperationDataset(Dataset):
    """PyTorch Dataset for operation classification."""
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = [LABEL2ID[label] for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx]) if hasattr(self.texts, 'iloc') else str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def compute_class_weights(labels):
    """Compute class weights for imbalanced dataset."""
    unique, counts = np.unique(labels, return_counts=True)
    class_weights = len(labels) / (len(unique) * counts)
    return torch.tensor(class_weights, dtype=torch.float).to(DEVICE)


def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, eval_loader):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
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
        "predictions": all_preds,
        "labels": all_labels,
    }


def load_training_set(csv_path=_DEFAULT_OUT):
    """Load the pseudo-labeled training pairs."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python -m src.data.pseudo_labeler` first."
        )
    return pd.read_csv(csv_path)


def build_val_set(json_path=_DEFAULT_JSON, val_csv=_VAL_CSV):
    """Build validation set from held-out PMIDs."""
    val_pmids = set(str(p) for p in pd.read_csv(val_csv)["pmid"].unique())
    df = build_pairs_from_plaba_json(json_path)
    df = df[df["pmid"].astype(str).isin(val_pmids)].reset_index(drop=True)
    return label_dataset(df)


if __name__ == "__main__":
    print("=== Loading data ===")
    train_df = load_training_set()
    val_df = build_val_set()
    print(f"Train pairs: {len(train_df)}   Val pairs: {len(val_df)}")

    print("\n=== Loading BERT tokenizer and model ===")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3
    ).to(DEVICE)

    print("=== Creating datasets and dataloaders ===")
    train_dataset = OperationDataset(train_df["source"], train_df["operation"], tokenizer)
    val_dataset = OperationDataset(val_df["source"], val_df["operation"], tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print("=== Computing class weights ===")
    train_labels = [LABEL2ID[label] for label in train_df["operation"]]
    class_weights = compute_class_weights(train_labels)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print("=== Training BERT-base ===")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = 0
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_results = evaluate(model, val_loader)

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val Accuracy : {val_results['accuracy']:.3f}")
        print(f"  Val Macro-F1 : {val_results['macro_f1']:.3f}")

        if val_results['macro_f1'] > best_val_f1:
            best_val_f1 = val_results['macro_f1']

    print("\n=== Final Evaluation on validation set ===")
    val_results = evaluate(model, val_loader)
    print(f"Accuracy : {val_results['accuracy']:.3f}")
    print(f"Macro-F1 : {val_results['macro_f1']:.3f}")
    print(f"\nComparison to TF-IDF baseline: {val_results['macro_f1']:.3f} vs 0.424")

    print("\nPer-class F1:")
    for label, score in val_results["per_class_f1"].items():
        print(f"  {label:15s} {score:.3f}")

    print("\nClassification report:")
    print(val_results["report"])
    print("Confusion matrix (rows=true, cols=pred):")
    print("labels:", LABELS)
    print(val_results["confusion_matrix"])
