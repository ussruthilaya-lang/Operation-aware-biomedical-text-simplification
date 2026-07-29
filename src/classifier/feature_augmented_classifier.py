# Feature-Augmented TF-IDF Operation Classifier
#
# Track 2: Test if domain features help TF-IDF predict operations.
# Model 2: TF-IDF + UMLS jargon features (3 features)
# Model 3: TF-IDF + UMLS + warning + numerical features (all 9 features)
#
# Input  = SOURCE sentence + domain features
# Output = operation label

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.data.pseudo_labeler import (
    build_pairs_from_plaba_json,
    label_dataset,
    _DEFAULT_JSON,
    _DEFAULT_OUT,
    _VAL_CSV,
)

LABELS = ["Substitution", "Explanation", "Generalization"]

# Repo-root-relative paths
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FEATURES_CSV = os.path.join(_REPO_ROOT, "results", "classifier_features.csv")


def load_training_set(csv_path=_DEFAULT_OUT):
    """Load the pseudo-labeled training pairs."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    return pd.read_csv(csv_path)


def load_features(csv_path=_FEATURES_CSV):
    """Load precomputed features."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    return pd.read_csv(csv_path)


def build_val_set(json_path=_DEFAULT_JSON, val_csv=_VAL_CSV):
    """Build validation set from held-out PMIDs."""
    val_pmids = set(str(p) for p in pd.read_csv(val_csv)["pmid"].unique())
    df = build_pairs_from_plaba_json(json_path)
    df = df[df["pmid"].astype(str).isin(val_pmids)].reset_index(drop=True)
    return label_dataset(df)


def train_model_2():
    """Model 2: TF-IDF + UMLS jargon features"""
    print("=" * 80)
    print("MODEL 2: TF-IDF + UMLS Jargon Features (3 features)")
    print("=" * 80)

    print("\n=== Loading data ===")
    train_df = load_training_set()
    val_df = build_val_set()
    features_df = load_features()

    print(f"Train pairs: {len(train_df)}   Val pairs: {len(val_df)}")

    # Ensure consistent types for merging
    for df in [train_df, val_df, features_df]:
        df["pmid"] = df["pmid"].astype(str)
        df["sent_id"] = df["sent_id"].astype(str)

    # Feature columns for Model 2
    feature_cols_m2 = [
        "umls_jargon_present",
        "umls_jargon_count",
        "umls_jargon_density",
    ]

    # Select only join keys + feature columns from features_df to avoid duplicate columns
    features_subset = features_df[["pmid", "adaptation", "sent_id"] + feature_cols_m2]

    # Join features with training data
    train_merged = train_df.merge(features_subset, on=["pmid", "adaptation", "sent_id"], how="left")

    # Join features with validation data
    val_merged = val_df.merge(features_subset, on=["pmid", "adaptation", "sent_id"], how="left")

    print("\n=== Building features ===")
    # TF-IDF features
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, lowercase=True)
    X_tfidf_train = vectorizer.fit_transform(train_merged["source"])
    X_tfidf_val = vectorizer.transform(val_merged["source"])

    # Numeric features
    scaler = StandardScaler()
    X_numeric_train = scaler.fit_transform(train_merged[feature_cols_m2].fillna(0))
    X_numeric_val = scaler.transform(val_merged[feature_cols_m2].fillna(0))

    # Combine features
    from scipy.sparse import hstack
    X_train = hstack([X_tfidf_train, X_numeric_train])
    X_val = hstack([X_tfidf_val, X_numeric_val])

    print("\n=== Training Model 2 ===")
    classifier = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)
    classifier.fit(X_train, train_merged["operation"])

    print("\n=== Evaluation on validation set ===")
    y_pred = classifier.predict(X_val)

    accuracy = accuracy_score(val_merged["operation"], y_pred)
    macro_f1 = f1_score(val_merged["operation"], y_pred, average="macro", labels=LABELS)
    per_class_f1 = dict(
        zip(LABELS, f1_score(val_merged["operation"], y_pred, average=None, labels=LABELS))
    )
    report = classification_report(val_merged["operation"], y_pred, labels=LABELS, digits=3)
    cm = confusion_matrix(val_merged["operation"], y_pred, labels=LABELS)

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Macro-F1 : {macro_f1:.3f}")
    print(f"Comparison to TF-IDF baseline: {macro_f1:.3f} vs 0.424")
    print("\nPer-class F1:")
    for label, score in per_class_f1.items():
        print(f"  {label:15s} {score:.3f}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print("labels:", LABELS)
    print(cm)

    print("\n=== Saving checkpoint ===")
    import joblib
    os.makedirs("models/tfidf_umls_classifier", exist_ok=True)
    joblib.dump(classifier, "models/tfidf_umls_classifier/model_2.pkl")
    joblib.dump(vectorizer, "models/tfidf_umls_classifier/vectorizer.pkl")
    joblib.dump(scaler, "models/tfidf_umls_classifier/scaler.pkl")
    print("✓ Model 2 checkpoint saved to models/tfidf_umls_classifier/")

    return macro_f1


def train_model_3():
    """Model 3: TF-IDF + UMLS + warning + numerical features (all 9 features)"""
    print("\n\n")
    print("=" * 80)
    print("MODEL 3: TF-IDF + All Domain Features (9 features)")
    print("=" * 80)

    print("\n=== Loading data ===")
    train_df = load_training_set()
    val_df = build_val_set()
    features_df = load_features()

    print(f"Train pairs: {len(train_df)}   Val pairs: {len(val_df)}")

    # Ensure consistent types for merging
    for df in [train_df, val_df, features_df]:
        df["pmid"] = df["pmid"].astype(str)
        df["sent_id"] = df["sent_id"].astype(str)

    # Join features with training data
    train_merged = train_df.merge(
        features_df,
        on=["pmid", "adaptation", "sent_id"],
        how="left"
    )

    # Feature columns for Model 3 (all 9)
    feature_cols_m3 = [
        "umls_jargon_present",
        "umls_jargon_count",
        "umls_jargon_density",
        "warning_present",
        "warning_count",
        "numerical_present",
        "numerical_count",
        "numerical_density",
    ]

    # Select only join keys + feature columns from features_df to avoid duplicate columns
    features_subset = features_df[["pmid", "adaptation", "sent_id"] + feature_cols_m3]

    # Join features with training data
    train_merged = train_df.merge(features_subset, on=["pmid", "adaptation", "sent_id"], how="left")

    # Join features with validation data
    val_merged = val_df.merge(features_subset, on=["pmid", "adaptation", "sent_id"], how="left")

    print("\n=== Building features ===")
    # TF-IDF features
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, lowercase=True)
    X_tfidf_train = vectorizer.fit_transform(train_merged["source"])
    X_tfidf_val = vectorizer.transform(val_merged["source"])

    # Numeric features
    scaler = StandardScaler()
    X_numeric_train = scaler.fit_transform(train_merged[feature_cols_m3].fillna(0))
    X_numeric_val = scaler.transform(val_merged[feature_cols_m3].fillna(0))

    # Combine features
    from scipy.sparse import hstack
    X_train = hstack([X_tfidf_train, X_numeric_train])
    X_val = hstack([X_tfidf_val, X_numeric_val])

    print("\n=== Training Model 3 ===")
    classifier = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)
    classifier.fit(X_train, train_merged["operation"])

    print("\n=== Evaluation on validation set ===")
    y_pred = classifier.predict(X_val)

    accuracy = accuracy_score(val_merged["operation"], y_pred)
    macro_f1 = f1_score(val_merged["operation"], y_pred, average="macro", labels=LABELS)
    per_class_f1 = dict(
        zip(LABELS, f1_score(val_merged["operation"], y_pred, average=None, labels=LABELS))
    )
    report = classification_report(val_merged["operation"], y_pred, labels=LABELS, digits=3)
    cm = confusion_matrix(val_merged["operation"], y_pred, labels=LABELS)

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Macro-F1 : {macro_f1:.3f}")
    print(f"Comparison to TF-IDF baseline: {macro_f1:.3f} vs 0.424")
    print("\nPer-class F1:")
    for label, score in per_class_f1.items():
        print(f"  {label:15s} {score:.3f}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print("labels:", LABELS)
    print(cm)

    print("\n=== Saving checkpoint ===")
    import joblib
    os.makedirs("models/tfidf_all_features_classifier", exist_ok=True)
    joblib.dump(classifier, "models/tfidf_all_features_classifier/model_3.pkl")
    joblib.dump(vectorizer, "models/tfidf_all_features_classifier/vectorizer.pkl")
    joblib.dump(scaler, "models/tfidf_all_features_classifier/scaler.pkl")
    print("✓ Model 3 checkpoint saved to models/tfidf_all_features_classifier/")

    return macro_f1


if __name__ == "__main__":
    m2_f1 = train_model_2()
    m3_f1 = train_model_3()

    print("\n\n")
    print("=" * 80)
    print("TRACK 2 SUMMARY: Feature Integration Results")
    print("=" * 80)
    print(f"TF-IDF baseline:           0.424")
    print(f"Model 2 (TF-IDF + UMLS):   {m2_f1:.3f}  ({m2_f1 - 0.424:+.3f})")
    print(f"Model 3 (TF-IDF + all):    {m3_f1:.3f}  ({m3_f1 - 0.424:+.3f})")