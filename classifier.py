"""
Step: train the severity-tier Random Forest.

Same model and split as before (80:20 stratified, random_state=42). Additions:
  - saves the test-row index inside the model file, so attacks.py evaluates on
    exactly the same held-out rows;
  - reports macro-F1 (accuracy alone hides weak classes);
  - optional stratified 5-fold cross-validation:  python src/classifier.py --cv

Evaluation is a random split of flow records. Flows from the same attack
session can fall in both train and test sets, which can inflate accuracy.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "results" / "processed_dataset.csv"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = RESULTS_DIR / "random_forest.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TIERS = ["Low", "Medium", "High", "Critical"]
NON_FEATURE_COLUMNS = ["Label", "Severity", "SourceFile"]


def prepare_features(df, columns=None):
    """Numeric feature matrix. With `columns`, select exactly those columns."""
    if columns is None:
        X = df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])
    else:
        X = df[columns]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    if columns is None:
        X = X.dropna(axis=1, how="all")
    return X


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true",
                        help="also run stratified 5-fold cross-validation (slower)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Run data.py first.")

    print("Loading processed dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.dropna(subset=["Severity"])

    y = df["Severity"]
    X = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Features:", X.shape[1])

    model = build_model()
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, labels=TIERS,
                        average="macro", zero_division=0)

    report = classification_report(
        y_test, predictions, labels=TIERS, output_dict=True, zero_division=0)

    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Test macro-F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=TIERS, zero_division=0))

    cm = confusion_matrix(y_test, predictions, labels=TIERS)
    pd.DataFrame(
        cm,
        index=[f"Actual {t}" for t in TIERS],
        columns=[f"Predicted {t}" for t in TIERS],
    ).to_csv(RESULTS_DIR / "confusion_matrix.csv")

    summary = {"accuracy": accuracy, "macro_f1": macro_f1, "report": report}

    if args.cv:
        print("Running stratified 5-fold cross-validation (macro-F1)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(build_model(), X, y, cv=cv, scoring="f1_macro")
        summary["cv_macro_f1_mean"] = float(scores.mean())
        summary["cv_macro_f1_std"] = float(scores.std())
        print(f"CV macro-F1: {scores.mean():.4f} +/- {scores.std():.4f}")

    with open(RESULTS_DIR / "classification_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    joblib.dump(
        {
            "model": model,
            "feature_columns": X.columns.tolist(),
            "test_index": X_test.index.tolist(),
        },
        MODEL_PATH,
    )

    print("\nSaved model:", MODEL_PATH)
    print("Saved confusion matrix and classification report.")


if __name__ == "__main__":
    main()
