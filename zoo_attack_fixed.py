from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODEL_PATH = RESULTS / "random_forest.joblib"
DATA_PATH = RESULTS / "processed_dataset.csv"

SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]


def severity_index(label):
    label = str(label).strip().title()
    return SEVERITY_ORDER.index(label)


def get_probabilities(model, X):
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    # Convert probabilities to Low, Medium, High, Critical order.
    ordered = np.zeros((len(X), len(SEVERITY_ORDER)))

    for i, severity in enumerate(SEVERITY_ORDER):
        if severity in classes:
            ordered[:, i] = probabilities[:, classes.index(severity)]

    return ordered


def attack_one(model, row, mutable_indices, query_budget, feature_columns):
    """
    Bounded score-based black-box coordinate search.
    Each feature is limited to +/- 10% of its original value.
    """
    original = row.copy().astype(float)
    current = original.copy()

    queries = 0
    step_size = 0.02
    max_change = 0.10

    def objective(candidate):
        nonlocal queries
        queries += 1

        candidate_df = pd.DataFrame(
            [candidate],
            columns=feature_columns
        )

        probs = get_probabilities(model, candidate_df)[0]

        # Lower expected severity is the attack's goal.
        score = np.dot(
            probs,
            np.arange(len(SEVERITY_ORDER))
        )

        return score, probs

    current_score, current_probs = objective(current)

   # Bound each feature's changes relative to its original value.
    lower_bounds = original - (np.abs(original) * max_change)
    upper_bounds = original + (np.abs(original) * max_change)

# Keep original zero-valued features fixed for this experiment.
    zero_features = (original == 0)
    lower_bounds[zero_features] = original[zero_features]
    upper_bounds[zero_features] = original[zero_features]

# Preserve negative values when they exist in the original data.

    while queries < query_budget:
        improved = False

        for feature_idx in np.random.permutation(mutable_indices):
            if queries >= query_budget:
                break

            value = current[feature_idx]
            delta = max(abs(original[feature_idx]) * step_size, 0.01)

            # Try increasing the feature within its bounds.
            candidate_up = current.copy()
            candidate_up[feature_idx] = min(
                value + delta,
                upper_bounds[feature_idx]
            )

            if candidate_up[feature_idx] != value:
                score_up, probs_up = objective(candidate_up)

                if score_up < current_score:
                    current = candidate_up
                    current_score = score_up
                    current_probs = probs_up
                    improved = True

            if queries >= query_budget:
                break

            # Try decreasing the feature within its bounds.
            candidate_down = current.copy()
            candidate_down[feature_idx] = max(
                value - delta,
                lower_bounds[feature_idx]
            )

            if candidate_down[feature_idx] != value:
                score_down, probs_down = objective(candidate_down)

                if score_down < current_score:
                    current = candidate_down
                    current_score = score_down
                    current_probs = probs_down
                    improved = True

        if not improved:
            break

    return current, current_probs, queries



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--query-budget", type=int, default=100)
    args = parser.parse_args()

    print("Loading saved model...")
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_columns = saved["feature_columns"]
    test_index = saved["test_index"]

    print("Loading processed dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df[feature_columns]
    y = df["Severity"].astype(str).str.strip().str.title()

    # Use the exact held-out test rows from classifier.py.
    X_test = X.loc[test_index].copy()
    y_test = y.loc[test_index].copy()

    # Focus on High and Critical samples for severity-lowering attacks.
    selected = y_test.isin(["High", "Critical"])
    X_attack = X_test.loc[selected]
    y_attack = y_test.loc[selected]

    if len(X_attack) > args.max_samples:
        X_attack = X_attack.sample(
            n=args.max_samples,
            random_state=42
        )
        y_attack = y_attack.loc[X_attack.index]

    constant_features = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate"
]

    print("Constant features found:", [
    feature for feature in X_attack.columns
    if feature in constant_features
])

    mutable_indices = [
    i for i, feature in enumerate(X_attack.columns)
    if feature not in constant_features
]

    print(f"Selected samples: {len(X_attack)}")
    print(f"Mutable features: {len(mutable_indices)}")
    print(f"Query budget per sample: {args.query_budget}")

    results = []

    for count, (index, row) in enumerate(
        X_attack.iterrows(),
        start=1
    ):
        original = row.to_numpy(dtype=float)

        adversarial, adversarial_probs, queries = attack_one(
            model,
            original,
            mutable_indices,
            args.query_budget,
            feature_columns
        )

        original_df = pd.DataFrame([original], columns=feature_columns)
        original_prediction = model.predict(original_df)[0]
        original_probs = get_probabilities(model, original_df)[0]

        print("Original prediction:", original_prediction)
        print("Original probabilities:", original_probs)
        print("Adversarial probabilities:", adversarial_probs)
        adversarial_prediction = SEVERITY_ORDER[
            int(np.argmax(adversarial_probs))
        ]

        original_tier = severity_index(y_test.loc[index])
        adversarial_tier = severity_index(adversarial_prediction)

        nonzero_mask = original != 0

        if np.any(nonzero_mask):
            max_relative_change = np.max(
                np.abs(adversarial[nonzero_mask] - original[nonzero_mask])
                / np.abs(original[nonzero_mask])
            )
        else:
            max_relative_change = 0.0

        results.append({
            "index": index,
            "true_severity": y_test.loc[index],
            "original_prediction": original_prediction,
            "adversarial_prediction": adversarial_prediction,
            "tier_lowered": adversarial_tier < original_tier,
            "critical_to_low": (
                y_test.loc[index] == "Critical"
                and adversarial_prediction == "Low"
            ),
            "queries": queries,
            "perturbation_l2": float(
                np.linalg.norm(adversarial - original)
            ),
            "max_relative_change": float(max_relative_change),
        })

        if count % 50 == 0 or count == len(X_attack):
            print(f"Processed {count}/{len(X_attack)} samples")
            print("Rows currently stored:", len(results))

    results_df = pd.DataFrame(results)
    output_path = RESULTS / "zoo_attack_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nZOO experiment complete.")
    print(f"Results saved to: {output_path}")
    print(
        "Tier-lowering rate:",
        round(results_df["tier_lowered"].mean(), 4)
    )
    print(
        "Critical-to-Low rate:",
        round(results_df["critical_to_low"].mean(), 4)
    )
    print(
        "Average queries:",
        round(results_df["queries"].mean(), 2)
    )
    print(
        "Average L2 perturbation:",
        round(results_df["perturbation_l2"].mean(), 4)
    )
if __name__ == "__main__":
    main()