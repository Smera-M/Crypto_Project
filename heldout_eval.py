"""
Stress test: hold out whole capture files, train on the rest, evaluate on the
held-out files.

THIS IS NOT ATTACK-SESSION-BASED EVALUATION. The CIC-IDS2017 CSV files carry no
session identifiers or timestamps, so whole-capture hold-out is the closest
available test. It is also limited: most severity tiers occur in only one or two
captures, so some tiers will be MISSING from the held-out set (they cannot be
scored) or from the training set (they cannot be predicted). The script reports
which. Present this as a stress test of generalisation across captures, never as
the headline accuracy, and do not claim unseen-attack detection from it.

It also answers a second question: when the classifier misses a severe event on
unfamiliar traffic (a natural downgrade, with no adversary), would the
confidence-based fail-safe catch it, and at what cost on Low traffic?

  python src/heldout_eval.py                          # holds out PortScan and WebAttacks captures
  python src/heldout_eval.py --holdout Tuesday Friday # any substrings of file names

Outputs: heldout_metrics.csv, heldout_confusion.csv, heldout_failsafe.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

import attacks as A
import classifier as C
import failsafe as F
import policy as P

RESULTS_DIR = A.RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", nargs="+", default=["PortScan", "WebAttacks"],
                        help="substrings of capture file names to hold out")
    args = parser.parse_args()

    df = pd.read_csv(A.DATA_PATH, low_memory=False)
    if "SourceFile" not in df.columns:
        raise ValueError("processed_dataset.csv has no SourceFile column. "
                         "Re-run data.py (the version in this package).")

    patterns = [h.lower() for h in args.holdout]
    is_held = df["SourceFile"].str.lower().apply(lambda f: any(p in f for p in patterns))
    files = sorted(df["SourceFile"].unique())
    if not is_held.any():
        raise ValueError(f"No capture file matches {args.holdout}. Files: {files}")
    if is_held.all():
        raise ValueError("Every file is held out; nothing left to train on.")

    held_files = sorted(df.loc[is_held, "SourceFile"].unique())
    print("Held-out captures:", ", ".join(held_files))
    print("Training captures:", ", ".join(sorted(set(files) - set(held_files))))

    X = C.prepare_features(df)
    cols = list(X.columns)
    y = df["Severity"].map(A.TIER_INDEX).to_numpy()
    X_train, X_test = X[~is_held], X[is_held]
    y_train, y_test = y[~is_held.to_numpy()], y[is_held.to_numpy()]
    print(f"Train rows {len(X_train):,}   held-out rows {len(X_test):,}")

    model = C.build_model().fit(X_train, df.loc[~is_held, "Severity"])
    proba = A.tier_proba(model, X_test.to_numpy(dtype=np.float32), cols)
    pred = proba.argmax(-1)

    present = [i for i in range(A.N_TIERS) if (y_test == i).any()]
    in_train = [i for i in range(A.N_TIERS) if (y_train == i).any()]
    print("\nTiers present in the held-out set:", [P.SEVERITIES[i] for i in present])
    print("Tiers absent from the held-out set (NOT scored):",
          [P.SEVERITIES[i] for i in range(A.N_TIERS) if i not in present] or "none")
    print("Tiers absent from the training set (cannot be predicted):",
          [P.SEVERITIES[i] for i in range(A.N_TIERS) if i not in in_train] or "none")

    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, pred, labels=present, zero_division=0)
    metrics = pd.DataFrame({
        "Tier": [P.SEVERITIES[i] for i in present],
        "Held-out rows": support,
        "Training rows": [int((y_train == i).sum()) for i in present],
        "Precision": prec, "Recall": rec, "F1": f1,
    })
    summary = pd.DataFrame([{
        "Tier": "ALL (present tiers)", "Held-out rows": len(y_test),
        "Training rows": len(y_train), "Precision": np.nan,
        "Recall": accuracy_score(y_test, pred), "F1": float(np.mean(f1)),
    }])
    out = pd.concat([metrics, summary], ignore_index=True)
    out.round(6).to_csv(RESULTS_DIR / "heldout_metrics.csv", index=False)
    print("\nHeld-out metrics (macro-F1 averages ONLY the tiers present; "
          "'Recall' on the last row is overall accuracy):")
    print(out.round(4).to_string(index=False))

    cm = confusion_matrix(y_test, pred, labels=list(range(A.N_TIERS)))
    pd.DataFrame(cm, index=[f"Actual {t}" for t in P.SEVERITIES],
                 columns=[f"Predicted {t}" for t in P.SEVERITIES]).to_csv(
        RESULTS_DIR / "heldout_confusion.csv")

    # ---- natural downgrades and the fail-safe ----
    V, L, _ = A.policy_matrices(P.DEFAULT_POLICY)
    nonlow = y_test >= 1
    low = y_test == 0
    if not nonlow.any():
        print("\nNo Medium/High/Critical rows in the held-out set; fail-safe analysis skipped.")
        return

    natural = A.TIER_MIN[pred] < A.TIER_MIN[y_test]
    print(f"\nNatural strength downgrades on held-out non-Low rows "
          f"({int(nonlow.sum()):,} rows): {natural[nonlow].mean():.4f}")
    severe_mass = proba[:, A.N_TIERS - 2:].sum(1)
    for label, mask in (("naturally downgraded severe/medium rows", nonlow & natural),
                        ("correctly classified Low rows", low & (pred == 0))):
        if mask.any():
            print(f"  {label}: median top-class probability "
                  f"{np.median(proba[mask].max(1)):.3f}, median P(High)+P(Critical) "
                  f"{np.median(severe_mass[mask]):.3f}")

    rows = []
    grids = {"none": [0.0], "max_prob": [0.6, 0.7, 0.8, 0.9, 0.95],
             "severe_mass": [0.5, 0.3, 0.2, 0.1, 0.05]}
    for rule, thresholds in grids.items():
        for thr in thresholds:
            eff = F.effective_tier(proba, rule, thr)
            rows.append({
                "Rule": rule, "Threshold": np.nan if rule == "none" else thr,
                "CDR non-Low rows": float(V[y_test[nonlow], eff[nonlow]].mean()),
                "Over-escalation on Low rows (levels)":
                    float(L[y_test[low], eff[low]].mean()) if low.any() else np.nan,
            })
    fs = pd.DataFrame(rows)
    fs.round(6).to_csv(RESULTS_DIR / "heldout_failsafe.csv", index=False)
    print(f"\nFail-safe on held-out captures (policy: {P.DEFAULT_POLICY}):")
    print(fs.round(4).to_string(index=False))
    print("\nStress test only: one random-forest fit, one choice of held-out files.")


if __name__ == "__main__":
    main()
