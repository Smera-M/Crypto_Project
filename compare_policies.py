"""
The main results table: every registered policy on the SAME held-out samples,
under the same attacks and the same classifier, in one place.

Per policy:
  Leakage (system level)  I(true severity; cipher identifier) in bits, and the
      accuracy of a passive observer who sees only the identifier. Uses the
      classifier's ACTUAL predictions on the held-out test rows, so classifier
      errors are included. The prior is the held-out set's tier mix, which is a
      capped sample and not real traffic. Covers the cipher identifier only.
  CDR (severe events)     probability the cipher drawn is below the true tier's
      minimum strength: with no attack, and under the structured and random
      attacks with the full query budget.
  Over-escalation         extra strength spent on clean held-out traffic.
  Expected encrypt ms     from results/latency.csv, if benchmark_crypto.py was run.

Optional: --rule severe_mass --threshold 0.2 adds the same policy WITH the
fail-safe (adaptive attacker), so the improved system can be read against the
base paper's policy side by side.

  python src/compare_policies.py [--rule max_prob|severe_mass --threshold T]
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

import attacks as A
import failsafe as F
import leakage as LK
import policy as P

RESULTS_DIR = A.RESULTS_DIR
SEVERE = "Severe (High+Critical)"


def system_matrices(cache):
    """Return (tier prior, confusion rows C[s, t], predicted-tier distribution)."""
    true = cache["true_tier"].astype(int)
    pred = cache["clean_proba"].argmax(-1)
    counts = np.zeros((A.N_TIERS, A.N_TIERS))
    np.add.at(counts, (true, pred), 1)
    totals = counts.sum(1, keepdims=True)
    confusion = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    prior = counts.sum(1) / counts.sum()
    predicted = counts.sum(0) / counts.sum()
    return prior, confusion, predicted


def system_leakage(cache, policy_name):
    prior, confusion, _ = system_matrices(cache)
    matrix = LK.to_matrix(P.POLICIES[policy_name])      # [predicted tier, cipher]
    return LK.analyse(confusion @ matrix, prior)         # [true tier, cipher]


def expected_latency(cache, policy_name, size):
    path = RESULTS_DIR / "latency.csv"
    if not path.exists():
        return np.nan
    lat = pd.read_csv(path)
    lat = lat[lat["Message bytes"] == size].set_index("Cipher")["Encrypt median ms"]
    if lat.empty:
        return np.nan
    _, _, predicted = system_matrices(cache)
    matrix = LK.to_matrix(P.POLICIES[policy_name])
    per_cipher = np.array([lat.get(c, np.nan) for c in P.CIPHERS])
    return float((predicted @ matrix) @ per_cipher)


def build_table(cache, rule=None, threshold=None, latency_bytes=1024):
    _, cdr, _, K = A.analyse(cache)
    rows = []
    for name in P.POLICIES:
        leak = system_leakage(cache, name)

        def cdr_at(attack, budget):
            sel = cdr[(cdr.Attack == attack) & (cdr.Budget == budget)
                      & (cdr.Group == SEVERE) & (cdr.Policy == name)]
            return float(sel.CDR.iloc[0])

        mats = A.policy_matrices(name)
        clean = F.evaluate(cache, mats, "structured", "none", 0.0)
        row = {
            "Policy": name,
            "Deployed": name == P.DEFAULT_POLICY,
            "Leakage I(severity;cipher) bits": leak["I(Severity;Cipher) bits"],
            "Leakage fraction": leak["Leakage fraction"],
            "Observer accuracy": leak["Observer accuracy"],
            "Majority baseline": leak["Majority baseline"],
            "CDR severe, no attack": cdr_at("structured", 0),
            f"CDR severe, structured attack ({K} queries)": cdr_at("structured", K),
            f"CDR severe, random attack ({K} queries)": cdr_at("random", K),
            "Over-escalation (levels, clean)": clean["Over-escalation (levels)"],
            "Over-escalated share (clean)": clean["Over-escalated share"],
            f"Expected encrypt ms ({latency_bytes} B)":
                expected_latency(cache, name, latency_bytes),
        }
        if rule:
            guarded = F.evaluate(cache, mats, "structured", rule, threshold)
            row["With fail-safe: CDR adaptive attacker"] = guarded["CDR adaptive (severe)"]
            row["With fail-safe: CDR label-only attacker"] = guarded["CDR label-only (severe)"]
            row["With fail-safe: over-escalation (levels)"] = guarded["Over-escalation (levels)"]
        rows.append(row)
    return pd.DataFrame(rows), K


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", choices=list(F.RULES), default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--latency-bytes", type=int, default=1024)
    args = parser.parse_args()
    if bool(args.rule) != (args.threshold is not None):
        parser.error("--rule and --threshold must be given together")

    cache = A.load_cache()
    table, K = build_table(cache, args.rule, args.threshold, args.latency_bytes)
    table.round(6).to_csv(RESULTS_DIR / "policy_comparison.csv", index=False)

    prior, _, _ = system_matrices(cache)
    print("Held-out test set tier mix (the prior used for leakage): "
          + ", ".join(f"{t} {p:.3f}" for t, p in zip(P.SEVERITIES, prior)))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print()
    print(table.round(4).T.to_string(header=False))
    print(f"\nSaved {RESULTS_DIR / 'policy_comparison.csv'}")
    print("Leakage covers the cipher identifier only. The tier mix is a capped sample, "
          "not real traffic. CDR attacks are feature-space attacks.")


if __name__ == "__main__":
    main()
