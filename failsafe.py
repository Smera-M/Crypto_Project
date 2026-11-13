"""
Gap 1 mitigation: a confidence-based fail-safe, evaluated honestly.

THE FAIL-SAFE
  When the classifier's top prediction is below High-strength (Low or Medium)
  and a rule says the result is untrustworthy, use the Critical tier's cipher
  distribution instead. The escalation target is the same distribution Critical
  events already use, so escalated events add no new observable cipher class.

TWO RULES (swept over a threshold)
  max_prob     : escalate when the top class probability is below the threshold
  severe_mass  : escalate when P(High) + P(Critical) is at or above the threshold

TWO ATTACKERS (both use the same query budget as attacks.py)
  label-only : picks the query that looks best by predicted label alone, unaware
               of the fail-safe
  adaptive   : knows the rule and threshold and the exact probabilities, and
               picks the query that does the most harm AFTER the fail-safe. This
               is the fair test; a defence that only beats a naive attacker is
               not much of a defence.

COST: over-escalation on clean test traffic, i.e. extra encryption strength
spent on events that did not need it (mostly Low traffic).

Run attacks.py first.   python src/failsafe.py [--policy NAME] [--attack structured|random]
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

import policy as shared
from policy import SEVERITIES
from attacks import RESULTS_DIR, TIER_MIN, load_cache, policy_matrices

ESCALATE_TO = SEVERITIES.index("Critical")
SEVERE_COLS = [SEVERITIES.index("High"), SEVERITIES.index("Critical")]
RULES = ("max_prob", "severe_mass")


def effective_tier(P, rule, thr):
    """Tier whose cipher distribution is used, given class probabilities P[..., 4]."""
    pred = P.argmax(-1)
    if rule == "none":
        return pred
    if rule == "max_prob":
        trigger = P.max(-1) < thr
    elif rule == "severe_mass":
        trigger = P[..., SEVERE_COLS].sum(-1) >= thr
    else:
        raise ValueError(f"unknown rule {rule}")
    below_severe = TIER_MIN[pred] < TIER_MIN[ESCALATE_TO]
    return np.where(trigger & below_severe, ESCALATE_TO, pred)


def evaluate(cache, mats, attack, rule, thr):
    V, L, S = mats
    rows = cache["att_rows"]
    true_all = cache["true_tier"].astype(int)
    true = true_all[rows]
    P_clean_all = cache["clean_proba"]
    P_clean = P_clean_all[rows]
    P_att = cache[f"proba_{attack}"]
    n = len(rows)

    # Cost on all clean test traffic
    eff_all = effective_tier(P_clean_all, rule, thr)
    over_levels = float(L[true_all, eff_all].mean())
    over_share = float(S[true_all, eff_all].mean())

    # No attack
    v_clean = V[true, eff_all[rows]]

    # Adaptive attacker: maximises harm after the fail-safe
    eff_q = effective_tier(P_att, rule, thr)
    adaptive = np.maximum(v_clean, V[true[:, None], eff_q].max(1))

    # Label-only attacker: chooses by predicted label, ignoring the fail-safe
    pred_q = P_att.argmax(-1)
    v0_q = V[true[:, None], pred_q]
    v0_clean = V[true, P_clean.argmax(-1)]
    q_star = v0_q.argmax(1)
    use_attack = v0_q.max(1) > v0_clean
    P_chosen = np.where(use_attack[:, None], P_att[np.arange(n), q_star], P_clean)
    label_only = V[true, effective_tier(P_chosen, rule, thr)]

    out = {
        "Over-escalation (levels)": over_levels,
        "Over-escalated share": over_share,
    }
    for gname, mask in (("severe", true >= 2), ("non-Low", true >= 1)):
        out[f"CDR clean ({gname})"] = float(v_clean[mask].mean())
        out[f"CDR label-only ({gname})"] = float(label_only[mask].mean())
        out[f"CDR adaptive ({gname})"] = float(adaptive[mask].mean())
    return out


def sweep(cache, policy_name, attack):
    mats = policy_matrices(policy_name)
    rows = []
    base = evaluate(cache, mats, attack, "none", 0.0)
    rows.append({"Rule": "none", "Threshold": np.nan, **base})

    grids = {
        "max_prob": list(np.round(np.linspace(0.0, 1.0, 101), 4)) + [1.0001],
        "severe_mass": [1.0001] + list(np.round(np.linspace(1.0, 0.0, 101), 4)),
    }
    for rule in RULES:
        for thr in grids[rule]:
            rows.append({"Rule": rule, "Threshold": thr,
                         **evaluate(cache, mats, attack, rule, thr)})
    df = pd.DataFrame(rows)
    df.insert(2, "Policy", policy_name)
    df.insert(3, "Attack", attack)
    return df


def operating_points(df, targets=(0.5, 0.75, 0.9)):
    base = df[df["Rule"] == "none"].iloc[0]["CDR adaptive (severe)"]
    rows = []
    if base <= 0:
        return pd.DataFrame(rows), base
    for target in targets:
        limit = (1 - target) * base
        for rule in RULES:
            sub = df[(df["Rule"] == rule) & (df["CDR adaptive (severe)"] <= limit + 1e-12)]
            if sub.empty:
                rows.append({"Target CDR reduction": target, "Rule": rule,
                             "Threshold": np.nan, "Reached": False})
                continue
            best = sub.sort_values(["Over-escalation (levels)", "Threshold"]).iloc[0]
            rows.append({
                "Target CDR reduction": target, "Rule": rule,
                "Threshold": best["Threshold"], "Reached": True,
                "CDR adaptive (severe)": best["CDR adaptive (severe)"],
                "CDR label-only (severe)": best["CDR label-only (severe)"],
                "Over-escalation (levels)": best["Over-escalation (levels)"],
                "Over-escalated share": best["Over-escalated share"],
            })
    return pd.DataFrame(rows), base


def plot(df, base_row, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for rule, color in zip(RULES, ("tab:blue", "tab:orange")):
        sub = df[df["Rule"] == rule].sort_values("Over-escalation (levels)")
        ax.plot(sub["Over-escalation (levels)"], sub["CDR adaptive (severe)"],
                "-", color=color, label=f"{rule}: adaptive attacker")
        ax.plot(sub["Over-escalation (levels)"], sub["CDR label-only (severe)"],
                "--", color=color, alpha=0.6, label=f"{rule}: label-only attacker")
    ax.scatter([base_row["Over-escalation (levels)"]],
               [base_row["CDR adaptive (severe)"]], c="k", zorder=3,
               label="no fail-safe (adaptive)")
    ax.set_xlabel("Over-escalation on clean traffic (mean strength levels)")
    ax.set_ylabel("Confidentiality downgrade rate, severe events")
    ax.set_title("Fail-safe trade-off")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved plot:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=shared.DEFAULT_POLICY,
                        choices=list(shared.POLICIES))
    parser.add_argument("--attack", default="structured",
                        choices=["structured", "random"])
    args = parser.parse_args()

    cache = load_cache()
    df = sweep(cache, args.policy, args.attack)
    df.round(6).to_csv(RESULTS_DIR / "failsafe_sweep.csv", index=False)

    points, base = operating_points(df)
    base_row = df[df["Rule"] == "none"].iloc[0]

    print(f"Policy: {args.policy}   Attack: {args.attack}")
    print("\n=== No fail-safe (baseline) ===")
    print(f"  CDR severe, no attack:            {base_row['CDR clean (severe)']:.4f}")
    print(f"  CDR severe, label-only attacker:  {base_row['CDR label-only (severe)']:.4f}")
    print(f"  CDR severe, adaptive attacker:    {base_row['CDR adaptive (severe)']:.4f}")
    print(f"  Over-escalation on clean traffic: {base_row['Over-escalation (levels)']:.4f} levels")

    if points.empty:
        print("\nBaseline CDR is 0 under this policy, so there is nothing to reduce.")
    else:
        points.round(6).to_csv(RESULTS_DIR / "failsafe_operating_points.csv", index=False)
        print("\n=== Cheapest threshold reaching each CDR reduction (adaptive attacker) ===")
        print(points.round(4).to_string(index=False))

    plot(df, base_row, RESULTS_DIR / "failsafe_tradeoff.png")
    print("\nSaved failsafe_sweep.csv, failsafe_operating_points.csv in", RESULTS_DIR)
    print("Note: the adaptive attacker is limited to the same query budget as attacks.py; "
          "a stronger attacker could do worse than this.")


if __name__ == "__main__":
    main()
