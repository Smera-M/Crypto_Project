"""
Gap 1: adversarial evasion as a confidentiality-downgrade attack.

THREAT MODEL (it determines what the numbers mean)
  - Black-box attacker. They can submit crafted flows and see the severity tier
    the classifier assigns (label only). They cannot see the model.
  - They CANNOT change the destination port, victim-side (backward-direction)
    features, flag counts or packet counts.
  - They CAN change how their own traffic is paced (time scale `alpha`: below 1
    is faster, above 1 is slower) and how large their packets are (padding
    factor `beta`, at least 1, since adding bytes is easier than removing them).
  - Three kinds of success are reported SEPARATELY, because High and Critical
    share the same minimum cipher strength, so a tier change is not always a
    strength downgrade:
      misclassified      : predicted tier differs from the true tier
      tier lowered       : predicted tier is below the true tier
      strength downgrade : the predicted tier's minimum cipher strength is below
                           the true tier's (a true Critical predicted Low or
                           Medium; a true Critical predicted High is NOT one)
    The downgrade rate (CDR) then measures the cipher actually drawn: the
    probability it falls below the true tier's minimum strength.

TWO ATTACKS, SAME QUERY BUDGET
  structured : grid over (alpha, beta); one coherent pacing/padding change is
               applied to every related feature at once (consistent flows)
  random     : independent random scaling of each mutable feature (baseline)

LIMITS: a feature-space attack. Nothing guarantees a real packet sequence would
produce exactly these feature values, so results estimate vulnerability; they do
not prove an attack is deployable.

OUTPUTS (results/): attack_asr.csv, attack_cdr.csv, attack_by_label.csv and
attack_cache.npz (per-query class probabilities, reused by failsafe.py).

  python src/attacks.py                      # full run
  python src/attacks.py --max-samples 3000   # quick run
"""

from pathlib import Path
import argparse
import time

import joblib
import numpy as np
import pandas as pd

import policy as shared
from policy import MIN_STRENGTH, SEVERITIES, STRENGTH
from classifier import prepare_features

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DATA_PATH = RESULTS_DIR / "processed_dataset.csv"
MODEL_PATH = RESULTS_DIR / "random_forest.joblib"
CACHE_PATH = RESULTS_DIR / "attack_cache.npz"

N_TIERS = len(SEVERITIES)
TIER_INDEX = {t: i for i, t in enumerate(SEVERITIES)}
TIER_MIN = np.array([MIN_STRENGTH[t] for t in SEVERITIES])

ALPHAS = np.geomspace(0.25, 8.0, 11)   # includes 1.0
BETAS = np.geomspace(1.0, 4.0, 5)      # padding only
NOISE_FACTOR = 2.0                     # random attack: each feature x [1/2, 2]

# Mutable features, grouped by how they move together (names as in data.py,
# i.e. with whitespace stripped). Missing columns are skipped and reported.
FEATURE_GROUPS = {
    "time": [   # scale by alpha
        "Flow Duration", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
        "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
        "Fwd IAT Max", "Fwd IAT Min", "Active Mean", "Active Std",
        "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max",
        "Idle Min",
    ],
    "pkt_rate": ["Flow Packets/s", "Fwd Packets/s"],          # scale by 1/alpha
    "byte_rate": ["Flow Bytes/s"],                            # scale by beta/alpha
    "size": [   # scale by beta
        "Fwd Packet Length Max", "Fwd Packet Length Min",
        "Fwd Packet Length Mean", "Fwd Packet Length Std", "Min Packet Length",
        "Max Packet Length", "Packet Length Mean", "Packet Length Std",
        "Average Packet Size", "Avg Fwd Segment Size",
    ],
    "size_sq": ["Packet Length Variance"],                    # scale by beta^2
    "bytes_total": ["Total Length of Fwd Packets", "Subflow Fwd Bytes"],  # beta
}


# --------------------------------------------------------------------------
# Perturbations
# --------------------------------------------------------------------------

def build_groups(feature_columns):
    groups, missing = {}, []
    for name, features in FEATURE_GROUPS.items():
        present = [f for f in features if f in feature_columns]
        missing += [f for f in features if f not in feature_columns]
        groups[name] = np.array(
            [feature_columns.index(f) for f in present], dtype=int)
    return groups, missing


def structured_queries(seed):
    pairs = [(a, b) for a in ALPHAS for b in BETAS
             if not (np.isclose(a, 1.0) and np.isclose(b, 1.0))]
    order = np.random.default_rng(seed).permutation(len(pairs))
    return np.array([pairs[i] for i in order])


def perturb_structured(Xc, queries, groups):
    """(n, F) -> (n, K, F): each query applies one coherent pacing/padding."""
    n, _ = Xc.shape
    V = np.repeat(Xc[:, None, :], len(queries), axis=1).astype(np.float32)
    for k, (a, b) in enumerate(queries):
        V[:, k, groups["time"]] *= a
        V[:, k, groups["pkt_rate"]] /= a
        V[:, k, groups["byte_rate"]] *= b / a
        V[:, k, groups["size"]] *= b
        V[:, k, groups["size_sq"]] *= b * b
        V[:, k, groups["bytes_total"]] *= b
    return V


def perturb_random(Xc, K, mutable, rng):
    n = Xc.shape[0]
    V = np.repeat(Xc[:, None, :], K, axis=1).astype(np.float32)
    log_f = np.log(NOISE_FACTOR)
    noise = np.exp(rng.uniform(-log_f, log_f, size=(n, K, len(mutable))))
    V[:, :, mutable] *= noise.astype(np.float32)
    return V


def tier_proba(model, X2d, feature_columns):
    """Class probabilities as an (n, 4) array in SEVERITIES order."""
    raw = model.predict_proba(pd.DataFrame(X2d, columns=feature_columns))
    out = np.zeros((raw.shape[0], N_TIERS), dtype=np.float32)
    for j, cls in enumerate(model.classes_):
        out[:, TIER_INDEX[cls]] = raw[:, j]
    return out


def run_attack(model, X_att, feature_columns, kind, queries, groups, mutable,
               K, chunk, rng):
    n = len(X_att)
    out = np.zeros((n, K, N_TIERS), dtype=np.float32)
    for start in range(0, n, chunk):
        Xc = X_att[start:start + chunk]
        if kind == "structured":
            V = perturb_structured(Xc, queries, groups)
        else:
            V = perturb_random(Xc, K, mutable, rng)
        P = tier_proba(model, V.reshape(-1, V.shape[-1]), feature_columns)
        out[start:start + len(Xc)] = P.reshape(len(Xc), K, N_TIERS)
        done = min(start + chunk, n)
        print(f"    {kind}: {done:,}/{n:,} samples", end="\r")
    print()
    return out


# --------------------------------------------------------------------------
# Policy-dependent harm: what the downgrade does to the cipher actually used
# --------------------------------------------------------------------------

def policy_matrices(policy_name):
    """Return (V, L, S), each indexed [true tier, effective tier].

    V: probability the cipher drawn is below the true tier's minimum strength
       (a confidentiality downgrade)
    L: expected strength levels ABOVE the true tier's minimum (over-escalation)
    S: probability the cipher is stronger than the true tier needs
    "Effective tier" is the tier whose cipher distribution is actually used.
    """
    pol = shared.POLICIES[policy_name]
    V = np.zeros((N_TIERS, N_TIERS))
    L = np.zeros((N_TIERS, N_TIERS))
    S = np.zeros((N_TIERS, N_TIERS))
    for s, true_tier in enumerate(SEVERITIES):
        for e, eff_tier in enumerate(SEVERITIES):
            for cipher, p in pol[eff_tier].items():
                excess = STRENGTH[cipher] - MIN_STRENGTH[true_tier]
                if excess < 0:
                    V[s, e] += p
                elif excess > 0:
                    L[s, e] += p * excess
                    S[s, e] += p
    return V, L, S


def group_masks(true_tier):
    return {
        "Medium": true_tier == 1,
        "High": true_tier == 2,
        "Critical": true_tier == 3,
        "Severe (High+Critical)": true_tier >= 2,
        "Non-Low (all)": true_tier >= 1,
    }


# --------------------------------------------------------------------------
# Analysis (reads the cache; also used by failsafe.py)
# --------------------------------------------------------------------------

def load_cache(path=CACHE_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run attacks.py first.")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def analyse(cache, policy_names=None):
    policy_names = policy_names or list(shared.POLICIES)
    rows = cache["att_rows"]
    true = cache["true_tier"][rows]
    labels = cache["labels"][rows]
    pred_clean = cache["clean_proba"][rows].argmax(-1)
    natural = TIER_MIN[pred_clean] < TIER_MIN[true]   # downgraded with no attack

    K = cache["proba_structured"].shape[1]
    budgets = sorted({b for b in (1, 5, 10, 25, K) if b <= K})
    masks = group_masks(true)
    V_by_policy = {p: policy_matrices(p)[0] for p in policy_names}

    asr_rows, cdr_rows, label_rows = [], [], []

    for attack in ("structured", "random"):
        pred = cache[f"proba_{attack}"].argmax(-1)                  # (n, K)
        clean_correct = pred_clean == true
        # Population for each metric excludes samples already in that state with
        # no attack, so the rate isolates what the attack adds.
        events = {
            "misclassified": (pred != true[:, None], pred_clean != true),
            "tier lowered": (pred < true[:, None], pred_clean < true),
            "strength downgrade": (TIER_MIN[pred] < TIER_MIN[true][:, None], natural),
        }
        first_by_metric = {}
        for metric, (succ_q, nat) in events.items():
            first_m = np.where(succ_q.any(1), succ_q.argmax(1), K)   # first winning query
            first_by_metric[metric] = first_m
            for gname, mask in masks.items():
                population = mask & ~nat
                for k in budgets:
                    # robust accuracy: correct on the clean flow AND on every one of k queries
                    robust = clean_correct & (pred[:, :k] == true[:, None]).all(1)
                    asr_rows.append({
                        "Attack": attack, "Budget": k, "Group": gname,
                        "Metric": metric, "N": int(mask.sum()),
                        "Natural rate": float(nat[mask].mean()) if mask.any() else np.nan,
                        "Clean accuracy": float(clean_correct[mask].mean()) if mask.any() else np.nan,
                        "Robust accuracy": float(robust[mask].mean()) if mask.any() else np.nan,
                        "N attackable": int(population.sum()),
                        "ASR": (float((first_m[population] < k).mean())
                                if population.any() else np.nan),
                    })
        first = first_by_metric["strength downgrade"]

        for pname in policy_names:
            v = V_by_policy[pname]
            v_clean = v[true, pred_clean]
            v_query = v[true[:, None], pred]
            for gname, mask in masks.items():
                if not mask.any():
                    continue
                cdr_rows.append({
                    "Attack": attack, "Budget": 0, "Group": gname,
                    "Policy": pname, "N": int(mask.sum()),
                    "CDR": float(v_clean[mask].mean()),
                })
                for k in budgets:
                    best = np.maximum(v_clean, v_query[:, :k].max(1))
                    cdr_rows.append({
                        "Attack": attack, "Budget": k, "Group": gname,
                        "Policy": pname, "N": int(mask.sum()),
                        "CDR": float(best[mask].mean()),
                    })

        severe_or_medium = true >= 1
        for label in sorted(set(labels[severe_or_medium])):
            m = severe_or_medium & (labels == label)
            pop = m & ~natural
            label_rows.append({
                "Attack": attack, "Label": label,
                "Tier": SEVERITIES[int(true[m][0])], "N": int(m.sum()),
                "Natural downgrade rate": float(natural[m].mean()),
                f"ASR at {K} queries": float((first[pop] < K).mean()) if pop.any() else np.nan,
            })

    return (pd.DataFrame(asr_rows), pd.DataFrame(cdr_rows),
            pd.DataFrame(label_rows), K)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None,
                        help="attack at most this many non-Low test samples (quick run)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk", type=int, default=400)
    args = parser.parse_args()

    started = time.time()
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run classifier.py first.")

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    if "test_index" not in artifact:
        raise ValueError("Model file has no test_index; re-run the new classifier.py.")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    test = df.loc[artifact["test_index"]]
    X_test = prepare_features(test, feature_columns).to_numpy(dtype=np.float32)
    true_tier = test["Severity"].map(TIER_INDEX).to_numpy()
    labels = np.array(test["Label"].astype(str).tolist(), dtype=str)

    groups, missing = build_groups(feature_columns)
    mutable = np.unique(np.concatenate(list(groups.values())))
    print("Mutable features found:", len(mutable))
    if missing:
        print("  not present in this dataset (skipped):", ", ".join(missing))
    if len(mutable) == 0:
        raise ValueError("None of the mutable features exist in the data.")

    print(f"Test rows: {len(test):,}")
    clean_proba = tier_proba(model, X_test, feature_columns)

    att_rows = np.flatnonzero(true_tier >= 1)       # Medium, High, Critical
    rng = np.random.default_rng(args.seed)
    if args.max_samples and len(att_rows) > args.max_samples:
        att_rows = np.sort(rng.choice(att_rows, args.max_samples, replace=False))
    print(f"Attacking {len(att_rows):,} non-Low test samples "
          f"(Medium {int((true_tier[att_rows] == 1).sum()):,}, "
          f"High {int((true_tier[att_rows] == 2).sum()):,}, "
          f"Critical {int((true_tier[att_rows] == 3).sum()):,})")

    queries = structured_queries(args.seed)
    K = len(queries)
    print(f"Query budget per sample: {K}")

    X_att = X_test[att_rows]
    proba_structured = run_attack(model, X_att, feature_columns, "structured",
                                  queries, groups, mutable, K, args.chunk, rng)
    proba_random = run_attack(model, X_att, feature_columns, "random",
                              queries, groups, mutable, K, args.chunk, rng)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        clean_proba=clean_proba,
        true_tier=true_tier.astype(np.int8),
        labels=labels,
        att_rows=att_rows,
        proba_structured=proba_structured,
        proba_random=proba_random,
        queries=queries,
    )

    cache = load_cache()
    asr, cdr, by_label, K = analyse(cache)
    asr.round(6).to_csv(RESULTS_DIR / "attack_asr.csv", index=False)
    cdr.round(6).to_csv(RESULTS_DIR / "attack_cdr.csv", index=False)
    by_label.round(6).to_csv(RESULTS_DIR / "attack_by_label.csv", index=False)

    sev = "Severe (High+Critical)"
    print("\n=== Attack success rates, severe events (High+Critical) ===")
    print("(misclassified, tier lowered and strength downgrade are different events)")
    view = asr[asr["Group"] == sev][
        ["Attack", "Metric", "Budget", "N", "Natural rate", "Robust accuracy", "ASR"]]
    print(view.round(4).to_string(index=False))

    print(f"\n=== Confidentiality downgrade rate, policy: {shared.DEFAULT_POLICY} ===")
    view = cdr[(cdr["Group"] == sev) & (cdr["Policy"] == shared.DEFAULT_POLICY)]
    print(view[["Attack", "Budget", "CDR"]].round(4).to_string(index=False))

    print("\nBy original label (full budget):")
    print(by_label.round(4).to_string(index=False))

    print(f"\nSaved attack_asr.csv, attack_cdr.csv, attack_by_label.csv, attack_cache.npz "
          f"in {RESULTS_DIR}")
    print(f"Done in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
