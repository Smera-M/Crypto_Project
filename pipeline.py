"""
End-to-end path for one flow: features -> severity tier -> cipher -> encryption.

GUARANTEES
  - Normal and attacked flows take the SAME path. Feature preparation is
    classifier.prepare_features and inference is attacks.tier_proba, the same
    code used for training and for the attacks.
  - Any failure to classify (missing or non-numeric feature, NaN or infinity,
    an exception, invalid probabilities) falls back to policy.FALLBACK_TIER, the
    strongest tier. The result is flagged, and the call never raises or silently
    selects weaker encryption.
  - An optional fail-safe rule (failsafe.effective_tier) can escalate uncertain
    results to the strongest tier.

  python src/pipeline.py      demo on test flows and on deliberately broken input
"""

import warnings

import joblib
import numpy as np
import pandas as pd

import policy as P
from attacks import MODEL_PATH, DATA_PATH, tier_proba
from classifier import prepare_features
from failsafe import effective_tier


class SeverityPipeline:
    def __init__(self, model_path=MODEL_PATH, policy=P.DEFAULT_POLICY,
                 rule="none", threshold=0.0):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.columns = artifact["feature_columns"]
        self.policy = policy
        self.rule = rule
        self.threshold = threshold

    def classify(self, flow):
        """flow: mapping of feature name -> value. Never raises."""
        try:
            row = pd.DataFrame([dict(flow)])
            missing = [c for c in self.columns if c not in row.columns]
            if missing:
                raise ValueError(f"missing features, e.g. {missing[:3]}")
            values = prepare_features(row, self.columns).to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("non-numeric, NaN or infinite feature value")
            proba = tier_proba(self.model, values, self.columns)[0]
            if not np.isfinite(proba).all() or abs(float(proba.sum()) - 1.0) > 1e-3:
                raise ValueError("invalid class probabilities")
            tier = int(effective_tier(proba, self.rule, self.threshold))
            return {"tier": P.SEVERITIES[tier], "proba": proba,
                    "fallback": False, "reason": None}
        except Exception as error:   # noqa: BLE001 - fail closed on anything
            return {"tier": P.FALLBACK_TIER, "proba": None, "fallback": True,
                    "reason": f"{type(error).__name__}: {error}"}

    def process(self, flow, message, key_store, rng=None):
        """Classify, then encrypt `message` under the policy for the result."""
        from crypto import encrypt_message   # imported here so classifying needs no crypto library

        result = self.classify(flow)
        record = encrypt_message(message, result["tier"], key_store,
                                 policy=self.policy, rng=rng)
        record["classifier_fallback"] = result["fallback"]
        record["classifier_reason"] = result["reason"]
        return record


def main():
    from crypto import InMemoryKeyStore, decrypt_message

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run classifier.py first.")
    pipeline = SeverityPipeline()
    artifact = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    test = df.loc[artifact["test_index"]]
    store = InMemoryKeyStore()
    message = "confidential log entry"

    print(f"Policy: {pipeline.policy}   fail-safe rule: {pipeline.rule}\n")
    print("Normal test flows:")
    print(f"  {'true tier':10s} {'classified':11s} {'cipher':19s} fallback  round trip")
    for tier in P.SEVERITIES:
        sample = test[test["Severity"] == tier].head(2)
        for _, row in sample.iterrows():
            record = pipeline.process(row[pipeline.columns].to_dict(), message, store)
            okay = decrypt_message(record, store) == message
            print(f"  {tier:10s} {record['severity']:11s} {record['cipher_id']:19s} "
                  f"{str(record['classifier_fallback']):8s}  {okay}")

    good = test.iloc[0][pipeline.columns].to_dict()
    first = pipeline.columns[0]
    broken = {
        "missing feature": {k: v for k, v in good.items() if k != first},
        "NaN value": {**good, first: float("nan")},
        "infinite value": {**good, first: float("inf")},
        "text in a numeric field": {**good, first: "not a number"},
        "None instead of a flow": None,
        "empty flow": {},
    }
    print("\nBroken input (every one must fall back to the strongest tier):")
    all_safe = True
    for label, flow in broken.items():
        with warnings.catch_warnings():
            warnings.simplefilter("error")        # a fallback must not warn or raise here
            record = pipeline.process(flow, message, store)
        safe = (record["severity"] == P.FALLBACK_TIER and record["classifier_fallback"]
                and P.STRENGTH[record["cipher_id"]] >= max(P.MIN_STRENGTH.values())
                and decrypt_message(record, store) == message)
        all_safe &= bool(safe)
        print(f"  {label:26s} -> {record['severity']:9s} {record['cipher_id']:19s} "
              f"{'safe' if safe else 'NOT SAFE'}")
    if not all_safe:
        raise SystemExit("A broken input did not fall back safely.")
    print("\nAll broken inputs fell back to the strongest tier.")


if __name__ == "__main__":
    main()
