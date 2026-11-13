"""
Consistency checks for the severity mapping across the whole project.

The mapping in data.SEVERITY_MAP is a project-defined policy assumption, not a
severity rating supplied by CIC-IDS2017. It is used in three places that must
stay in step:
  1. preprocessing  (data.py writes the Severity column used as the target)
  2. classification (classifier.py trains on that stored column)
  3. policy / runtime (leakage.py evaluates policies; crypto.py applies one)

This script checks what can be checked automatically:
  - every label in the raw CSV files has an explicit mapping
  - the saved original label counts match the raw files
  - the processed dataset still agrees with the current mapping
  - downstream results are not older than the files they depend on
  - crypto.py takes its cipher policy from policy.py, the same module leakage.py uses
  - results were produced under the CURRENT mapping, policies, classifier and attack
    settings and the CURRENT raw data files (fingerprints in results/run_config.json)

It cannot check whether the tier assignments are DEFENSIBLE; that needs the
written rationale in the report.

Run from the project root:  python src/check_mapping.py
Exit status is 1 if any hard check fails.
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(SRC_DIR))
from data import SEVERITY_MAP, normalize_label  # noqa: E402
import leakage as L  # noqa: E402
import policy as shared  # noqa: E402
import record_config as RC  # noqa: E402

TIERS = L.SEVERITIES
MIN_ROWS_PER_TIER = 50  # warn if a tier is thinner than this in the training set

failures = []
warnings = []


def ok(msg):
    print(f"  ok:   {msg}")


def fail(msg):
    failures.append(msg)
    print(f"  FAIL: {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"  WARN: {msg}")


def section(title):
    print(f"\n[{title}]")


# --------------------------------------------------------------------------

def scan_raw_labels():
    """Count normalised labels across all raw CSVs (labels column only)."""
    files = sorted(DATA_DIR.rglob("*.csv"))
    counts = {}
    for f in files:
        labels = pd.read_csv(
            f, usecols=lambda c: c.strip() == "Label", low_memory=False)
        labels.columns = labels.columns.str.strip()
        if "Label" not in labels.columns:
            warn(f"{f.name}: no Label column (data.py skips this file too)")
            continue
        for label, n in labels["Label"].map(normalize_label).value_counts().items():
            counts[label] = counts.get(label, 0) + int(n)
    return files, counts


def check_raw_labels():
    section("1. Raw CSV labels vs SEVERITY_MAP")
    if not DATA_DIR.exists() or not list(DATA_DIR.rglob("*.csv")):
        fail(f"no CSV files found under {DATA_DIR}")
        return None

    files, counts = scan_raw_labels()
    total = sum(counts.values())
    print(f"  {len(files)} CSV files, {total:,} labelled records, "
          f"{len(counts)} distinct labels")

    unmapped = {l: n for l, n in counts.items() if l not in SEVERITY_MAP}
    if unmapped:
        detail = ", ".join(f"{l} ({n:,})" for l, n in unmapped.items())
        fail(f"labels with no mapping: {detail}")
    else:
        ok("every label in the raw files has an explicit mapping")

    bad_tiers = {l: t for l, t in SEVERITY_MAP.items() if t not in TIERS}
    if bad_tiers:
        fail(f"mapping uses tiers outside {TIERS}: {bad_tiers}")
    else:
        ok(f"all mapped tiers are among {TIERS}")

    never_seen = sorted(set(SEVERITY_MAP) - set(counts))
    if never_seen:
        warn("mapping entries never seen in the data: " + ", ".join(never_seen))

    print("\n  Tier composition of the raw data:")
    for tier in TIERS:
        labels = [(l, n) for l, n in counts.items() if SEVERITY_MAP.get(l) == tier]
        tier_total = sum(n for _, n in labels)
        if tier_total == 0:
            fail(f"tier '{tier}' has no records")
        listing = ", ".join(f"{l} ({n:,})" for l, n in sorted(labels))
        share = tier_total / total if total else 0
        print(f"    {tier:9s} {tier_total:>10,}  {share:7.3%}  {listing}")
    return counts


def check_original_counts(counts):
    section("2. results/original_label_counts.csv vs raw files")
    path = RESULTS_DIR / "original_label_counts.csv"
    if counts is None:
        warn("skipped (raw labels unavailable)")
        return
    if not path.exists():
        warn(f"{path.name} not found; run data.py")
        return
    saved = pd.read_csv(path, index_col=0).iloc[:, 0].astype(int).to_dict()
    diffs = []
    for label in sorted(set(saved) | set(counts)):
        if saved.get(label, 0) != counts.get(label, 0):
            diffs.append(f"{label}: saved {saved.get(label, 0):,} vs raw "
                         f"{counts.get(label, 0):,}")
    if diffs:
        fail("saved counts differ from the raw files (re-run data.py): "
             + "; ".join(diffs))
    else:
        ok("saved original label counts match the raw files exactly")


def check_processed_dataset():
    section("3. results/processed_dataset.csv vs current SEVERITY_MAP")
    path = RESULTS_DIR / "processed_dataset.csv"
    if not path.exists():
        warn(f"{path.name} not found; run data.py")
        return
    df = pd.read_csv(path, usecols=["Label", "Severity"])
    expected = df["Label"].map(SEVERITY_MAP)
    mismatched = int((expected != df["Severity"]).sum())
    if mismatched:
        fail(f"{mismatched:,} of {len(df):,} rows disagree with the current "
             "SEVERITY_MAP; re-run data.py, classifier.py and leakage.py")
    else:
        ok(f"all {len(df):,} processed rows agree with the current mapping")

    sizes = df["Severity"].value_counts().reindex(TIERS, fill_value=0)
    print("\n  Training-set rows per tier (capped sample, not the original mix):")
    for tier, n in sizes.items():
        print(f"    {tier:9s} {n:>10,}")
        if n < MIN_ROWS_PER_TIER:
            warn(f"tier '{tier}' has only {n} processed rows")


def check_freshness():
    section("4. Are downstream results older than their inputs?")
    paths = {
        "data.py": SRC_DIR / "data.py",
        "original_label_counts.csv": RESULTS_DIR / "original_label_counts.csv",
        "processed_dataset.csv": RESULTS_DIR / "processed_dataset.csv",
        "random_forest.joblib": RESULTS_DIR / "random_forest.joblib",
        "leakage_summary.csv": RESULTS_DIR / "leakage_summary.csv",
    }
    pairs = [
        ("processed_dataset.csv", "data.py"),
        ("random_forest.joblib", "processed_dataset.csv"),
        ("leakage_summary.csv", "data.py"),
        ("leakage_summary.csv", "original_label_counts.csv"),
    ]
    any_issue = False
    for downstream, upstream in pairs:
        d, u = paths[downstream], paths[upstream]
        if not d.exists() or not u.exists():
            continue
        if d.stat().st_mtime < u.stat().st_mtime:
            any_issue = True
            warn(f"{downstream} is older than {upstream}; re-run the step "
                 "that produces it (timestamps can mislead after copying files)")
    if not any_issue:
        ok("no downstream file is older than its inputs")


def check_runtime_policy():
    section("5. crypto.py and leakage.py share one policy definition")
    path = SRC_DIR / "crypto.py"
    if not path.exists():
        warn("crypto.py not found; skipped")
        return
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as error:
        fail(f"crypto.py has a syntax error: {error}")
        return

    imports_policy = any(
        (isinstance(n, ast.ImportFrom) and n.module == "policy")
        or (isinstance(n, ast.Import) and any(a.name == "policy" for a in n.names))
        for n in ast.walk(tree))
    private_table = any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CIPHER_POLICY" for t in n.targets)
        for n in tree.body)

    if private_table:
        fail("crypto.py still defines its own CIPHER_POLICY; use policy.py so "
             "runtime and analysis cannot drift apart")
    elif not imports_policy:
        fail("crypto.py does not import policy.py")
    else:
        ok("crypto.py takes its cipher policy from policy.py (no private table)")

    analysed = set(L.build_policies())
    registered = set(shared.POLICIES)
    if analysed != registered:
        fail("leakage.py analyses a different set of policies than policy.py "
             f"registers: {sorted(analysed ^ registered)}")
    else:
        ok(f"leakage.py analyses all {len(registered)} registered policies")

    ok(f"deployed default policy: {shared.DEFAULT_POLICY}")
    print("      (this must be the policy you describe in the report; "
          "change DEFAULT_POLICY in policy.py, then re-run leakage.py)")


def check_run_config():
    section("6. Results were produced under the current configuration")
    path = RESULTS_DIR / "run_config.json"
    if not path.exists():
        warn("run_config.json not found; run record_config.py before the pipeline")
        return
    stored = json.loads(path.read_text())

    current = RC.build_config(hash_files=False)
    if RC.fingerprint(current) != stored.get("fingerprint"):
        changed = ", ".join(RC.differing_sections(stored, current)) or "unknown section"
        fail(f"configuration changed since the results were produced ({changed}); "
             "re-run record_config.py and the steps after it")
    else:
        ok("severity mapping, policies, classifier and attack settings match the recorded run")

    if stored.get("dataset_fingerprint"):
        full = RC.build_config(hash_files=True)
        if RC.dataset_fingerprint(full) != stored["dataset_fingerprint"]:
            fail("the raw data files differ from the ones the results were produced from")
        else:
            ok("raw data files match the recorded SHA-256 hashes")
    else:
        warn("run_config.json has no data hashes (recorded with --no-hash)")

    print(f"      recorded: deployed policy = {stored['policy']['default']}, "
          f"created {stored.get('created_utc')}")


# --------------------------------------------------------------------------

def main():
    print("Severity-mapping consistency check")
    print(f"Project root: {ROOT}")
    print(f"Tiers: {TIERS}")

    counts = check_raw_labels()
    check_original_counts(counts)
    check_processed_dataset()
    check_freshness()
    check_runtime_policy()
    check_run_config()

    print("\n" + "=" * 60)
    print(f"{len(failures)} failure(s), {len(warnings)} warning(s)")
    for msg in failures:
        print("  FAIL:", msg)
    for msg in warnings:
        print("  WARN:", msg)
    print("These checks do not establish that the tier assignments are "
          "defensible; document the rationale for each tier in the report.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
