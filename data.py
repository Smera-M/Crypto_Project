
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

MAX_ROWS_PER_LABEL_PER_FILE = 20000
RANDOM_STATE = 42

SEVERITY_MAP = {
    "BENIGN": "Low",
    "PortScan": "Low",
    "FTP-Patator": "Medium",
    "SSH-Patator": "Medium",
    "Web Attack - Brute Force": "Medium",
    "Web Attack - XSS": "Medium",
    "Web Attack - Sql Injection": "Medium",
    "DoS Hulk": "High",
    "DoS GoldenEye": "High",
    "DoS slowloris": "High",
    "DoS Slowhttptest": "High",
    "Bot": "High",
    "Infiltration": "High",
    "Heartbleed": "High",
    "DDoS": "Critical",
}


def normalize_label(label):
    label = str(label).strip()
    label = label.replace("\ufffd", "-")
    label = " ".join(label.split())

    lower = label.lower()

    if lower.startswith("web attack"):
        if "brute force" in lower:
            return "Web Attack - Brute Force"
        if "xss" in lower:
            return "Web Attack - XSS"
        if "sql injection" in lower:
            return "Web Attack - Sql Injection"

    return label


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(DATA_DIR.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}. "
            "Check that the dataset is extracted into the data folder."
        )

    print("Project root:", ROOT)
    print("Dataset folder:", DATA_DIR)
    print("CSV files found:", len(csv_files))

    sampled_frames = []
    original_counts = {}
    unknown_labels = set()

    for file in csv_files:
        print(f"\nReading: {file.name}")

        df = pd.read_csv(file, low_memory=False)
        df.columns = df.columns.str.strip()

        if "Label" not in df.columns:
            print("Skipping: Label column not found.")
            continue

        df["Label"] = df["Label"].map(normalize_label)
        df = df.dropna(subset=["Label"])
        df["SourceFile"] = file.name   # kept for grouping/evaluation, never a model feature

        for label, count in df["Label"].value_counts().items():
            original_counts[label] = (
                original_counts.get(label, 0) + int(count)
            )

        unknown = set(df["Label"].unique()) - set(SEVERITY_MAP)
        unknown_labels.update(unknown)

        # Sample each label within each file.
        parts = []

        for label, group in df.groupby("Label", sort=False):
            n = min(len(group), MAX_ROWS_PER_LABEL_PER_FILE)
            parts.append(
                group.sample(n=n, random_state=RANDOM_STATE)
            )

        if parts:
            sampled_frames.append(
                pd.concat(parts, ignore_index=True)
            )

        print("Rows sampled:", sum(len(part) for part in parts))

        del df

    if unknown_labels:
        raise ValueError(
            "These labels need explicit severity mappings: "
            + ", ".join(sorted(unknown_labels))
        )

    if not sampled_frames:
        raise ValueError("No usable dataset rows were found.")

    data = pd.concat(sampled_frames, ignore_index=True)
    stage_sampled = data["Label"].value_counts()

    before = len(data)
    # Same duplicate rule as before: SourceFile is ignored, so a row repeated
    # across capture files is still removed.
    data = data.drop_duplicates(
        subset=[col for col in data.columns if col != "SourceFile"])
    stage_dedup = data["Label"].value_counts()
    print("\nDuplicate rows removed:", before - len(data))

    feature_columns = [
        col for col in data.columns if col not in ("Label", "SourceFile")
    ]

    # Convert features to numeric values.
    data[feature_columns] = data[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    data = data.replace([np.inf, -np.inf], np.nan)
    before = len(data)
    data = data.dropna(subset=feature_columns)
    print("Rows removed for invalid features:", before - len(data))
    stage_final = data["Label"].value_counts()

    data["Severity"] = data["Label"].map(SEVERITY_MAP)

    if data["Severity"].isna().any():
        raise ValueError("Some rows have no severity assignment.")

    output_path = RESULTS_DIR / "processed_dataset.csv"
    data.to_csv(output_path, index=False)

    pd.Series(
        original_counts, name="original_rows"
    ).sort_index().to_csv(
        RESULTS_DIR / "original_label_counts.csv"
    )

    severity_counts = (
        data["Severity"]
        .value_counts()
        .reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
    )
    severity_counts.rename_axis("Severity").to_csv(
        RESULTS_DIR / "severity_counts.csv"
    )

    # Document what each preprocessing stage did to every class.
    report = pd.DataFrame({
        "Original rows": pd.Series(original_counts),
        "After sampling cap": stage_sampled,
        "After duplicate removal": stage_dedup,
        "Final rows": stage_final,
    }).fillna(0).astype(int)
    report.insert(0, "Severity", [SEVERITY_MAP[label] for label in report.index])
    report["Retained share"] = report["Final rows"] / report["Original rows"]
    report["Original share of all rows"] = (
        report["Original rows"] / report["Original rows"].sum())
    report["Final share of all rows"] = (
        report["Final rows"] / report["Final rows"].sum())
    report.rename_axis("Label").round(6).to_csv(
        RESULTS_DIR / "preprocessing_report.csv")

    tiers = report.groupby("Severity")[
        ["Original rows", "Final rows", "Original share of all rows",
         "Final share of all rows"]].sum().reindex(
        ["Low", "Medium", "High", "Critical"])
    print("\nClass balance before and after preprocessing (by severity tier):")
    print(tiers.round(4).to_string())

    print("\nOriginal attack-label counts:")
    print(pd.Series(original_counts).sort_index().to_string())

    print("\nProcessed severity counts:")
    print(severity_counts.to_string())

    print("\nFinal dataset shape:", data.shape)
    print("Saved processed dataset:", output_path)
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
