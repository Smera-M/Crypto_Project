
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
csv_files = sorted(DATA_DIR.rglob("*.csv"))

if not csv_files:
    print("No CSV files found in:", DATA_DIR)
    print("Check whether the ZIP was extracted into the data folder.")
    raise SystemExit(1)

print("Dataset folder:", DATA_DIR)
print("CSV files found:", len(csv_files))

for file in csv_files:
    print("\n" + "=" * 65)
    print("FILE:", file.relative_to(DATA_DIR))

    try:
        # Inspect column names and a small sample first
        sample = pd.read_csv(file, nrows=5, low_memory=False)
        sample.columns = sample.columns.str.strip()

        print("Sample shape:", sample.shape)
        print("Columns:", sample.columns.tolist())

        label_columns = [
            col for col in sample.columns
            if col.lower() == "label"
        ]

        if not label_columns:
            print("WARNING: No Label column found.")
            continue

        label_col = label_columns[0]

        # Read only the label column to count all labels
        labels = pd.read_csv(
            file,
            usecols=lambda col: col.strip() == label_col,
            low_memory=False
        )
        labels.columns = labels.columns.str.strip()

        print("\nLabel counts:")
        print(labels[label_col].value_counts(dropna=False).to_string())

        print("\nMissing labels:", labels[label_col].isna().sum())

    except Exception as error:
        print("ERROR:", error)

print("\nDataset inspection complete.")
