import json

import pandas as pd

from src.config import DATA_FILE, REPORTS_DIR, TARGET_COLUMN


def load_data(file_path: str = str(DATA_FILE)) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(file_path)


def build_profile(df: pd.DataFrame) -> dict:
    """Build a basic profiling summary for the fraud dataset."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    total_rows = len(df)
    fraud_count = int(class_counts.get(1, 0))
    non_fraud_count = int(class_counts.get(0, 0))

    majority_class_baseline_accuracy = max(class_counts) / total_rows if total_rows > 0 else 0.0

    profile = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "target_column": TARGET_COLUMN,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_values_by_column": {col: int(val) for col, val in df.isna().sum().items()},
        "duplicated_rows": int(df.duplicated().sum()),
        "class_distribution": {
            "0": non_fraud_count,
            "1": fraud_count,
        },
        "class_ratio_fraud": fraud_count / total_rows if total_rows > 0 else 0.0,
        "majority_class_baseline_accuracy": majority_class_baseline_accuracy,
        "amount_stats": df["Amount"].describe().to_dict() if "Amount" in df.columns else {},
        "time_stats": df["Time"].describe().to_dict() if "Time" in df.columns else {},
    }

    return profile


def save_profile(profile: dict) -> None:
    """Save profiling summary to reports directory."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "profile_summary.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"[OK] Saved profile report to {output_path}")


def main() -> None:
    print(f"Loading dataset from: {DATA_FILE}")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. Place creditcard.csv inside data/raw/."
        )

    df = load_data()

    print("\nDataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Duplicated rows: {df.duplicated().sum()}")

    if TARGET_COLUMN in df.columns:
        print("\nClass distribution:")
        print(df[TARGET_COLUMN].value_counts(normalize=False).sort_index())
        print("\nClass distribution (%):")
        print((df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100).round(4))

    profile = build_profile(df)
    save_profile(profile)


if __name__ == "__main__":
    main()