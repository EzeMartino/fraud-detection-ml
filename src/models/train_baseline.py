import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DATA_FILE, REPORTS_DIR, TARGET_COLUMN


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.drop_duplicates()
    
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def train_model(X_train, y_train):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


def save_results(results: dict):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    path = REPORTS_DIR / "baseline_metrics.json"

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Saved results to {path}")


def main():
    print("Loading data...")
    df = load_data()

    print("Preprocessing...")
    X, y = preprocess(df)

    print("Splitting...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Scaling...")
    X_train, X_test = scale_data(X_train, X_test)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating...")
    results = evaluate_model(model, X_test, y_test)

    print("Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    save_results(results)


if __name__ == "__main__":
    main()