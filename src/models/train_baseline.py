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
from src.models.evaluate import compute_threshold_metrics, save_threshold_metrics, save_top_k_metrics, top_k_metrics


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

    threshold_df = compute_threshold_metrics(y_test, y_proba)
    top_1pct = top_k_metrics(y_test, y_proba, top_fraction=0.001)
    
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold_df": threshold_df,
        "top_0_1pct": top_1pct
    }


def save_results(results: dict):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = REPORTS_DIR / "baseline_metrics.json"
    threshold_path = REPORTS_DIR / "baseline_threshold_metrics.csv"
    top_k_path = REPORTS_DIR / "baseline_top_0_1pct.json"

    serializable_metrics = {
        "pr_auc": results["pr_auc"],
        "roc_auc": results["roc_auc"],
    }
    
    with open(metrics_path, "w") as f:
        json.dump(serializable_metrics, f, indent=2)

    save_threshold_metrics(results["threshold_df"], threshold_path)
    save_top_k_metrics(results["top_0_1pct"], top_k_path)
    
    print(f"[OK] Saved baseline metrics to {REPORTS_DIR}")


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
    print(f"PR-AUC: {results['pr_auc']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    
    print("Top 0.1% metrics:")
    for k, v in results["top_0_1pct"].items():
        print(f"{k}: {v}")

    save_results(results)


if __name__ == "__main__":
    main()