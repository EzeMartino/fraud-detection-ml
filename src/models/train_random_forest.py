import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import DATA_FILE, REPORTS_DIR, TARGET_COLUMN
from src.models.evaluate import top_k_metrics
from src.features.build_features import build_features


def load_data():
    return pd.read_csv(DATA_FILE)


def preprocess(df):
    df = df.drop_duplicates()
    
    df = build_features(df)
    
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


def train_model(X_train, y_train, config):
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"] or 200,
        max_depth=config["max_depth"] or None,
        min_samples_leaf=config["min_samples_leaf"] or 1,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    
    calibrated_model = CalibratedClassifierCV(
        model, 
        method="isotonic", 
        cv=3)
    
    calibrated_model.fit(X_train, y_train)
    
    return calibrated_model


def evaluate_model(model, X_test, y_test, save_plot=True, plot_name="calibration_curve_RF.png"):
    y_proba = model.predict_proba(X_test)[:, 1]
    
    brier = brier_score_loss(y_test, y_proba)

    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    
    if save_plot:
        plt.figure()
        plt.plot(prob_pred, prob_true, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")

        plt.title("Calibration Curve")
        plt.xlabel("Predicted probability")
        plt.ylabel("True probability")

        plt.savefig(f"reports/{plot_name}")
        plt.close()

    return {
        "pr_auc": average_precision_score(y_test, y_proba),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "top_1pct": top_k_metrics(y_test, y_proba, 0.01),
        "top_0_5pct": top_k_metrics(y_test, y_proba, 0.005),
        "top_0_2pct": top_k_metrics(y_test, y_proba, 0.002),
        "top_0_1pct": top_k_metrics(y_test, y_proba, 0.001),
        "brier_score": brier,
    }


def save_results(results, custom_path="rf_metrics.json"):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    path = REPORTS_DIR / custom_path

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Saved RF results to {path}")


def main():
    print("Loading data...")
    df = load_data()

    print("Preprocessing...")
    X, y = preprocess(df)

    print("Splitting...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training Random Forest...")
    calibrated_model = train_model(X_train, y_train)

    print("Evaluating...")
    results = evaluate_model(calibrated_model, X_test, y_test)

    print("\nResults:")
    for k, v in results.items():
        print(k, ":", v)

    save_results(results)


if __name__ == "__main__":
    main()