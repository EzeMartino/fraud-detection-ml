from datetime import datetime, timezone
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import CONFIG_FILENAME, DATA_FILE, LATEST_FILE, METADATA_FILENAME, MODELS_DIR, PIPELINE_FILENAME, REPORTS_DIR, TARGET_COLUMN, THRESHOLD, get_best_config_path
from src.models.evaluate import top_k_metrics
from src.utils.versioning import compute_model_version
# from src.features.build_features import build_features didn't help in TopK


def load_data():
    return pd.read_csv(DATA_FILE)

def preprocess(df):
    df = df.drop_duplicates()
    
    # df = build_features(df) Didn't help in TopK
    
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
        "top_k_performance":{
            "top_1pct": top_k_metrics(y_test, y_proba, 0.01),
            "top_0_5pct": top_k_metrics(y_test, y_proba, 0.005),
            "top_0_2pct": top_k_metrics(y_test, y_proba, 0.002),
            "top_0_1pct": top_k_metrics(y_test, y_proba, 0.001),
        },
        "brier_score": brier,
    }


def save_results(results, custom_path="rf_metrics.json"):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    path = REPORTS_DIR / custom_path

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Saved RF results to {path}")


def main():
    df = load_data()

    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    config_path = get_best_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"{CONFIG_FILENAME} not found. Run tune_random_forest.py to get best config")
    
    with config_path.open("r", encoding="utf-8") as f:
            best_config = json.load(f)
    
    calibrated_model = train_model(X_train, y_train, best_config)

    results = evaluate_model(calibrated_model, X_test, y_test)

    save_results(results)

    # Export model
    # Save best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    staging_dir = MODELS_DIR / "_staging"
    staging_dir.mkdir(exist_ok=True)
    
    pipeline_path = staging_dir / PIPELINE_FILENAME
    joblib.dump(calibrated_model, pipeline_path)
        
    # Save metadata
    metadata = {
        "model_version": None,  # Placeholder, will be updated after computing the version
        "model_type": "RandomForestClassifier",
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": "Credit Card Fraud Detection (Kaggle)",
        "exported_model_name": PIPELINE_FILENAME,
        "target_column": TARGET_COLUMN,
        "positive_class_rate": float(y_train.mean()),
        "feature_count": X.shape[1],
        "features_used": list(X.columns),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "default_threshold": THRESHOLD,
        "calibrated": True,
        "calibration_method": "isotonic",
        "selected_metric": "precision_at_0_2pct",
        "selection_criteria": "Best operational performance with lowest computational cost among tied configs",
        "metrics": results,
    }
    metadata_path = staging_dir / METADATA_FILENAME
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    model_version = compute_model_version(pipeline_path, metadata_path)
    metadata["model_version"] = model_version
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    final_model_dir = MODELS_DIR / f"model_{model_version}"
    
    if final_model_dir.exists():
        raise FileExistsError(f"Model directory already exists: {final_model_dir}")

    staging_dir.rename(final_model_dir)
    
    LATEST_FILE.write_text(final_model_dir.name, encoding="utf-8")
    
    print(f"[OK] Saved model and metadata to {final_model_dir}")

if __name__ == "__main__":
    main()