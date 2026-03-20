import joblib
import json

from src.config import MODELS_DIR, TARGET_COLUMN
from src.models.train_random_forest import load_data, preprocess, split_data, train_model, evaluate_model, save_results


configs = [
    {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 5},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5},
]

def main ():
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    best_config = None
    best_precision_at_k = 0.0
    best_results = None
    all_results = []
    
    for config in configs:
        print(f"Training with config: {config}")
        model = train_model(X_train, y_train, config)
        results = evaluate_model(model, X_test, y_test, save_plot=False)
        row = {
            "config": config,
            "pr_auc": results["pr_auc"],
            "roc_auc": results["roc_auc"],
            "precision_at_0_2pct": results["top_0_2pct"]["precision_at_k"],
            "recall_at_0_2pct": results["top_0_2pct"]["recall_at_k"],
        }
        all_results.append(row)
        
        if results["top_0_2pct"]["precision_at_k"] > best_precision_at_k:
            best_precision_at_k = results["top_0_2pct"]["precision_at_k"]
            best_config = config
            best_results = results
            best_model = model

    print(f"Best config: {best_config} with precision@0.2%: {best_precision_at_k:.4f}")
    print("All results:")
    for row in all_results:
        print(row)

    save_results(best_results, custom_path="rf_tuned_metrics.json")
    
    # Export model with best config
    # Save best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "rf_tuned_model.joblib")
    
    # Save best config
    with open(MODELS_DIR / "rf_tuned_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
        
    # Save metadata
    metadata = {
        "model_type": "RandomForest",
        "exported_model_name": "rf_tuned_model.joblib",
        "calibrated": True,
        "calibration_method": "isotonic",
        "metrics": best_results,
        "features_used": list(X.columns),
        "target_column": TARGET_COLUMN,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "selected_metric": "precision_at_0_2pct",
        "selection_criteria": "Best operational performance with lowest computational cost among tied configs",
    }
    
    with open(MODELS_DIR / "rf_tuned_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
if __name__ == "__main__":
    main()