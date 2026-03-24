import json

from src.config import get_best_config_path
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
            "precision_at_0_2pct": results["top_k_performance"]["top_0_2pct"]["precision_at_k"],
            "recall_at_0_2pct": results["top_k_performance"]["top_0_2pct"]["recall_at_k"],
        }
        all_results.append(row)
            
        if results["top_k_performance"]["top_0_2pct"]["precision_at_k"] > best_precision_at_k:
            best_precision_at_k = results["top_k_performance"]["top_0_2pct"]["precision_at_k"]
            best_config = config
            best_results = results

    print(f"Best config: {best_config} with precision@0.2%: {best_precision_at_k:.4f}")
    print("All results:")
    for row in all_results:
        print(row)
    save_results(best_results, custom_path="rf_tuned_metrics.json")

    # Save best config
    config_path = get_best_config_path()
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
        
    
if __name__ == "__main__":
    main()