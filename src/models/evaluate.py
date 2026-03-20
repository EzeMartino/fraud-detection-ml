import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

def compute_threshold_metrics(y_true, y_scores) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # precision_recall_curve returns thresholds of length n-1
    precision = precision[:-1]
    recall = recall[:-1]

    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)  # Avoid division by zero  

    metrics_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

    return metrics_df.sort_values(by="threshold").reset_index(drop=True)


def top_k_metrics(y_true, y_scores, top_fraction=0.01) -> dict:
    n = len(y_scores)
    k = max(1, int(n * top_fraction))

    sorted_idx = np.argsort(y_scores)[::-1]
    top_idx = sorted_idx[:k]

    y_true_top = np.array(y_true)[top_idx]

    tp = int(y_true_top.sum())
    precision_at_k = float(tp / k)
    total_positives = int(np.sum(y_true))
    recall_at_k = float(tp / total_positives) if total_positives > 0 else 0.0

    return {
        "top_fraction": top_fraction,
        "k": k,
        "true_positives_in_top_k": tp,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
    }
    
def save_threshold_metrics(metrics_df: pd.DataFrame, output_path: str) -> None:
    metrics_df.to_csv(output_path, index=False)
    print(f"[OK] Saved threshold metrics to {output_path}")


def save_top_k_metrics(results: dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved top-k metrics to {output_path}")