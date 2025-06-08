import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)
import joblib
from pathlib import Path
from config import RAW_DATA_DIR, OUTPUT_DATA_DIR, PROJ_ROOT
from spamclassifier.experiment_configs import EXPERIMENT_CONFIG


def load_data():
    """Load the spam dataset"""
    data_path = RAW_DATA_DIR / "spam_assassin.csv"
    df = pd.read_csv(data_path)
    return df["mail"].values, df["spam"].values


def load_trained_models():
    """Load all trained models"""
    models_dir = PROJ_ROOT / "models"
    model_files = list(models_dir.glob("algorithm_comparison_*.joblib"))

    models = {}
    for model_file in model_files:
        model_name = model_file.stem.replace("algorithm_comparison_", "")
        try:
            models[model_name] = joblib.load(model_file)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
    return models


def main():
    # Load data and split
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=EXPERIMENT_CONFIG["test_size"],
        random_state=EXPERIMENT_CONFIG["random_state"],
        stratify=y,
    )

    # Load models
    models = load_trained_models()
    if not models:
        print("No trained models found.")
        return

    # Create output directory
    output_dir = OUTPUT_DATA_DIR / "curve_data"
    output_dir.mkdir(exist_ok=True)

    roc_data = []
    pr_data = []
    metrics_data = []

    for model_name, model in models.items():
        try:
            # Get probabilities
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_test)
            else:
                continue

            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            # PR curve
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            avg_precision = average_precision_score(y_test, y_proba)

            # Store data
            for i in range(len(fpr)):
                roc_data.append({"model": model_name, "fpr": fpr[i], "tpr": tpr[i]})

            for i in range(len(precision)):
                pr_data.append(
                    {
                        "model": model_name,
                        "precision": precision[i],
                        "recall": recall[i],
                    }
                )

            metrics_data.append(
                {"model": model_name, "roc_auc": roc_auc, "pr_auc": avg_precision}
            )

        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    # Save to CSV
    pd.DataFrame(roc_data).to_csv(output_dir / "roc_data.csv", index=False)
    pd.DataFrame(pr_data).to_csv(output_dir / "pr_data.csv", index=False)
    pd.DataFrame(metrics_data).to_csv(output_dir / "metrics_data.csv", index=False)

    print(f"Curve data saved to: {output_dir}")


if __name__ == "__main__":
    main()
