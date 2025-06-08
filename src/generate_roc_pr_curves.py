#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from sklearn.preprocessing import label_binarize
import joblib
from pathlib import Path
from config import RAW_DATA_DIR, OUTPUT_DATA_DIR, PROJ_ROOT
from experiment_configs import EXPERIMENT_CONFIG, PREPROCESSING_CONFIGS, VECTORIZER_CONFIGS
from preprocessing import TextPreprocessor
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the spam dataset"""
    data_path = RAW_DATA_DIR / "spam_assassin.csv"
    df = pd.read_csv(data_path)
    return df['mail'].values, df['spam'].values

def load_trained_models():
    """Load all trained models from the algorithm comparison experiment"""
    models_dir = PROJ_ROOT / "models"
    model_files = list(models_dir.glob("algorithm_comparison_*.joblib"))
    
    models = {}
    for model_file in model_files:
        model_name = model_file.stem.replace("algorithm_comparison_", "")
        try:
            model = joblib.load(model_file)
            models[model_name] = model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
    
    return models

def create_roc_curves(models, X_test, y_test, classes):
    """Create ROC curves for all models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # For binary classification
    if len(classes) == 2:
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
                elif hasattr(model, 'decision_function'):
                    y_proba = model.decision_function(X_test)
                else:
                    print(f"Model {model_name} doesn't support probability prediction")
                    continue
                
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                ax1.plot(fpr, tpr, linewidth=2, 
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                print(f"Error generating ROC curve for {model_name}: {e}")
        
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves - Binary Classification')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
    
    # For multiclass (if applicable)
    else:
        # Binarize the output for multiclass ROC
        y_test_bin = label_binarize(y_test, classes=classes)
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                else:
                    print(f"Model {model_name} doesn't support probability prediction for multiclass")
                    continue
                
                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(len(classes)):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                
                ax2.plot(fpr["micro"], tpr["micro"], linewidth=2,
                        label=f'{model_name} (micro-avg AUC = {roc_auc["micro"]:.3f})')
                
            except Exception as e:
                print(f"Error generating multiclass ROC curve for {model_name}: {e}")
        
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves - Multiclass (Micro-average)')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_pr_curves(models, X_test, y_test, classes):
    """Create Precision-Recall curves for all models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # For binary classification
    if len(classes) == 2:
        baseline_precision = np.sum(y_test == 1) / len(y_test)  # Proportion of positive class
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
                elif hasattr(model, 'decision_function'):
                    y_proba = model.decision_function(X_test)
                else:
                    print(f"Model {model_name} doesn't support probability prediction")
                    continue
                
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                avg_precision = average_precision_score(y_test, y_proba)
                
                ax1.plot(recall, precision, linewidth=2,
                        label=f'{model_name} (AP = {avg_precision:.3f})')
            except Exception as e:
                print(f"Error generating PR curve for {model_name}: {e}")
        
        ax1.axhline(y=baseline_precision, color='k', linestyle='--', linewidth=2,
                   label=f'Baseline (AP = {baseline_precision:.3f})')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Curves - Binary Classification')
        ax1.legend(loc='lower left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        ax1.set_xlim([0, 1.05])
    
    # For multiclass (if applicable)
    else:
        y_test_bin = label_binarize(y_test, classes=classes)
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                else:
                    print(f"Model {model_name} doesn't support probability prediction for multiclass")
                    continue
                
                # Compute PR curve for each class and micro-average
                precision = dict()
                recall = dict()
                avg_precision = dict()
                
                for i in range(len(classes)):
                    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
                    avg_precision[i] = average_precision_score(y_test_bin[:, i], y_proba[:, i])
                
                # Compute micro-average
                precision["micro"], recall["micro"], _ = precision_recall_curve(
                    y_test_bin.ravel(), y_proba.ravel())
                avg_precision["micro"] = average_precision_score(y_test_bin, y_proba, average="micro")
                
                ax2.plot(recall["micro"], precision["micro"], linewidth=2,
                        label=f'{model_name} (micro-avg AP = {avg_precision["micro"]:.3f})')
                
            except Exception as e:
                print(f"Error generating multiclass PR curve for {model_name}: {e}")
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves - Multiclass (Micro-average)')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        ax2.set_xlim([0, 1.05])
    
    plt.tight_layout()
    return fig

def create_combined_roc_pr_curves(models, X_test, y_test, classes):
    """Create combined ROC and PR curves in a single figure"""
    n_models = len(models)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC curves
    ax_roc = axes[0, 0]
    ax_pr = axes[0, 1]
    
    # Model comparison table
    ax_table = axes[1, :]
    
    model_metrics = []
    
    if len(classes) == 2:
        baseline_precision = np.sum(y_test == 1) / len(y_test)
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_proba = model.decision_function(X_test)
                else:
                    continue
                
                # ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
                
                # PR curve
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                avg_precision = average_precision_score(y_test, y_proba)
                ax_pr.plot(recall, precision, linewidth=2, label=f'{model_name} (AP = {avg_precision:.3f})')
                
                model_metrics.append({
                    'Model': model_name,
                    'ROC AUC': f'{roc_auc:.3f}',
                    'PR AUC (AP)': f'{avg_precision:.3f}'
                })
                
            except Exception as e:
                print(f"Error processing {model_name}: {e}")
        
        # Add baselines
        ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        ax_pr.axhline(y=baseline_precision, color='k', linestyle='--', linewidth=2,
                     label=f'Baseline (AP = {baseline_precision:.3f})')
    
    # Configure ROC plot
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves Comparison')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, alpha=0.3)
    
    # Configure PR plot
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curves Comparison')
    ax_pr.legend(loc='lower left')
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_ylim([0, 1.05])
    ax_pr.set_xlim([0, 1.05])
    
    # Create metrics table
    if model_metrics:
        df_metrics = pd.DataFrame(model_metrics)
        ax_table[0].axis('tight')
        ax_table[0].axis('off')
        ax_table[1].axis('tight')
        ax_table[1].axis('off')
        
        table = ax_table[0].table(cellText=df_metrics.values,
                                 colLabels=df_metrics.columns,
                                 cellLoc='center',
                                 loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax_table[0].set_title('Model Performance Metrics', pad=20)
    
    plt.tight_layout()
    return fig

def main():
    # Load data
    X, y = load_data()
    
    # Use same train-test split as in experiments
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=EXPERIMENT_CONFIG['test_size'], 
        random_state=EXPERIMENT_CONFIG['random_state'], stratify=y
    )
    
    # Load trained models
    models = load_trained_models()
    
    if not models:
        print("No trained models found. Please run algorithm comparison experiment first.")
        return
    
    print(f"Found {len(models)} trained models: {list(models.keys())}")
    
    # Get unique classes
    classes = np.unique(y)
    print(f"Classes: {classes}")
    
    # Create output directory
    output_dir = OUTPUT_DATA_DIR / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Generate ROC curves
    print("Generating ROC curves...")
    fig_roc = create_roc_curves(models, X_test, y_test, classes)
    fig_roc.savefig(output_dir / "algorithm_roc_curves.png", dpi=300, bbox_inches='tight')
    
    # Generate PR curves
    print("Generating Precision-Recall curves...")
    fig_pr = create_pr_curves(models, X_test, y_test, classes)
    fig_pr.savefig(output_dir / "algorithm_pr_curves.png", dpi=300, bbox_inches='tight')
    
    # Generate combined curves
    print("Generating combined ROC and PR curves...")
    fig_combined = create_combined_roc_pr_curves(models, X_test, y_test, classes)
    fig_combined.savefig(output_dir / "algorithm_combined_roc_pr_curves.png", dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    print(f"\nROC and PR curve visualizations saved to: {output_dir}")
    print("Generated files:")
    print("- algorithm_roc_curves.png")
    print("- algorithm_pr_curves.png")
    print("- algorithm_combined_roc_pr_curves.png")

if __name__ == "__main__":
    main() 