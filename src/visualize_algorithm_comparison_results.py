#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from config import OUTPUT_DATA_DIR

def load_results():
    results_path = OUTPUT_DATA_DIR / "experiments" / "algorithm_comparison_results.csv"
    return pd.read_csv(results_path)

def create_performance_comparison(df):
    """Compare algorithms across different metrics"""
    metrics = ['test_f1_macro', 'test_accuracy', 'test_precision_macro', 'test_recall_macro']
    metric_names = ['F1 Score', 'Accuracy', 'Precision', 'Recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        # Sort by performance for better visualization
        df_sorted = df.sort_values(metric, ascending=True)
        
        bars = ax.barh(range(len(df_sorted)), df_sorted[metric], 
                      color=plt.cm.Set3(np.linspace(0, 1, len(df_sorted))))
        
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['model'])
        ax.set_xlabel(f'Test {name}')
        ax.set_title(f'Algorithm Comparison: Test {name}')
        
        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, df_sorted[metric])):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_cv_vs_test_comparison(df):
    """Compare cross-validation vs test set performance"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = [('cv_test_f1_macro', 'test_f1_macro', 'F1 Score'),
               ('cv_test_accuracy', 'test_accuracy', 'Accuracy'),
               ('cv_test_precision_macro', 'test_precision_macro', 'Precision'),
               ('cv_test_recall_macro', 'test_recall_macro', 'Recall')]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    
    for i, (cv_metric, test_metric, name) in enumerate(metrics):
        ax = axes[i]
        
        # Scatter plot with algorithm labels
        for j, (_, row) in enumerate(df.iterrows()):
            ax.scatter(row[cv_metric], row[test_metric], 
                      color=colors[j], s=100, alpha=0.7, label=row['model'])
        
        # Add diagonal line for perfect correlation
        min_val = min(df[cv_metric].min(), df[test_metric].min()) - 0.01
        max_val = max(df[cv_metric].max(), df[test_metric].max()) + 0.01
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=2)
        
        ax.set_xlabel(f'Cross-Validation {name}')
        ax.set_ylabel(f'Test Set {name}')
        ax.set_title(f'CV vs Test: {name}')
        
        # Calculate and display correlation
        corr = df[cv_metric].corr(df[test_metric])
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    return fig

def create_training_time_analysis(df):
    """Analyze training and prediction times"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training time comparison
    df_sorted = df.sort_values('training_time', ascending=True)
    bars1 = ax1.barh(range(len(df_sorted)), df_sorted['training_time'], 
                     color=plt.cm.viridis(np.linspace(0, 1, len(df_sorted))))
    
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(df_sorted['model'])
    ax1.set_xlabel('Training Time (seconds)')
    ax1.set_title('Training Time by Algorithm')
    ax1.set_xscale('log')  # Log scale for better visualization
    
    # Add value labels
    for bar, value in zip(bars1, df_sorted['training_time']):
        ax1.text(value * 1.1, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}s', va='center', fontsize=9)
    
    # Prediction time comparison
    df_sorted2 = df.sort_values('prediction_time', ascending=True)
    bars2 = ax2.barh(range(len(df_sorted2)), df_sorted2['prediction_time'], 
                     color=plt.cm.plasma(np.linspace(0, 1, len(df_sorted2))))
    
    ax2.set_yticks(range(len(df_sorted2)))
    ax2.set_yticklabels(df_sorted2['model'])
    ax2.set_xlabel('Prediction Time (seconds)')
    ax2.set_title('Prediction Time by Algorithm')
    
    # Add value labels
    for bar, value in zip(bars2, df_sorted2['prediction_time']):
        ax2.text(value + max(df_sorted2['prediction_time']) * 0.05, 
                bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}s', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_performance_vs_time_tradeoff(df):
    """Show performance vs time tradeoffs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(df)))
    
    # F1 Score vs Training Time
    for i, (_, row) in enumerate(df.iterrows()):
        ax1.scatter(row['training_time'], row['test_f1_macro'], 
                   color=colors[i], s=150, alpha=0.7, label=row['model'])
        ax1.annotate(row['model'], (row['training_time'], row['test_f1_macro']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Training Time (seconds)')
    ax1.set_ylabel('Test F1 Score')
    ax1.set_title('Performance vs Training Time Tradeoff')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # F1 Score vs Prediction Time
    for i, (_, row) in enumerate(df.iterrows()):
        ax2.scatter(row['prediction_time'], row['test_f1_macro'], 
                   color=colors[i], s=150, alpha=0.7, label=row['model'])
        ax2.annotate(row['model'], (row['prediction_time'], row['test_f1_macro']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Prediction Time (seconds)')
    ax2.set_ylabel('Test F1 Score')
    ax2.set_title('Performance vs Prediction Time Tradeoff')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_error_bars_comparison(df):
    """Show performance with error bars from cross-validation"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = [('test_f1_macro', 'cv_test_f1_macro_std', 'F1 Score'),
               ('test_accuracy', 'cv_test_accuracy_std', 'Accuracy'),
               ('test_precision_macro', 'cv_test_precision_macro_std', 'Precision'),
               ('test_recall_macro', 'cv_test_recall_macro_std', 'Recall')]
    
    for i, (metric, std_metric, name) in enumerate(metrics):
        ax = axes[i]
        
        df_sorted = df.sort_values(metric, ascending=True)
        
        bars = ax.barh(range(len(df_sorted)), df_sorted[metric], 
                      xerr=df_sorted[std_metric],
                      color=plt.cm.Set3(np.linspace(0, 1, len(df_sorted))),
                      alpha=0.7, capsize=5)
        
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['model'])
        ax.set_xlabel(f'Test {name} (with CV std)')
        ax.set_title(f'{name} with Cross-Validation Uncertainty')
        
        # Add value labels
        for j, (bar, value, std) in enumerate(zip(bars, df_sorted[metric], df_sorted[std_metric])):
            ax.text(value + std + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}±{std:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    return fig

def create_auc_comparison(df):
    """Compare AUC-ROC scores across algorithms"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filter out rows where ROC AUC is None/NaN
    df_roc = df.dropna(subset=['test_roc_auc_ovr'])
    
    if len(df_roc) > 0:
        # Test ROC AUC comparison
        df_sorted = df_roc.sort_values('test_roc_auc_ovr', ascending=True)
        bars1 = ax1.barh(range(len(df_sorted)), df_sorted['test_roc_auc_ovr'], 
                         color=plt.cm.viridis(np.linspace(0, 1, len(df_sorted))))
        
        ax1.set_yticks(range(len(df_sorted)))
        ax1.set_yticklabels(df_sorted['model'])
        ax1.set_xlabel('Test ROC AUC (OvR)')
        ax1.set_title('ROC AUC Comparison Across Algorithms')
        ax1.set_xlim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars1, df_sorted['test_roc_auc_ovr']):
            ax1.text(value + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', fontsize=9)
        
        # CV vs Test ROC AUC comparison (if CV data available)
        df_cv_roc = df_roc.dropna(subset=['cv_test_roc_auc_ovr'])
        if len(df_cv_roc) > 0:
            colors = plt.cm.Set1(np.linspace(0, 1, len(df_cv_roc)))
            
            for i, (_, row) in enumerate(df_cv_roc.iterrows()):
                ax2.scatter(row['cv_test_roc_auc_ovr'], row['test_roc_auc_ovr'], 
                           color=colors[i], s=150, alpha=0.7, label=row['model'])
                ax2.annotate(row['model'], (row['cv_test_roc_auc_ovr'], row['test_roc_auc_ovr']),
                            xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            # Add diagonal line for perfect correlation
            ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2)
            ax2.set_xlabel('Cross-Validation ROC AUC')
            ax2.set_ylabel('Test Set ROC AUC')
            ax2.set_title('CV vs Test ROC AUC Comparison')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            
            # Calculate and display correlation
            corr = df_cv_roc['cv_test_roc_auc_ovr'].corr(df_cv_roc['test_roc_auc_ovr'])
            ax2.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax2.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, 'No CV ROC AUC data available', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title('CV vs Test ROC AUC Comparison')
    else:
        ax1.text(0.5, 0.5, 'No ROC AUC data available', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=12)
        ax1.set_title('ROC AUC Comparison Across Algorithms')
        ax2.text(0.5, 0.5, 'No ROC AUC data available', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        ax2.set_title('CV vs Test ROC AUC Comparison')
    
    plt.tight_layout()
    return fig

def create_comprehensive_auc_analysis(df):
    """Create comprehensive AUC analysis including ROC AUC comparison with other metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Filter data with ROC AUC
    df_roc = df.dropna(subset=['test_roc_auc_ovr'])
    
    if len(df_roc) == 0:
        for ax in axes.flatten():
            ax.text(0.5, 0.5, 'No ROC AUC data available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
        plt.tight_layout()
        return fig
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(df_roc)))
    
    # ROC AUC vs F1 Score
    ax1 = axes[0, 0]
    for i, (_, row) in enumerate(df_roc.iterrows()):
        ax1.scatter(row['test_roc_auc_ovr'], row['test_f1_macro'], 
                   color=colors[i], s=150, alpha=0.7, label=row['model'])
        ax1.annotate(row['model'], (row['test_roc_auc_ovr'], row['test_f1_macro']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Test ROC AUC')
    ax1.set_ylabel('Test F1 Score')
    ax1.set_title('ROC AUC vs F1 Score')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # ROC AUC vs Precision
    ax2 = axes[0, 1]
    for i, (_, row) in enumerate(df_roc.iterrows()):
        ax2.scatter(row['test_roc_auc_ovr'], row['test_precision_macro'], 
                   color=colors[i], s=150, alpha=0.7)
        ax2.annotate(row['model'], (row['test_roc_auc_ovr'], row['test_precision_macro']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Test ROC AUC')
    ax2.set_ylabel('Test Precision')
    ax2.set_title('ROC AUC vs Precision')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    # ROC AUC vs Recall
    ax3 = axes[1, 0]
    for i, (_, row) in enumerate(df_roc.iterrows()):
        ax3.scatter(row['test_roc_auc_ovr'], row['test_recall_macro'], 
                   color=colors[i], s=150, alpha=0.7)
        ax3.annotate(row['model'], (row['test_roc_auc_ovr'], row['test_recall_macro']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel('Test ROC AUC')
    ax3.set_ylabel('Test Recall')
    ax3.set_title('ROC AUC vs Recall')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    
    # ROC AUC with error bars (if CV data available)
    ax4 = axes[1, 1]
    df_cv_roc = df_roc.dropna(subset=['cv_test_roc_auc_ovr', 'cv_test_roc_auc_ovr_std'])
    
    if len(df_cv_roc) > 0:
        df_sorted = df_cv_roc.sort_values('test_roc_auc_ovr', ascending=True)
        bars = ax4.barh(range(len(df_sorted)), df_sorted['test_roc_auc_ovr'], 
                       xerr=df_sorted['cv_test_roc_auc_ovr_std'],
                       color=plt.cm.Set3(np.linspace(0, 1, len(df_sorted))),
                       alpha=0.7, capsize=5)
        
        ax4.set_yticks(range(len(df_sorted)))
        ax4.set_yticklabels(df_sorted['model'])
        ax4.set_xlabel('Test ROC AUC (with CV std)')
        ax4.set_title('ROC AUC with Cross-Validation Uncertainty')
        ax4.set_xlim(0, 1)
        
        # Add value labels
        for j, (bar, value, std) in enumerate(zip(bars, df_sorted['test_roc_auc_ovr'], df_sorted['cv_test_roc_auc_ovr_std'])):
            ax4.text(value + std + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}±{std:.3f}', va='center', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No CV ROC AUC std data available', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        ax4.set_title('ROC AUC with Cross-Validation Uncertainty')
    
    plt.tight_layout()
    return fig

def main():
    df = load_results()
    
    output_dir = OUTPUT_DATA_DIR / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Create all visualizations
    fig1 = create_performance_comparison(df)
    fig1.savefig(output_dir / "algorithm_performance_comparison.png", dpi=300, bbox_inches='tight')
    
    fig2 = create_cv_vs_test_comparison(df)
    fig2.savefig(output_dir / "algorithm_cv_vs_test_comparison.png", dpi=300, bbox_inches='tight')
    
    fig3 = create_training_time_analysis(df)
    fig3.savefig(output_dir / "algorithm_training_time_analysis.png", dpi=300, bbox_inches='tight')
    
    fig4 = create_performance_vs_time_tradeoff(df)
    fig4.savefig(output_dir / "algorithm_performance_vs_time_tradeoff.png", dpi=300, bbox_inches='tight')
    
    fig5 = create_error_bars_comparison(df)
    fig5.savefig(output_dir / "algorithm_performance_with_uncertainty.png", dpi=300, bbox_inches='tight')
    
    # New AUC visualizations
    fig6 = create_auc_comparison(df)
    fig6.savefig(output_dir / "algorithm_auc_comparison.png", dpi=300, bbox_inches='tight')
    
    fig7 = create_comprehensive_auc_analysis(df)
    fig7.savefig(output_dir / "algorithm_comprehensive_auc_analysis.png", dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    print("Algorithm comparison visualizations saved to:", output_dir)
    print("Generated files:")
    print("- algorithm_performance_comparison.png")
    print("- algorithm_cv_vs_test_comparison.png") 
    print("- algorithm_training_time_analysis.png")
    print("- algorithm_performance_vs_time_tradeoff.png")
    print("- algorithm_performance_with_uncertainty.png")
    print("- algorithm_auc_comparison.png")
    print("- algorithm_comprehensive_auc_analysis.png")
    
    # Note about ROC and PR curves
    print("\nNote: For actual ROC and PR curves, predicted probabilities need to be saved.")
    print("Consider modifying the experiment runner to save predictions for curve generation.")
    print("Alternatively, run 'python generate_roc_pr_curves.py' to generate ROC and PR curves")
    print("using the saved trained models from algorithm comparison experiments.")

if __name__ == "__main__":
    main() 