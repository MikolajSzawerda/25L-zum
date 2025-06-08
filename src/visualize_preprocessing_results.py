#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from config import OUTPUT_DATA_DIR

def load_results():
    results_path = OUTPUT_DATA_DIR / "experiments" / "preprocessing_study_results.csv"
    return pd.read_csv(results_path)

def create_performance_matrices(df):
    # Use test set metrics for true generalization performance
    metrics = ['test_f1_macro', 'test_accuracy', 'test_precision_macro', 'test_recall_macro']
    metric_names = ['Test F1 Score', 'Test Accuracy', 'Test Precision', 'Test Recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        pivot_table = df.pivot(index='preprocessing', columns='vectorization', values=metric)
        
        sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd', 
                   ax=axes[i], cbar_kws={'label': name})
        axes[i].set_title(f'{name} by Preprocessing and Vectorization')
        axes[i].set_xlabel('Vectorization Method')
        axes[i].set_ylabel('Preprocessing Method')
    
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
    
    for i, (cv_metric, test_metric, name) in enumerate(metrics):
        ax = axes[i]
        ax.scatter(df[cv_metric], df[test_metric], alpha=0.7)
        
        # Add diagonal line for perfect correlation
        min_val = min(df[cv_metric].min(), df[test_metric].min())
        max_val = max(df[cv_metric].max(), df[test_metric].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        ax.set_xlabel(f'Cross-Validation {name}')
        ax.set_ylabel(f'Test Set {name}')
        ax.set_title(f'CV vs Test: {name}')
        
        # Calculate correlation
        corr = df[cv_metric].corr(df[test_metric])
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_best_combinations_plot(df):
    # Use test set metrics for true performance
    metrics = ['test_f1_macro', 'test_accuracy', 'test_precision_macro', 'test_recall_macro']
    metric_names = ['F1', 'Accuracy', 'Precision', 'Recall']
    
    best_combinations = []
    for metric in metrics:
        best_idx = df[metric].idxmax()
        best_row = df.loc[best_idx]
        best_combinations.append({
            'metric': metric.replace('test_', ''),
            'preprocessing': best_row['preprocessing'],
            'vectorization': best_row['vectorization'],
            'value': best_row[metric],
            'combination': f"{best_row['preprocessing']}_{best_row['vectorization']}"
        })
    
    best_df = pd.DataFrame(best_combinations)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(best_df)), best_df['value'], 
                  color=['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4'])
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Test Set Performance Score')
    ax.set_title('Best Preprocessing-Vectorization Combinations by Test Set Performance')
    ax.set_xticks(range(len(best_df)))
    ax.set_xticklabels([name.title() for name in metric_names])
    
    for i, (bar, row) in enumerate(zip(bars, best_df.itertuples())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{row.combination}\n{height:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_performance_comparison(df):
    # Use test set metrics
    metrics = ['test_f1_macro', 'test_accuracy', 'test_precision_macro', 'test_recall_macro']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        metric_name = metric.replace('test_', '').replace('_macro', '').title()
        
        preprocessing_means = df.groupby('preprocessing')[metric].mean().sort_values(ascending=False)
        vectorization_means = df.groupby('vectorization')[metric].mean().sort_values(ascending=False)
        
        if i < 2:
            ax = axes[i]
            preprocessing_means.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(f'Test {metric_name} by Preprocessing Method')
            ax.set_ylabel(f'Test {metric_name}')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax = axes[i]
            vectorization_means.plot(kind='bar', ax=ax, color='lightcoral')
            ax.set_title(f'Test {metric_name} by Vectorization Method')
            ax.set_ylabel(f'Test {metric_name}')
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def main():
    df = load_results()
    
    output_dir = OUTPUT_DATA_DIR / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    fig1 = create_performance_matrices(df)
    fig1.savefig(output_dir / "preprocessing_test_performance_matrices.png", dpi=300, bbox_inches='tight')
    
    fig2 = create_cv_vs_test_comparison(df)
    fig2.savefig(output_dir / "cv_vs_test_comparison.png", dpi=300, bbox_inches='tight')
    
    fig3 = create_best_combinations_plot(df)
    fig3.savefig(output_dir / "best_preprocessing_combinations_test.png", dpi=300, bbox_inches='tight')
    
    fig4 = create_performance_comparison(df)
    fig4.savefig(output_dir / "preprocessing_method_comparison_test.png", dpi=300, bbox_inches='tight')
    
    plt.close('all')

if __name__ == "__main__":
    main() 