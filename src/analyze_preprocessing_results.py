#!/usr/bin/env python3
import pandas as pd
import numpy as np
from config import OUTPUT_DATA_DIR

def load_results():
    results_path = OUTPUT_DATA_DIR / "experiments" / "preprocessing_study_results.csv"
    return pd.read_csv(results_path)

def create_performance_summary(df):
    # Focus on test set metrics for true generalization performance
    metrics = ['test_f1_macro', 'test_accuracy', 'test_precision_macro', 'test_recall_macro']
    metric_names = ['Test F1', 'Test Accuracy', 'Test Precision', 'Test Recall']
    
    summary_data = []
    
    for metric, name in zip(metrics, metric_names):
        best_idx = df[metric].idxmax()
        best_row = df.loc[best_idx]
        worst_idx = df[metric].idxmin()
        worst_row = df.loc[worst_idx]
        
        summary_data.append({
            'Metric': name,
            'Best_Combination': f"{best_row['preprocessing']}_{best_row['vectorization']}",
            'Best_Score': f"{best_row[metric]:.4f}",
            'Worst_Combination': f"{worst_row['preprocessing']}_{worst_row['vectorization']}",
            'Worst_Score': f"{worst_row[metric]:.4f}",
            'Difference': f"{best_row[metric] - worst_row[metric]:.4f}"
        })
    
    return pd.DataFrame(summary_data)

def create_cv_vs_test_analysis(df):
    """Analyze the difference between CV and test performance"""
    cv_test_pairs = [
        ('cv_test_f1_macro', 'test_f1_macro', 'F1'),
        ('cv_test_accuracy', 'test_accuracy', 'Accuracy'),
        ('cv_test_precision_macro', 'test_precision_macro', 'Precision'),
        ('cv_test_recall_macro', 'test_recall_macro', 'Recall')
    ]
    
    analysis_data = []
    
    for cv_metric, test_metric, name in cv_test_pairs:
        df_copy = df.copy()
        df_copy['difference'] = df_copy[cv_metric] - df_copy[test_metric]
        df_copy['abs_difference'] = abs(df_copy['difference'])
        
        # Find combinations with largest overestimation (CV >> Test)
        overestimated_idx = df_copy['difference'].idxmax()
        overestimated_row = df_copy.loc[overestimated_idx]
        
        # Find combinations with largest underestimation (CV << Test)
        underestimated_idx = df_copy['difference'].idxmin()
        underestimated_row = df_copy.loc[underestimated_idx]
        
        # Overall statistics
        mean_diff = df_copy['difference'].mean()
        std_diff = df_copy['difference'].std()
        correlation = df_copy[cv_metric].corr(df_copy[test_metric])
        
        analysis_data.append({
            'Metric': name,
            'Mean_CV_Test_Diff': f"{mean_diff:.4f}",
            'Std_CV_Test_Diff': f"{std_diff:.4f}",
            'Correlation': f"{correlation:.4f}",
            'Most_Overestimated': f"{overestimated_row['preprocessing']}_{overestimated_row['vectorization']}",
            'Overestimation': f"{overestimated_row['difference']:.4f}",
            'Most_Underestimated': f"{underestimated_row['preprocessing']}_{underestimated_row['vectorization']}",
            'Underestimation': f"{underestimated_row['difference']:.4f}"
        })
    
    return pd.DataFrame(analysis_data)

def create_ranking_table(df):
    # Use test set metrics for ranking
    metrics = ['test_f1_macro', 'test_accuracy', 'test_precision_macro', 'test_recall_macro']
    
    df_copy = df.copy()
    df_copy['combination'] = df_copy['preprocessing'] + '_' + df_copy['vectorization']
    
    rankings = {}
    for metric in metrics:
        metric_name = metric.replace('test_', '').replace('_macro', '')
        df_sorted = df_copy.sort_values(metric, ascending=False)
        rankings[f'{metric_name}_rank'] = range(1, len(df_sorted) + 1)
        rankings[f'{metric_name}_score'] = df_sorted[metric].values
        if metric == metrics[0]:
            rankings['combination'] = df_sorted['combination'].values
    
    ranking_df = pd.DataFrame(rankings)
    ranking_df['avg_rank'] = ranking_df[['f1_rank', 'accuracy_rank', 'precision_rank', 'recall_rank']].mean(axis=1)
    ranking_df = ranking_df.sort_values('avg_rank')
    
    return ranking_df

def create_method_analysis(df):
    # Analyze both CV and test performance
    preprocessing_analysis = df.groupby('preprocessing').agg({
        'cv_test_f1_macro': ['mean', 'std'],
        'test_f1_macro': ['mean', 'std'],
        'cv_test_accuracy': ['mean', 'std'],
        'test_accuracy': ['mean', 'std'],
        'training_time': ['mean', 'std']
    }).round(4)
    
    vectorization_analysis = df.groupby('vectorization').agg({
        'cv_test_f1_macro': ['mean', 'std'],
        'test_f1_macro': ['mean', 'std'],
        'cv_test_accuracy': ['mean', 'std'],
        'test_accuracy': ['mean', 'std'],
        'training_time': ['mean', 'std']
    }).round(4)
    
    return preprocessing_analysis, vectorization_analysis

def main():
    df = load_results()
    
    output_dir = OUTPUT_DATA_DIR / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    summary = create_performance_summary(df)
    cv_test_analysis = create_cv_vs_test_analysis(df)
    ranking = create_ranking_table(df)
    prep_analysis, vec_analysis = create_method_analysis(df)
    
    summary.to_csv(output_dir / "preprocessing_test_performance_summary.csv", index=False)
    cv_test_analysis.to_csv(output_dir / "cv_vs_test_analysis.csv", index=False)
    ranking.to_csv(output_dir / "preprocessing_test_combination_rankings.csv", index=False)
    prep_analysis.to_csv(output_dir / "preprocessing_method_analysis_updated.csv")
    vec_analysis.to_csv(output_dir / "vectorization_method_analysis_updated.csv")
    
    print("PREPROCESSING STUDY ANALYSIS (Updated with Test Set Evaluation)")
    print("=" * 70)
    print("\n1. TEST SET PERFORMANCE SUMMARY:")
    print(summary.to_string(index=False))
    
    print("\n\n2. CROSS-VALIDATION vs TEST SET ANALYSIS:")
    print(cv_test_analysis.to_string(index=False))
    
    print("\n\n3. TOP 5 COMBINATIONS (by test set average rank):")
    print(ranking.head()[['combination', 'f1_score', 'accuracy_score', 'avg_rank']].to_string(index=False))
    
    print("\n\n4. PREPROCESSING METHOD PERFORMANCE (Test Set):")
    print("Test F1 Score by Preprocessing Method:")
    prep_f1 = df.groupby('preprocessing')['test_f1_macro'].agg(['mean', 'std']).round(4)
    prep_f1_sorted = prep_f1.sort_values('mean', ascending=False)
    print(prep_f1_sorted.to_string())
    
    print("\n\n5. VECTORIZATION METHOD PERFORMANCE (Test Set):")
    print("Test F1 Score by Vectorization Method:")
    vec_f1 = df.groupby('vectorization')['test_f1_macro'].agg(['mean', 'std']).round(4)
    vec_f1_sorted = vec_f1.sort_values('mean', ascending=False)
    print(vec_f1_sorted.to_string())
    
    print("\n\n6. GENERALIZATION GAP ANALYSIS:")
    df_copy = df.copy()
    df_copy['f1_gap'] = df_copy['cv_test_f1_macro'] - df_copy['test_f1_macro']
    df_copy['accuracy_gap'] = df_copy['cv_test_accuracy'] - df_copy['test_accuracy']
    
    print(f"Average F1 generalization gap (CV - Test): {df_copy['f1_gap'].mean():.4f} ± {df_copy['f1_gap'].std():.4f}")
    print(f"Average Accuracy generalization gap (CV - Test): {df_copy['accuracy_gap'].mean():.4f} ± {df_copy['accuracy_gap'].std():.4f}")
    
    largest_gap_idx = df_copy['f1_gap'].idxmax()
    largest_gap_row = df_copy.loc[largest_gap_idx]
    print(f"Largest F1 gap: {largest_gap_row['preprocessing']}_{largest_gap_row['vectorization']} ({largest_gap_row['f1_gap']:.4f})")
    
    print("\n\n7. TRAINING TIME ANALYSIS:")
    time_analysis = df.groupby(['preprocessing', 'vectorization'])['training_time'].mean().round(2)
    fastest = time_analysis.idxmin()
    slowest = time_analysis.idxmax()
    print(f"Fastest combination: {fastest[0]}_{fastest[1]} ({time_analysis.min():.2f}s)")
    print(f"Slowest combination: {slowest[0]}_{slowest[1]} ({time_analysis.max():.2f}s)")
    print(f"Speed difference: {time_analysis.max() - time_analysis.min():.2f}s")

if __name__ == "__main__":
    main() 