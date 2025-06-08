#!/usr/bin/env python3
import pandas as pd
import numpy as np
from config import OUTPUT_DATA_DIR

def load_results():
    results_path = OUTPUT_DATA_DIR / "experiments" / "algorithm_comparison_results.csv"
    return pd.read_csv(results_path)

def create_performance_summary(df):
    """Create summary of algorithm performance"""
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
            'Best_Algorithm': best_row['model'],
            'Best_Score': f"{best_row[metric]:.4f}",
            'Worst_Algorithm': worst_row['model'],
            'Worst_Score': f"{worst_row[metric]:.4f}",
            'Performance_Gap': f"{best_row[metric] - worst_row[metric]:.4f}"
        })
    
    return pd.DataFrame(summary_data)

def create_generalization_analysis(df):
    """Analyze generalization gap between CV and test performance"""
    cv_test_pairs = [
        ('cv_test_f1_macro', 'test_f1_macro', 'F1'),
        ('cv_test_accuracy', 'test_accuracy', 'Accuracy'),
        ('cv_test_precision_macro', 'test_precision_macro', 'Precision'),
        ('cv_test_recall_macro', 'test_recall_macro', 'Recall')
    ]
    
    analysis_data = []
    
    for cv_metric, test_metric, name in cv_test_pairs:
        df_copy = df.copy()
        df_copy['gap'] = df_copy[cv_metric] - df_copy[test_metric]
        df_copy['abs_gap'] = abs(df_copy['gap'])
        
        # Find algorithm with largest overestimation
        overestimated_idx = df_copy['gap'].idxmax()
        overestimated_row = df_copy.loc[overestimated_idx]
        
        # Find algorithm with largest underestimation
        underestimated_idx = df_copy['gap'].idxmin()
        underestimated_row = df_copy.loc[underestimated_idx]
        
        # Overall statistics
        mean_gap = df_copy['gap'].mean()
        std_gap = df_copy['gap'].std()
        correlation = df_copy[cv_metric].corr(df_copy[test_metric])
        
        analysis_data.append({
            'Metric': name,
            'Mean_Gap_CV_Test': f"{mean_gap:.4f}",
            'Std_Gap': f"{std_gap:.4f}",
            'CV_Test_Correlation': f"{correlation:.4f}",
            'Most_Overestimated': overestimated_row['model'],
            'Overestimation': f"{overestimated_row['gap']:.4f}",
            'Most_Underestimated': underestimated_row['model'],
            'Underestimation': f"{underestimated_row['gap']:.4f}"
        })
    
    return pd.DataFrame(analysis_data)

def create_efficiency_analysis(df):
    """Analyze training and prediction efficiency"""
    efficiency_data = []
    
    for _, row in df.iterrows():
        # Calculate efficiency metrics
        f1_per_training_second = row['test_f1_macro'] / row['training_time']
        f1_per_prediction_second = row['test_f1_macro'] / row['prediction_time']
        
        efficiency_data.append({
            'Algorithm': row['model'],
            'Training_Time': f"{row['training_time']:.2f}s",
            'Prediction_Time': f"{row['prediction_time']:.4f}s",
            'Test_F1': f"{row['test_f1_macro']:.4f}",
            'F1_per_Training_Second': f"{f1_per_training_second:.6f}",
            'F1_per_Prediction_Second': f"{f1_per_prediction_second:.2f}",
            'Speed_Rank_Training': 0,  # Will be filled below
            'Speed_Rank_Prediction': 0,  # Will be filled below
            'Performance_Rank': 0  # Will be filled below
        })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    
    # Add rankings
    efficiency_df['Speed_Rank_Training'] = df['training_time'].rank(ascending=True).astype(int)
    efficiency_df['Speed_Rank_Prediction'] = df['prediction_time'].rank(ascending=True).astype(int)
    efficiency_df['Performance_Rank'] = df['test_f1_macro'].rank(ascending=False).astype(int)
    
    return efficiency_df

def create_statistical_significance_analysis(df):
    """Analyze statistical significance of performance differences"""
    # This is a simplified analysis - in practice you'd want proper statistical tests
    significance_data = []
    
    algorithms = df['model'].tolist()
    
    for i, alg1 in enumerate(algorithms):
        for j, alg2 in enumerate(algorithms):
            if i < j:  # Avoid duplicate comparisons
                row1 = df[df['model'] == alg1].iloc[0]
                row2 = df[df['model'] == alg2].iloc[0]
                
                # Calculate difference and combined uncertainty
                f1_diff = row1['test_f1_macro'] - row2['test_f1_macro']
                combined_std = np.sqrt(row1['cv_test_f1_macro_std']**2 + row2['cv_test_f1_macro_std']**2)
                
                # Simple significance test (difference > 2 * combined std)
                is_significant = abs(f1_diff) > 2 * combined_std
                
                significance_data.append({
                    'Algorithm_1': alg1,
                    'Algorithm_2': alg2,
                    'F1_Difference': f"{f1_diff:.4f}",
                    'Combined_Std': f"{combined_std:.4f}",
                    'Likely_Significant': is_significant,
                    'Better_Algorithm': alg1 if f1_diff > 0 else alg2
                })
    
    return pd.DataFrame(significance_data)

def main():
    df = load_results()
    
    output_dir = OUTPUT_DATA_DIR / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Generate all analyses
    performance_summary = create_performance_summary(df)
    generalization_analysis = create_generalization_analysis(df)
    efficiency_analysis = create_efficiency_analysis(df)
    significance_analysis = create_statistical_significance_analysis(df)
    
    # Save to CSV files
    performance_summary.to_csv(output_dir / "algorithm_performance_summary.csv", index=False)
    generalization_analysis.to_csv(output_dir / "algorithm_generalization_analysis.csv", index=False)
    efficiency_analysis.to_csv(output_dir / "algorithm_efficiency_analysis.csv", index=False)
    significance_analysis.to_csv(output_dir / "algorithm_significance_analysis.csv", index=False)
    
    # Print comprehensive analysis
    print("ALGORITHM COMPARISON ANALYSIS")
    print("=" * 60)
    
    print("\n1. PERFORMANCE SUMMARY:")
    print(performance_summary.to_string(index=False))
    
    print("\n\n2. ALGORITHM RANKINGS:")
    print("Test F1 Score Rankings:")
    f1_rankings = df.sort_values('test_f1_macro', ascending=False)[['model', 'test_f1_macro']]
    for i, (_, row) in enumerate(f1_rankings.iterrows(), 1):
        print(f"  {i}. {row['model']}: {row['test_f1_macro']:.4f}")
    
    print("\n\n3. GENERALIZATION ANALYSIS:")
    print(generalization_analysis.to_string(index=False))
    
    print("\n\n4. EFFICIENCY ANALYSIS:")
    print("Training Time Rankings (fastest to slowest):")
    time_rankings = df.sort_values('training_time')[['model', 'training_time', 'test_f1_macro']]
    for i, (_, row) in enumerate(time_rankings.iterrows(), 1):
        print(f"  {i}. {row['model']}: {row['training_time']:.2f}s (F1: {row['test_f1_macro']:.4f})")
    
    print("\nPrediction Time Rankings (fastest to slowest):")
    pred_time_rankings = df.sort_values('prediction_time')[['model', 'prediction_time', 'test_f1_macro']]
    for i, (_, row) in enumerate(pred_time_rankings.iterrows(), 1):
        print(f"  {i}. {row['model']}: {row['prediction_time']:.4f}s (F1: {row['test_f1_macro']:.4f})")
    
    print("\n\n5. PERFORMANCE vs EFFICIENCY TRADEOFFS:")
    print("F1 Score per Training Second (efficiency metric):")
    efficiency_metric = df.copy()
    efficiency_metric['f1_per_second'] = efficiency_metric['test_f1_macro'] / efficiency_metric['training_time']
    efficiency_rankings = efficiency_metric.sort_values('f1_per_second', ascending=False)[['model', 'f1_per_second', 'test_f1_macro', 'training_time']]
    for i, (_, row) in enumerate(efficiency_rankings.iterrows(), 1):
        print(f"  {i}. {row['model']}: {row['f1_per_second']:.6f} (F1: {row['test_f1_macro']:.4f}, Time: {row['training_time']:.2f}s)")
    
    print("\n\n6. STATISTICAL SIGNIFICANCE ANALYSIS:")
    significant_differences = significance_analysis[significance_analysis['Likely_Significant'] == True]
    if len(significant_differences) > 0:
        print("Likely significant performance differences:")
        for _, row in significant_differences.iterrows():
            print(f"  {row['Algorithm_1']} vs {row['Algorithm_2']}: {row['F1_Difference']} (±{row['Combined_Std']})")
    else:
        print("No clearly significant performance differences detected.")
        print("Note: This is a simplified analysis. Proper statistical testing would require more rigorous methods.")
    
    print("\n\n7. KEY INSIGHTS:")
    best_f1 = df.loc[df['test_f1_macro'].idxmax()]
    fastest_training = df.loc[df['training_time'].idxmin()]
    fastest_prediction = df.loc[df['prediction_time'].idxmin()]
    
    print(f"• Best overall performance: {best_f1['model']} (F1: {best_f1['test_f1_macro']:.4f})")
    print(f"• Fastest training: {fastest_training['model']} ({fastest_training['training_time']:.2f}s)")
    print(f"• Fastest prediction: {fastest_prediction['model']} ({fastest_prediction['prediction_time']:.4f}s)")
    
    # Calculate generalization gaps
    df_gaps = df.copy()
    df_gaps['f1_gap'] = df_gaps['cv_test_f1_macro'] - df_gaps['test_f1_macro']
    best_generalization = df_gaps.loc[df_gaps['f1_gap'].idxmin()]
    worst_generalization = df_gaps.loc[df_gaps['f1_gap'].idxmax()]
    
    print(f"• Best generalization: {best_generalization['model']} (gap: {best_generalization['f1_gap']:.4f})")
    print(f"• Worst generalization: {worst_generalization['model']} (gap: {worst_generalization['f1_gap']:.4f})")
    
    print(f"\nAnalysis files saved to: {output_dir}")

if __name__ == "__main__":
    main() 