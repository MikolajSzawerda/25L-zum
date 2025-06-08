#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from config import OUTPUT_DATA_DIR
from scipy import stats

def load_results():
    """Load dimensionality study results"""
    results_path = OUTPUT_DATA_DIR / "experiments" / "dimensionality_study_results.csv"
    df = pd.read_csv(results_path)
    
    # Extract method type for grouping
    df['method_type'] = df['dimensionality_method'].str.extract(r'([A-Za-z]+)')[0]
    df['method_type'] = df['method_type'].replace('SelectKBest', 'Feature Selection')
    df['method_type'] = df['method_type'].replace('baseline', 'No Reduction')
    
    # Calculate additional metrics
    df['reduction_ratio'] = df['n_components'] / df['original_features']
    df['efficiency'] = df['test_f1_macro'] / df['training_time']
    df['cv_test_gap'] = abs(df['cv_test_f1_macro'] - df['test_f1_macro'])
    
    return df

def create_performance_summary(df):
    """Create comprehensive performance summary"""
    print("DIMENSIONALITY REDUCTION STUDY - PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Filter out NaN values for analysis
    df_clean = df.dropna(subset=['test_f1_macro'])
    
    # Overall statistics
    print(f"\nOVERALL STATISTICS:")
    print(f"Total methods tested: {len(df_clean)}")
    print(f"Original feature count: {df_clean['original_features'].iloc[0]:,}")
    print(f"Best F1 Score: {df_clean['test_f1_macro'].max():.4f}")
    print(f"Worst F1 Score: {df_clean['test_f1_macro'].min():.4f}")
    print(f"F1 Score Range: {df_clean['test_f1_macro'].max() - df_clean['test_f1_macro'].min():.4f}")
    print(f"Average F1 Score: {df_clean['test_f1_macro'].mean():.4f} ± {df_clean['test_f1_macro'].std():.4f}")
    
    # Training time statistics
    print(f"\nTRAINING TIME STATISTICS:")
    print(f"Fastest training: {df_clean['training_time'].min():.2f}s")
    print(f"Slowest training: {df_clean['training_time'].max():.2f}s")
    print(f"Average training time: {df_clean['training_time'].mean():.2f}s ± {df_clean['training_time'].std():.2f}s")
    
    return df_clean

def create_method_rankings(df_clean):
    """Create rankings by different criteria"""
    print(f"\nMETHOD RANKINGS:")
    print("-" * 40)
    
    # Rank by test F1 score
    f1_ranking = df_clean.sort_values('test_f1_macro', ascending=False)[
        ['dimensionality_method', 'test_f1_macro', 'n_components', 'training_time']
    ].reset_index(drop=True)
    f1_ranking.index += 1
    
    print("\n1. RANKING BY TEST F1 SCORE:")
    print(f1_ranking.to_string(formatters={
        'test_f1_macro': '{:.4f}'.format,
        'training_time': '{:.2f}s'.format
    }))
    
    # Rank by efficiency (F1/time)
    efficiency_ranking = df_clean.sort_values('efficiency', ascending=False)[
        ['dimensionality_method', 'efficiency', 'test_f1_macro', 'training_time']
    ].reset_index(drop=True)
    efficiency_ranking.index += 1
    
    print("\n2. RANKING BY EFFICIENCY (F1 Score / Training Time):")
    print(efficiency_ranking.to_string(formatters={
        'efficiency': '{:.4f}'.format,
        'test_f1_macro': '{:.4f}'.format,
        'training_time': '{:.2f}s'.format
    }))
    
    # Rank by generalization (smallest CV-test gap)
    generalization_ranking = df_clean.sort_values('cv_test_gap', ascending=True)[
        ['dimensionality_method', 'cv_test_gap', 'cv_test_f1_macro', 'test_f1_macro']
    ].reset_index(drop=True)
    generalization_ranking.index += 1
    
    print("\n3. RANKING BY GENERALIZATION (Smallest CV-Test Gap):")
    print(generalization_ranking.to_string(formatters={
        'cv_test_gap': '{:.4f}'.format,
        'cv_test_f1_macro': '{:.4f}'.format,
        'test_f1_macro': '{:.4f}'.format
    }))
    
    return f1_ranking, efficiency_ranking, generalization_ranking

def analyze_method_types(df_clean):
    """Analyze performance by method type"""
    print(f"\nMETHOD TYPE ANALYSIS:")
    print("-" * 40)
    
    method_analysis = df_clean.groupby('method_type').agg({
        'test_f1_macro': ['count', 'mean', 'std', 'min', 'max'],
        'training_time': ['mean', 'std'],
        'efficiency': ['mean', 'std'],
        'cv_test_gap': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    method_analysis.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in method_analysis.columns]
    method_analysis = method_analysis.rename(columns={
        'count_test_f1_macro': 'count',
        'mean_test_f1_macro': 'f1_mean',
        'std_test_f1_macro': 'f1_std',
        'min_test_f1_macro': 'f1_min',
        'max_test_f1_macro': 'f1_max',
        'mean_training_time': 'time_mean',
        'std_training_time': 'time_std',
        'mean_efficiency': 'efficiency_mean',
        'std_efficiency': 'efficiency_std',
        'mean_cv_test_gap': 'gap_mean',
        'std_cv_test_gap': 'gap_std'
    })
    
    print("\nPERFORMANCE BY METHOD TYPE:")
    print(method_analysis.to_string())
    
    # Statistical significance testing between method types
    print(f"\nSTATISTICAL SIGNIFICANCE TESTING:")
    method_types = df_clean['method_type'].unique()
    
    for i, method1 in enumerate(method_types):
        for method2 in method_types[i+1:]:
            group1 = df_clean[df_clean['method_type'] == method1]['test_f1_macro']
            group2 = df_clean[df_clean['method_type'] == method2]['test_f1_macro']
            
            if len(group1) > 1 and len(group2) > 1:
                statistic, p_value = stats.ttest_ind(group1, group2)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"{method1} vs {method2}: t={statistic:.3f}, p={p_value:.4f} {significance}")
    
    return method_analysis

def analyze_dimensionality_impact(df_clean):
    """Analyze the impact of dimensionality reduction"""
    print(f"\nDIMENSIONALITY REDUCTION IMPACT ANALYSIS:")
    print("-" * 50)
    
    # Compare baseline vs dimensionality reduction methods
    baseline = df_clean[df_clean['dimensionality_method'] == 'baseline']
    reduced = df_clean[df_clean['dimensionality_method'] != 'baseline']
    
    if not baseline.empty and not reduced.empty:
        baseline_f1 = baseline['test_f1_macro'].iloc[0]
        baseline_time = baseline['training_time'].iloc[0]
        
        print(f"BASELINE PERFORMANCE:")
        print(f"F1 Score: {baseline_f1:.4f}")
        print(f"Training Time: {baseline_time:.2f}s")
        print(f"Components: {baseline['n_components'].iloc[0]:,}")
        
        print(f"\nDIMENSIONALITY REDUCTION METHODS:")
        print(f"Best F1 Score: {reduced['test_f1_macro'].max():.4f}")
        print(f"Average F1 Score: {reduced['test_f1_macro'].mean():.4f}")
        print(f"F1 Score vs Baseline: {reduced['test_f1_macro'].max() - baseline_f1:+.4f}")
        
        # Methods that outperform baseline
        better_than_baseline = reduced[reduced['test_f1_macro'] > baseline_f1]
        print(f"\nMethods outperforming baseline: {len(better_than_baseline)}/{len(reduced)}")
        
        if not better_than_baseline.empty:
            print("Methods better than baseline:")
            for _, row in better_than_baseline.iterrows():
                improvement = row['test_f1_macro'] - baseline_f1
                time_ratio = row['training_time'] / baseline_time
                print(f"  {row['dimensionality_method']}: +{improvement:.4f} F1, {time_ratio:.2f}x time")
    
    # Analyze component count vs performance
    print(f"\nCOMPONENT COUNT ANALYSIS:")
    component_analysis = reduced.groupby('n_components').agg({
        'test_f1_macro': ['count', 'mean', 'std'],
        'training_time': 'mean'
    }).round(4)
    
    component_analysis.columns = ['count', 'f1_mean', 'f1_std', 'time_mean']
    print(component_analysis.to_string())

def create_detailed_insights(df_clean):
    """Generate detailed insights and recommendations"""
    print(f"\nKEY INSIGHTS AND RECOMMENDATIONS:")
    print("=" * 50)
    
    # Best overall method
    best_method = df_clean.loc[df_clean['test_f1_macro'].idxmax()]
    print(f"\n1. BEST OVERALL PERFORMANCE:")
    print(f"   Method: {best_method['dimensionality_method']}")
    print(f"   F1 Score: {best_method['test_f1_macro']:.4f}")
    print(f"   Components: {best_method['n_components']:,} ({best_method['reduction_ratio']:.1%} of original)")
    print(f"   Training Time: {best_method['training_time']:.2f}s")
    
    # Most efficient method
    most_efficient = df_clean.loc[df_clean['efficiency'].idxmax()]
    print(f"\n2. MOST EFFICIENT METHOD:")
    print(f"   Method: {most_efficient['dimensionality_method']}")
    print(f"   Efficiency: {most_efficient['efficiency']:.4f} F1/second")
    print(f"   F1 Score: {most_efficient['test_f1_macro']:.4f}")
    print(f"   Training Time: {most_efficient['training_time']:.2f}s")
    
    # Best generalization
    best_generalization = df_clean.loc[df_clean['cv_test_gap'].idxmin()]
    print(f"\n3. BEST GENERALIZATION:")
    print(f"   Method: {best_generalization['dimensionality_method']}")
    print(f"   CV-Test Gap: {best_generalization['cv_test_gap']:.4f}")
    print(f"   CV F1: {best_generalization['cv_test_f1_macro']:.4f}")
    print(f"   Test F1: {best_generalization['test_f1_macro']:.4f}")
    
    # Method type recommendations
    method_performance = df_clean.groupby('method_type')['test_f1_macro'].mean().sort_values(ascending=False)
    print(f"\n4. METHOD TYPE RANKING:")
    for i, (method_type, avg_f1) in enumerate(method_performance.items(), 1):
        count = len(df_clean[df_clean['method_type'] == method_type])
        print(f"   {i}. {method_type}: {avg_f1:.4f} average F1 ({count} methods)")
    
    # Dimensionality reduction effectiveness
    baseline_f1 = df_clean[df_clean['dimensionality_method'] == 'baseline']['test_f1_macro'].iloc[0]
    reduced_methods = df_clean[df_clean['dimensionality_method'] != 'baseline']
    better_count = len(reduced_methods[reduced_methods['test_f1_macro'] > baseline_f1])
    
    print(f"\n5. DIMENSIONALITY REDUCTION EFFECTIVENESS:")
    print(f"   {better_count}/{len(reduced_methods)} methods outperform baseline")
    print(f"   Best improvement: +{(reduced_methods['test_f1_macro'].max() - baseline_f1):.4f} F1")
    print(f"   Average performance: {reduced_methods['test_f1_macro'].mean():.4f} F1")

def save_analysis_results(df_clean, f1_ranking, efficiency_ranking, generalization_ranking, method_analysis):
    """Save analysis results to CSV files"""
    analysis_dir = OUTPUT_DATA_DIR / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Save rankings
    f1_ranking.to_csv(analysis_dir / "dimensionality_f1_ranking.csv", index=True)
    efficiency_ranking.to_csv(analysis_dir / "dimensionality_efficiency_ranking.csv", index=True)
    generalization_ranking.to_csv(analysis_dir / "dimensionality_generalization_ranking.csv", index=True)
    
    # Save method type analysis
    method_analysis.to_csv(analysis_dir / "dimensionality_method_analysis.csv", index=True)
    
    # Save detailed results with calculated metrics
    detailed_results = df_clean[[
        'dimensionality_method', 'method_type', 'n_components', 'reduction_ratio',
        'test_f1_macro', 'test_accuracy', 'training_time', 'efficiency',
        'cv_test_f1_macro', 'cv_test_gap'
    ]].sort_values('test_f1_macro', ascending=False)
    
    detailed_results.to_csv(analysis_dir / "dimensionality_detailed_analysis.csv", index=False)
    
    print(f"\nAnalysis files saved to: {analysis_dir}")
    print("- dimensionality_f1_ranking.csv")
    print("- dimensionality_efficiency_ranking.csv")
    print("- dimensionality_generalization_ranking.csv")
    print("- dimensionality_method_analysis.csv")
    print("- dimensionality_detailed_analysis.csv")

def main():
    # Load and analyze results
    df = load_results()
    df_clean = create_performance_summary(df)
    
    # Create rankings and analysis
    f1_ranking, efficiency_ranking, generalization_ranking = create_method_rankings(df_clean)
    method_analysis = analyze_method_types(df_clean)
    analyze_dimensionality_impact(df_clean)
    create_detailed_insights(df_clean)
    
    # Save results
    save_analysis_results(df_clean, f1_ranking, efficiency_ranking, generalization_ranking, method_analysis)

if __name__ == "__main__":
    main() 