#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support
import joblib
from pathlib import Path
from config import RAW_DATA_DIR, OUTPUT_DATA_DIR, PROJ_ROOT
from spamclassifier.experiment_configs import EXPERIMENT_CONFIG

def load_data():
    """Load the spam dataset with categories"""
    data_path = RAW_DATA_DIR / "spam_assassin.csv"
    df = pd.read_csv(data_path)
    return df

def load_trained_models():
    """Load all trained models"""
    models_dir = PROJ_ROOT / "models"
    model_files = list(models_dir.glob("algorithm_comparison_*.joblib"))
    
    models = {}
    for model_file in model_files:
        model_name = model_file.stem.replace("algorithm_comparison_", "")
        try:
            models[model_name] = joblib.load(model_file)
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
    return models

def evaluate_by_category(models, X_test, y_test, categories_test):
    """Evaluate models by spam/ham categories"""
    results = []
    
    # Get unique categories
    unique_categories = np.unique(categories_test)
    print(f"Evaluating on categories: {unique_categories}")
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Overall F1 score
            overall_f1 = f1_score(y_test, y_pred, average='macro')
            print(f"  Overall F1: {overall_f1:.3f}")
            
            results.append({
                'model': model_name,
                'category': 'Overall',
                'f1_score': overall_f1,
                'support': len(y_test)
            })
            
            # F1 score by category
            for category in unique_categories:
                # Get indices for this category
                cat_indices = categories_test == category
                cat_count = cat_indices.sum()
                
                if cat_count > 0:  # Make sure category exists in test set
                    y_true_cat = y_test[cat_indices]
                    y_pred_cat = y_pred[cat_indices]
                    
                    # Count spam/ham in this category
                    spam_count = (y_true_cat == 1).sum()
                    ham_count = (y_true_cat == 0).sum()
                    
                    print(f"  {category}: {cat_count} samples ({spam_count} spam, {ham_count} ham)")
                    
                    # Calculate F1 score for this category
                    if len(np.unique(y_true_cat)) > 1:  # Both classes present
                        f1_cat = f1_score(y_true_cat, y_pred_cat, average='binary')
                        precision, recall, f1_macro, _ = precision_recall_fscore_support(
                            y_true_cat, y_pred_cat, average='macro'
                        )
                        print(f"    F1 (binary): {f1_cat:.3f}, F1 (macro): {f1_macro:.3f}")
                        f1_to_use = f1_cat  # Use binary F1 for consistency
                    else:  # Only one class present
                        # If only one class, check if predictions are correct
                        accuracy = (y_true_cat == y_pred_cat).mean()
                        f1_to_use = accuracy  # Use accuracy as proxy for F1
                        print(f"    Single class accuracy: {accuracy:.3f}")
                    
                    results.append({
                        'model': model_name,
                        'category': category,
                        'f1_score': f1_to_use,
                        'support': cat_count,
                        'spam_count': spam_count,
                        'ham_count': ham_count
                    })
                    
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return pd.DataFrame(results)

def create_category_comparison_plot(results_df):
    """Create visualization comparing F1 scores by category"""
    # Separate spam and ham categories
    spam_categories = results_df[results_df['category'].str.contains('spam', case=False, na=False)]
    ham_categories = results_df[results_df['category'].str.contains('ham', case=False, na=False)]
    overall_results = results_df[results_df['category'] == 'Overall']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall comparison
    if not overall_results.empty:
        sns.barplot(data=overall_results, x='model', y='f1_score', ax=axes[0, 0])
        axes[0, 0].set_title('Overall F1 Score Comparison')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, bar in enumerate(axes[0, 0].patches):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom')
    
    # 2. Spam categories
    if not spam_categories.empty:
        pivot_spam = spam_categories.pivot(index='category', columns='model', values='f1_score')
        sns.heatmap(pivot_spam, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0, 1], vmin=0, vmax=1)
        axes[0, 1].set_title('F1 Scores for Spam Categories')
        axes[0, 1].set_ylabel('Spam Categories')
    
    # 3. Ham categories  
    if not ham_categories.empty:
        pivot_ham = ham_categories.pivot(index='category', columns='model', values='f1_score')
        sns.heatmap(pivot_ham, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[1, 0], vmin=0, vmax=1)
        axes[1, 0].set_title('F1 Scores for Ham Categories')
        axes[1, 0].set_ylabel('Ham Categories')
    
    # 4. Category comparison by model
    category_results = results_df[results_df['category'] != 'Overall']
    if not category_results.empty:
        sns.boxplot(data=category_results, x='model', y='f1_score', ax=axes[1, 1])
        axes[1, 1].set_title('F1 Score Distribution by Model Across Categories')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    return fig

def create_detailed_comparison_plot(results_df):
    """Create detailed comparison plot"""
    # Filter out overall results for detailed view
    category_results = results_df[results_df['category'] != 'Overall']
    
    if category_results.empty:
        return None
    
    # Create pivot table for heatmap
    pivot_df = category_results.pivot(index='category', columns='model', values='f1_score')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Heatmap
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax1, vmin=0, vmax=1)
    ax1.set_title('F1 Scores by Category and Model')
    ax1.set_ylabel('Email Categories')
    ax1.set_xlabel('Models')
    
    # Bar plot comparing models across categories
    category_results_grouped = category_results.groupby('model')['f1_score'].agg(['mean', 'std']).reset_index()
    
    bars = ax2.bar(category_results_grouped['model'], category_results_grouped['mean'], 
                   yerr=category_results_grouped['std'], capsize=5, alpha=0.7)
    ax2.set_title('Average F1 Score Across All Categories')
    ax2.set_ylabel('Average F1 Score')
    ax2.set_xlabel('Models')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, category_results_grouped['mean']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{mean_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def main():
    # Load data
    df = load_data()
    X = df['mail'].values
    y = df['spam'].values
    categories = df['category'].values
    
    print(f"Dataset shape: {df.shape}")
    print(f"Categories: {np.unique(categories)}")
    print(f"Category distribution:")
    for cat in np.unique(categories):
        cat_mask = categories == cat
        spam_in_cat = (y[cat_mask] == 1).sum()
        ham_in_cat = (y[cat_mask] == 0).sum()
        print(f"  {cat}: {cat_mask.sum()} total ({spam_in_cat} spam, {ham_in_cat} ham)")
    
    # Use same train-test split as in experiments
    X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
        X, y, categories, test_size=EXPERIMENT_CONFIG['test_size'], 
        random_state=EXPERIMENT_CONFIG['random_state'], stratify=y
    )
    
    print(f"\nTest set shape: {len(X_test)}")
    print(f"Test set categories: {np.unique(cat_test)}")
    
    # Load trained models
    models = load_trained_models()
    
    if not models:
        print("No trained models found. Please run algorithm comparison experiment first.")
        return
    
    print(f"\nFound {len(models)} trained models: {list(models.keys())}")
    
    # Evaluate models by category
    print("\nEvaluating models by category...")
    results_df = evaluate_by_category(models, X_test, y_test, cat_test)
    
    # Create output directory
    output_dir = OUTPUT_DATA_DIR / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Save results to CSV
    results_df.to_csv(OUTPUT_DATA_DIR / 'experiments' / "category_performance_results.csv", index=False)
    print(f"\nResults saved to: {output_dir / 'category_performance_results.csv'}")
    
    # Create visualizations
    print("Creating category comparison plots...")
    
    # Main comparison plot
    fig1 = create_category_comparison_plot(results_df)
    fig1.savefig(output_dir / "algorithm_category_comparison.png", dpi=300, bbox_inches='tight')
    
    # Detailed comparison plot
    fig2 = create_detailed_comparison_plot(results_df)
    if fig2:
        fig2.savefig(output_dir / "algorithm_detailed_category_comparison.png", dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    # Print summary
    print(f"\nCategory performance analysis saved to: {output_dir}")
    print("Generated files:")
    print("- category_performance_results.csv")
    print("- algorithm_category_comparison.png")
    print("- algorithm_detailed_category_comparison.png")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    
    # Overall performance
    overall_results = results_df[results_df['category'] == 'Overall']
    if not overall_results.empty:
        print("\nOverall F1 Scores:")
        for _, row in overall_results.iterrows():
            print(f"  {row['model']:<15}: {row['f1_score']:.3f}")
    
    # Best performing model per category
    category_results = results_df[results_df['category'] != 'Overall']
    if not category_results.empty:
        print("\nBest performing model per category:")
        for category in category_results['category'].unique():
            cat_data = category_results[category_results['category'] == category]
            best_model = cat_data.loc[cat_data['f1_score'].idxmax()]
            print(f"  {category:<15}: {best_model['model']} (F1: {best_model['f1_score']:.3f})")
    
    # Show detailed results table
    print("\nDetailed Results:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main() 