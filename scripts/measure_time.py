#!/usr/bin/env python3
import pandas as pd
import numpy as np
import time
import ast
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from config import RAW_DATA_DIR, OUTPUT_DATA_DIR
from spamclassifier.experiment_configs import EXPERIMENT_CONFIG, ALGORITHM_CONFIGS, PREPROCESSING_CONFIGS, VECTORIZER_CONFIGS
from spamclassifier.preprocessing import TextPreprocessor

def load_data():
    """Load the spam dataset"""
    data_path = RAW_DATA_DIR / "spam_assassin.csv"
    df = pd.read_csv(data_path)
    return df['mail'].values, df['spam'].values

def parse_best_params(params_str):
    """Parse best parameters string back to dictionary"""
    try:
        return ast.literal_eval(params_str)
    except:
        return {}

def create_pipeline_with_params(model_name, best_params):
    """Create pipeline with best parameters"""
    # Get base model configuration
    algo_config = ALGORITHM_CONFIGS[model_name]
    model = algo_config['model']
    
    # Set parameters on the model
    for param, value in best_params.items():
        if param.startswith('model__'):
            param_name = param.replace('model__', '')
            setattr(model, param_name, value)
    
    # Create pipeline
    preprocessor = TextPreprocessor(**PREPROCESSING_CONFIGS['standard'])
    vectorizer_config = VECTORIZER_CONFIGS['tfidf_bigram']
    vectorizer = vectorizer_config['vectorizer'](**vectorizer_config['params'])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('vectorizer', vectorizer),
        ('model', model)
    ])
    
    return pipeline

def measure_training_time(pipeline, X_train, y_train, cv):
    """Measure actual training time with cross-validation"""
    start_time = time.time()
    
    # Run cross-validation to get realistic training time
    cv_results = cross_validate(
        pipeline, X_train, y_train, 
        cv=cv, 
        scoring='f1_macro',
        return_train_score=False,
        n_jobs=1,  # Use single job to get accurate timing
        verbose=3
    )
    
    training_time = time.time() - start_time
    return training_time

def main():
    # Load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=EXPERIMENT_CONFIG['test_size'], 
        random_state=EXPERIMENT_CONFIG['random_state'], stratify=y
    )
    
    # Load results
    results_path = OUTPUT_DATA_DIR / "experiments" / "algorithm_comparison_results.csv"
    df = pd.read_csv(results_path)
    
    print("Measuring actual training times...")
    
    # Measure training time for each model
    for idx, row in df.iterrows():
        model_name = row['model']
        best_params = parse_best_params(row['best_params'])
        
        print(f"Measuring training time for {model_name}...")
        
        try:
            # Create pipeline with best parameters
            pipeline = create_pipeline_with_params(model_name, best_params)
            
            # Measure training time
            actual_training_time = measure_training_time(
                pipeline, X_train, y_train, EXPERIMENT_CONFIG['cv']
            )
            
            # Update the dataframe
            df.at[idx, 'training_time_best'] = actual_training_time
            
            print(f"  {model_name}: {actual_training_time:.2f} seconds")
            
        except Exception as e:
            print(f"  Error measuring {model_name}: {e}")
    
    # Save updated results
    df.to_csv(results_path, index=False)
    print(f"\nUpdated training times saved to: {results_path}")
    
    # Print summary
    print("\nTraining Time Summary:")
    print("-" * 40)
    for _, row in df.iterrows():
        print(f"{row['model']:<15}: {row['training_time_best']:.2f} seconds")

if __name__ == "__main__":
    main() 