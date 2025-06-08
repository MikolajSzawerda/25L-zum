#!/usr/bin/env python3
"""
Script to run algorithm comparison experiments.

This script compares the performance of different machine learning algorithms
on the spam classification task including Naive Bayes, SVM, Random Forest, and XGBoost.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from config import INPUT_DATA_DIR, RAW_DATA_DIR
from experiment_configs import EXPERIMENT_CONFIG, ALGORITHM_CONFIGS, PREPROCESSING_CONFIGS, VECTORIZER_CONFIGS
from experiment_runner import ExperimentRunner
from preprocessing import TextPreprocessor
from tqdm import tqdm

def load_data():
    data_path = RAW_DATA_DIR / "spam_assassin.csv"
    df = pd.read_csv(data_path)
    return df['mail'].values, df['spam'].values


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=EXPERIMENT_CONFIG['test_size'], 
        random_state=EXPERIMENT_CONFIG['random_state'], stratify=y
    )
    
    runner = ExperimentRunner()
    results = []
    
    preprocessor = TextPreprocessor(**PREPROCESSING_CONFIGS['standard'])
    vectorizer_config = VECTORIZER_CONFIGS['tfidf_bigram']
    vectorizer = vectorizer_config['vectorizer'](**vectorizer_config['params'])
    
    for algo_name, algo_config in tqdm(ALGORITHM_CONFIGS.items(), desc="Algorithm Configs"):
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('vectorizer', vectorizer),
            ('model', algo_config['model'])
        ])
        
        result = runner.run_experiment(
            X_train, y_train, X_test, y_test, pipeline, algo_config['param_grid'],
            EXPERIMENT_CONFIG['cv'], EXPERIMENT_CONFIG['scoring'],
            'algorithm_comparison', algo_name
        )
        results.append(result)
    
        runner.save_results(results, 'algorithm_comparison_results.csv')


if __name__ == "__main__":
    main() 