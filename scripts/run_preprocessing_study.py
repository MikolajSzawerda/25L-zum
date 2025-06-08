#!/usr/bin/env python3
"""
Script to study the impact of different text preprocessing configurations.

This script evaluates how different preprocessing steps affect model performance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from config import INPUT_DATA_DIR, RAW_DATA_DIR, OUTPUT_DATA_DIR
from spamclassifier.experiment_configs import EXPERIMENT_CONFIG, PREPROCESSING_CONFIGS, VECTORIZER_CONFIGS
from spamclassifier.experiment_runner import ExperimentRunner
from spamclassifier.preprocessing import TextPreprocessor
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
    
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    for prep_name, prep_config in tqdm(PREPROCESSING_CONFIGS.items(), desc="Preprocessing Configs"):
        for vec_name, vec_config in tqdm(VECTORIZER_CONFIGS.items(), desc="Vectorizer Configs"):
            preprocessor = TextPreprocessor(**prep_config)
            vectorizer = vec_config['vectorizer'](**vec_config['params'])
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('vectorizer', vectorizer),
                ('model', base_model)
            ])
            
            result = runner.run_experiment(
                X_train, y_train, X_test, y_test, pipeline, None,
                EXPERIMENT_CONFIG['cv'], EXPERIMENT_CONFIG['scoring'],
                'preprocessing_study', f"{prep_name}_{vec_name}"
            )
            result['preprocessing'] = prep_name
            result['vectorization'] = vec_name
            results.append(result)
    
            runner.save_results(results, 'preprocessing_study_results.csv')


if __name__ == "__main__":
    main() 