#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from config import RAW_DATA_DIR
from experiment_configs import EXPERIMENT_CONFIG, PREPROCESSING_CONFIGS, VECTORIZER_CONFIGS, CLASS_BALANCE_CONFIGS
from experiment_runner import ExperimentRunner
from preprocessing import TextPreprocessor
from tqdm import tqdm

def load_data():
    data_path = RAW_DATA_DIR / "spam_assassin.csv"
    df = pd.read_csv(data_path)
    # Use a smaller subset for faster execution
    return df['mail'].values, df['spam'].values


def create_imbalanced_dataset(X, y, ratio=0.3, random_state=42):
    np.random.seed(random_state)
    classes = np.unique(y)
    majority_class = classes[np.argmax([np.sum(y == c) for c in classes])]
    minority_class = classes[np.argmin([np.sum(y == c) for c in classes])]
    
    majority_indices = np.where(y == majority_class)[0]
    minority_indices = np.where(y == minority_class)[0]
    
    minority_target_size = int(len(majority_indices) * ratio)
    if minority_target_size <= len(minority_indices):
        sampled_minority_indices = np.random.choice(minority_indices, size=minority_target_size, replace=False)
    else:
        sampled_minority_indices = np.random.choice(minority_indices, size=minority_target_size, replace=True)
    
    combined_indices = np.concatenate([majority_indices, sampled_minority_indices])
    np.random.shuffle(combined_indices)
    
    return X[combined_indices], y[combined_indices]


def main():
    X, y = load_data()
    
    runner = ExperimentRunner()
    results = []
    
    preprocessor = TextPreprocessor(**PREPROCESSING_CONFIGS['standard'])
    vectorizer_config = VECTORIZER_CONFIGS['tfidf_bigram']
    vectorizer = vectorizer_config['vectorizer'](**vectorizer_config['params'])
    
    imbalance_ratios = [0.1, 0.3, 0.5, 1.0]
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),  # Reduced for speed
        'SVM': SVC(kernel='linear', probability=True, random_state=42),  # Added probability=True
        'MultinomialNB': MultinomialNB()
    }
    
    for ratio in tqdm(imbalance_ratios, desc="Imbalance Ratios"):
        if ratio == 1.0:
            X_imb, y_imb = X, y
        else:
            X_imb, y_imb = create_imbalanced_dataset(X, y, ratio)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_imb, y_imb, test_size=EXPERIMENT_CONFIG['test_size'], 
            random_state=EXPERIMENT_CONFIG['random_state'], stratify=y_imb
        )
        
        for balance_name, balance_params in CLASS_BALANCE_CONFIGS.items():
            for model_name, base_model in models.items():
                # Skip class_weight for MultinomialNB as it doesn't support it
                if model_name == 'MultinomialNB' and balance_params:
                    continue
                
                try:
                    if balance_params:
                        model = base_model.__class__(**{**base_model.get_params(), **balance_params})
                    else:
                        model = base_model
                    
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('vectorizer', vectorizer),
                        ('model', model)
                    ])
                    
                    result = runner.run_experiment(
                        X_train, y_train, X_test, y_test, pipeline, None,
                        EXPERIMENT_CONFIG['cv'], EXPERIMENT_CONFIG['scoring'],
                        'class_balancing_study', f"{model_name}_{balance_name}_ratio_{ratio}"
                    )
                    result['model_type'] = model_name
                    result['balance_strategy'] = balance_name
                    result['imbalance_ratio'] = ratio
                    result['class_distribution'] = str(Counter(y_train))
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error with {model_name}_{balance_name}_ratio_{ratio}: {str(e)}")
                    continue
    
    if results:
        runner.save_results(results, 'class_balancing_study_results.csv')
        print(f"Saved {len(results)} experiment results")
    else:
        print("No successful experiments to save")


if __name__ == "__main__":
    main() 