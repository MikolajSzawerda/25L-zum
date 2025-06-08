#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from config import RAW_DATA_DIR
from experiment_configs import EXPERIMENT_CONFIG, PREPROCESSING_CONFIGS, VECTORIZER_CONFIGS, DIMENSIONALITY_CONFIGS
from experiment_runner import ExperimentRunner
from preprocessing import TextPreprocessor
from tqdm import tqdm


def load_data():
    data_path = RAW_DATA_DIR / "spam_assassin.csv"
    df = pd.read_csv(data_path)
    return df['mail'].values, df['spam'].values


def get_feature_count(X_train, y_train, preprocessor, vectorizer):
    """Get the number of features after preprocessing and vectorization"""
    temp_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('vectorizer', vectorizer)
    ])
    X_transformed = temp_pipeline.fit_transform(X_train, y_train)
    return X_transformed.shape[1]


def create_adaptive_dimensionality_configs(n_features):
    """Create dimensionality reduction configs adapted to the actual feature count"""
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.feature_selection import SelectKBest, f_classif
    
    configs = {}
    
    # PCA configurations (need to be less than n_features)
    for n_comp in [50, 100, min(200, n_features - 1)]:
        if n_comp < n_features:
            configs[f'PCA_{n_comp}'] = PCA(n_components=n_comp, random_state=42)
    
    # SVD configurations (can handle sparse matrices better)
    for n_comp in [50, 100, min(200, n_features - 1)]:
        if n_comp < n_features:
            configs[f'SVD_{n_comp}'] = TruncatedSVD(n_components=n_comp, random_state=42)
    
    # SelectKBest configurations
    for k in [50, 100, min(200, n_features - 1), min(500, n_features - 1)]:
        if k < n_features:
            configs[f'SelectKBest_{k}'] = SelectKBest(f_classif, k=k)
    
    return configs


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
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Get the actual feature count after vectorization
    n_features = get_feature_count(X_train, y_train, preprocessor, vectorizer)
    print(f"Number of features after vectorization: {n_features}")
    
    # Create adaptive dimensionality reduction configs
    adaptive_configs = create_adaptive_dimensionality_configs(n_features)
    print(f"Created {len(adaptive_configs)} dimensionality reduction configurations")
    
    for dim_name, dim_reducer in tqdm(adaptive_configs.items(), desc="Dimensionality Configs"):
        try:
            if 'PCA' in dim_name:
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('vectorizer', vectorizer),
                    ('scaler', StandardScaler(with_mean=False)),
                    ('dim_reduction', dim_reducer),
                    ('model', base_model)
                ])
            else:
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('vectorizer', vectorizer),
                    ('dim_reduction', dim_reducer),
                    ('model', base_model)
                ])
            
            result = runner.run_experiment(
                X_train, y_train, X_test, y_test, pipeline, None,
                EXPERIMENT_CONFIG['cv'], EXPERIMENT_CONFIG['scoring'],
                'dimensionality_study', dim_name
            )
            result['dimensionality_method'] = dim_name
            result['n_components'] = dim_reducer.n_components if hasattr(dim_reducer, 'n_components') else dim_reducer.k
            result['original_features'] = n_features
            results.append(result)
            
        except Exception as e:
            print(f"Error with {dim_name}: {str(e)}")
            continue
    
    # Baseline without dimensionality reduction
    baseline_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('vectorizer', vectorizer),
        ('model', base_model)
    ])
    
    try:
        baseline_result = runner.run_experiment(
            X_train, y_train, X_test, y_test, baseline_pipeline, None,
            EXPERIMENT_CONFIG['cv'], EXPERIMENT_CONFIG['scoring'],
            'dimensionality_study', 'baseline'
        )
        baseline_result['dimensionality_method'] = 'baseline'
        baseline_result['n_components'] = n_features
        baseline_result['original_features'] = n_features
        results.append(baseline_result)
    except Exception as e:
        print(f"Error with baseline: {str(e)}")
    
    if results:
        runner.save_results(results, 'dimensionality_study_results.csv')
        print(f"Saved {len(results)} experiment results")
    else:
        print("No successful experiments to save")


if __name__ == "__main__":
    main() 