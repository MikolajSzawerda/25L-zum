import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from pathlib import Path
import time
from config import OUTPUT_DATA_DIR, PROJ_ROOT


class ExperimentRunner:
    def __init__(self):
        self.results_dir = OUTPUT_DATA_DIR / "experiments"
        self.models_dir = PROJ_ROOT / "models"
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
    
    def _get_compatible_scoring(self, pipeline, X_sample, y_sample, scoring):
        """Get scoring metrics compatible with the pipeline"""
        compatible_scoring = []
        
        # Fit pipeline on a small sample to test capabilities
        try:
            pipeline.fit(X_sample[:10], y_sample[:10])
        except:
            # If fitting fails, return basic scoring
            return ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        for metric in scoring:
            if metric == 'roc_auc_ovr':
                # Check if pipeline supports predict_proba
                try:
                    if hasattr(pipeline, 'predict_proba'):
                        pipeline.predict_proba(X_sample[:5])
                        compatible_scoring.append(metric)
                except:
                    pass  # Skip this metric
            else:
                compatible_scoring.append(metric)
        
        return compatible_scoring
        
    def run_experiment(self, X_train, y_train, X_test, y_test, pipeline, param_grid, cv, scoring, experiment_name, model_name):
        start_time = time.time()
        
        # Get compatible scoring metrics
        compatible_scoring = self._get_compatible_scoring(pipeline, X_train, y_train, scoring)
        
        if param_grid:
            grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            best_pipeline = pipeline
            best_params = {}
            
        cv_results = cross_validate(best_pipeline, X_train, y_train, cv=cv, scoring=compatible_scoring, n_jobs=-1)
        training_time = time.time() - start_time
        
        best_pipeline.fit(X_train, y_train)
        
        start_time = time.time()
        y_pred = best_pipeline.predict(X_test)
        prediction_time = time.time() - start_time
        
        try:
            y_pred_proba = best_pipeline.predict_proba(X_test)
            test_roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        except:
            test_roc_auc = None
        
        # Initialize result with default values for all possible metrics
        result = {
            'experiment': experiment_name,
            'model': model_name,
            'best_params': str(best_params),
            'training_time': training_time,
            'prediction_time': prediction_time,
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision_macro': precision_score(y_test, y_pred, average='macro'),
            'test_recall_macro': recall_score(y_test, y_pred, average='macro'),
            'test_f1_macro': f1_score(y_test, y_pred, average='macro'),
            'test_roc_auc_ovr': test_roc_auc
        }
        
        # Add CV results for metrics that were actually computed
        for metric, scores in cv_results.items():
            if metric.startswith('test_'):
                result[f'cv_{metric}'] = np.mean(scores)
                result[f'cv_{metric}_std'] = np.std(scores)
        
        # Add default values for missing CV metrics
        default_cv_metrics = ['cv_test_accuracy', 'cv_test_precision_macro', 'cv_test_recall_macro', 'cv_test_f1_macro', 'cv_test_roc_auc_ovr']
        for metric in default_cv_metrics:
            if metric not in result:
                result[metric] = None
                result[f'{metric}_std'] = None
        
        model_path = self.models_dir / f"{experiment_name}_{model_name}.joblib"
        joblib.dump(best_pipeline, model_path)
        
        return result
    
    def save_results(self, results, filename):
        df = pd.DataFrame(results)
        path = self.results_dir / filename
        df.to_csv(path, index=False)
        return df 