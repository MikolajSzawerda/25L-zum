from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold
from spamclassifier.preprocessing import Word2VecVectorizer

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

EXPERIMENT_CONFIG = {
    'cv': RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42),
    'scoring': ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr'],
    'test_size': 0.2,
    'random_state': 42,
    'n_jobs': -1
}

PREPROCESSING_CONFIGS = {
    'minimal': {'lowercase': True, 'remove_punctuation': False, 'remove_numbers': False, 'remove_stopwords': False, 'stemming': False},
    'standard': {'lowercase': True, 'remove_punctuation': True, 'remove_numbers': True, 'remove_stopwords': True, 'stemming': False},
    'aggressive': {'lowercase': True, 'remove_punctuation': True, 'remove_numbers': True, 'remove_stopwords': True, 'stemming': True},
    'custom': {'lowercase': True, 'remove_punctuation': True, 'remove_numbers': False, 'remove_stopwords': True, 'stemming': True, 'min_word_length': 3}
}

VECTORIZER_CONFIGS = {
    'tfidf_unigram': {'vectorizer': TfidfVectorizer, 'params': {'max_features': 10000, 'ngram_range': (1, 1), 'min_df': 2, 'max_df': 0.95}},
    'tfidf_bigram': {'vectorizer': TfidfVectorizer, 'params': {'max_features': 10000, 'ngram_range': (1, 2), 'min_df': 2, 'max_df': 0.95}},
    'count_unigram': {'vectorizer': CountVectorizer, 'params': {'max_features': 10000, 'ngram_range': (1, 1), 'min_df': 2, 'max_df': 0.95}},
    'count_bigram': {'vectorizer': CountVectorizer, 'params': {'max_features': 10000, 'ngram_range': (1, 2), 'min_df': 2, 'max_df': 0.95}},
    'word2vec_skipgram': {'vectorizer': Word2VecVectorizer, 'params': {'vector_size': 100, 'window': 5, 'min_count': 1, 'sg': 1, 'epochs': 15, 'seed': 42}},
    'word2vec_large': {'vectorizer': Word2VecVectorizer, 'params': {'vector_size': 200, 'window': 10, 'min_count': 2, 'sg': 0, 'epochs': 15, 'seed': 42}}
}

ALGORITHM_CONFIGS = {
    'MultinomialNB': {
        'model': MultinomialNB(),
        'param_grid': {'model__alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
    },
    'SVM_Linear': {
        'model': SVC(kernel='linear', probability=True, random_state=42),
        'param_grid': {'model__C': [0.1, 1.0, 10.0, 100.0]}
    },
    'SVM_RBF': {
        'model': SVC(kernel='rbf', probability=True, random_state=42),
        'param_grid': {'model__C': [0.1, 1.0, 10.0], 'model__gamma': ['scale', 'auto', 0.001, 0.01]}
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'param_grid': {
            'model__n_estimators': [100, 300],
            'model__max_depth': [None, 20],
            'model__max_features': ['sqrt'],
            'model__class_weight': [None, 'balanced']
        }
    }
}

if XGBClassifier is not None:
    ALGORITHM_CONFIGS['XGBoost'] = {
        'model': XGBClassifier(random_state=42, eval_metric='logloss'),
        'param_grid': {
            'model__max_depth': [4, 6],
            'model__min_child_weight': [1, 4],
            'model__learning_rate': [0.05, 0.1],
            'model__subsample': [0.85],
            'model__colsample_bytree': [0.85]
        }
    }

DIMENSIONALITY_CONFIGS = {
    'PCA_50': PCA(n_components=50, random_state=42),
    'PCA_100': PCA(n_components=100, random_state=42),
    'SVD_50': TruncatedSVD(n_components=50, random_state=42),
    'SVD_100': TruncatedSVD(n_components=100, random_state=42),
    'SelectKBest_50': SelectKBest(f_classif, k=50),
    'SelectKBest_100': SelectKBest(f_classif, k=100),
    'SelectKBest_200': SelectKBest(f_classif, k=200)
}

CLASS_BALANCE_CONFIGS = {
    'no_balance': {},
    'balanced': {'class_weight': 'balanced'},
    'balanced_subsample': {'class_weight': 'balanced_subsample'}
} 