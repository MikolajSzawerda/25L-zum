from src.config import *
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import Any


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, lowercase=True, remove_punctuation=True,
                 remove_numbers=True, remove_stopwords=True,
                 stemming=True, language='english'):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.language = language

        # Initialize stemmer and stopwords
        if self.stemming:
            self.stemmer = PorterStemmer()
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words(self.language))

    def fit(self, X, y=None):
        # Nothing to fit for this transformer
        return self

    def transform(self, X, y=None):
        # Apply preprocessing to each document in X
        return [self._preprocess_text(text) for text in X]

    def _preprocess_text(self, text):
        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)

        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        tokens = text.split()

        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]

        if self.stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]

        preprocessed_text = ' '.join(tokens)
        return preprocessed_text


def search_best_parameters(X: Any, y: Any, cls: Any, preprocessor: TextPreprocessor, cv: StratifiedKFold,
                           param_grid: dict | list, scoring: Any) -> GridSearchCV:
    text_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('vectorizer', CountVectorizer()),
        ('classifier', cls)
    ])

    grid_search = GridSearchCV(text_pipeline, param_grid=param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X, y)

    return grid_search