from sklearn.base import BaseEstimator, TransformerMixin
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List, Optional
import numpy as np


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = True,
                 remove_stopwords: bool = True,
                 stemming: bool = True,
                 language: str = 'english',
                 min_word_length: int = 2,
                 custom_stopwords: Optional[List[str]] = None):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.language = language
        self.min_word_length = min_word_length
        self.custom_stopwords = custom_stopwords

    def _initialize_components(self):
        if not hasattr(self, '_components_initialized'):
            try:
                if self.stemming:
                    self.stemmer = PorterStemmer()
                
                if self.remove_stopwords:
                    try:
                        self.stop_words = set(stopwords.words(self.language))
                    except LookupError:
                        nltk.download('stopwords', quiet=True)
                        self.stop_words = set(stopwords.words(self.language))
                    
                    if self.custom_stopwords:
                        self.stop_words.update(self.custom_stopwords)
                
                self._components_initialized = True
            except Exception as e:
                raise

    def fit(self, X, y=None):
        self._initialize_components()
        return self

    def transform(self, X, y=None):
        if not hasattr(self, '_components_initialized'):
            self._initialize_components()
        
        if not hasattr(X, '__iter__') or isinstance(X, str):
            raise ValueError("X must be an iterable of strings")
        return [self._preprocess_text(text) for text in X]

    def _preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        
        if not text or text.isspace():
            return ""
        
        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        tokens = text.split()

        if self.min_word_length > 0:
            tokens = [token for token in tokens if len(token) >= self.min_word_length]

        if self.remove_stopwords and hasattr(self, 'stop_words'):
            tokens = [token for token in tokens if token not in self.stop_words]

        if self.stemming and hasattr(self, 'stemmer'):
            tokens = [self.stemmer.stem(token) for token in tokens]

        return ' '.join(tokens)

    def get_feature_names_out(self, input_features=None):
        return np.array(['preprocessed_text'], dtype=object) 