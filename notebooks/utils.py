import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from scripts.config import *
from spamclassifier.experiment_configs import *
from spamclassifier.preprocessing import *
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import Any
