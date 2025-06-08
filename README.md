# Klasyfikator Spamu - Badania Eksperymentalne

Repozytorium zawiera kompleksowe badania nad budową skutecznego klasyfikatora wykrywającego wiadomości spam. Projekt obejmuje trzy główne obszary eksperymentalne: wpływ wstępnej obróbki tekstu i metod wektoryzacji, porównanie algorytmów uczenia maszynowego oraz zastosowanie technik redukcji wymiarowości.

## 📁 Struktura Repozytorium

```
├── README.md                    
├── pyproject.toml              # Konfiguracja projektu i zależności
├── justfile                    # Automatyzacja zadań (lint, test, init)
├── logging.conf                # Konfiguracja logowania
├── uv.lock                     
│
├── spamclassifier/             # Główny moduł projektu
│   ├── __init__.py
│   ├── preprocessing.py        # Klasy do wstępnej obróbki tekstu
│   ├── experiment_configs.py   # Konfiguracje eksperymentów
│   └── experiment_runner.py    # Silnik uruchamiania eksperymentów
│
├── scripts/                    # Skrypty eksperymentalne
│   ├── config.py              # Konfiguracja ścieżek i parametrów
│   ├── run_preprocessing_study.py      # Eksperyment 1: Obróbka tekstu
│   ├── run_algorithm_comparison.py     # Eksperyment 2: Porównanie algorytmów
│   ├── run_dimensionality_study.py     # Eksperyment 3: Redukcja wymiarowości
│   ├── evaluate_models.py             # Ewaluacja modeli
│   ├── measure_time.py                # Pomiary wydajności
│   └── compare_by_category.py         # Analiza według kategorii spam/ham
│
├── notebooks/                  # Jupyter notebooks z analizami
│   ├── 01-spam-dataset.ipynb         # Eksploracja danych
│   ├── 02-pipeline.ipynb             # Budowa pipeline'u
│   ├── 03-test-search-best.ipynb     # Wyszukiwanie najlepszych parametrów
│   ├── 04-text-preprocessing-study.ipynb  # Analiza obróbki tekstu
│   ├── 05-algorithms-comparison.ipynb     # Porównanie algorytmów
│   ├── 06-dimensionality-study.ipynb      # Badanie redukcji wymiarowości
│   └── utils.py                       # Funkcje pomocnicze
│
├── data/                       # Katalogi danych
│   ├── raw/                   # Surowe dane wejściowe
│   ├── input/                 # Przetworzone dane wejściowe  
│   └── output/                # Wyniki eksperymentów
│
├── docs/                       # Dokumentacja
│   └── main.tex               # Sprawozdanie końcowe (LaTeX)
│
├── models/                     # Zapisane modele
├── logs/                       # Logi aplikacji
└── tests/                      
```

## 🔬 Opis Eksperymentów

### 1. Badanie Wstępnej Obróbki Tekstu (`run_preprocessing_study.py`)

**Cel**: Analiza wpływu różnych operacji przetwarzania tekstu na jakość klasyfikacji.

**Badane operacje**:
- Konwersja na małe litery
- Usuwanie interpunkcji i liczb
- Eliminacja "stop words"
- Stemming (usuwanie końcówek fleksyjnych)

**Metody wektoryzacji**:
- Bag of Words (BoW) - unigram/bigram
- TF-IDF - unigram/bigram  
- Word2Vec - różne rozmiary wektorów

**Model bazowy**: Random Forest (100 drzew)

**Kluczowe wnioski**:
- Największy wpływ ma wybór metody wektoryzacji, nie preprocessing
- TF-IDF unigram z podstawową obróbką daje najlepsze wyniki
- Agresywne czyszczenie (stemming) pogarsza rezultaty

### 2. Porównanie Algorytmów (`run_algorithm_comparison.py`)

**Cel**: Ocena skuteczności różnych algorytmów uczenia maszynowego.

**Testowane algorytmy**:
- **Multinomial Naive Bayes** - szybki, prosty baseline
- **SVM Linear** - liniowe granice decyzyjne
- **SVM RBF** - nieliniowe granice decyzyjne  
- **Random Forest** - zespół drzew decyzyjnych
- **XGBoost** - gradient boosting

**Metodologia**:
- Grid search dla optymalizacji hiperparametrów
- Walidacja krzyżowa (5-fold, 3 powtórzenia)
- Pomiary czasu treningu i predykcji
- Analiza trudnych przypadków ("hard ham")

**Kluczowe wnioski**:
- Wszystkie zaawansowane modele osiągają F1 > 0.98
- XGBoost i SVM Linear mają najlepszy stosunek jakość/czas
- SVM RBF osiąga najwyższy F1 ≈ 0.991, ale jest 4x wolniejszy

### 3. Redukcja Wymiarowości (`run_dimensionality_study.py`)

**Cel**: Badanie wpływu redukcji wymiarowości na wydajność i jakość modelu.

**Testowane metody**:
- **PCA** (Principal Component Analysis) - 50, 100, 200 składowych
- **SVD** (Singular Value Decomposition) - 50, 100, 200 składowych
- **SelectKBest** (ANOVA F-test) - 50, 100, 200, 500 cech

**Model bazowy**: SVM Linear z TF-IDF bigram

**Kluczowe wnioski**:
- PCA z 200 składowymi utrzymuje jakość przy 2x krótszym treningu
- SVD ma najniższe wymagania pamięciowe
- SelectKBest okazał się nieefektywny (spadek F1 o 17 p.p.)

## 🚀 Instalacja i Uruchomienie

### Wymagania
- Python ≥ 3.9
- [uv](https://docs.astral.sh/uv/) - menedżer pakietów
- [just](https://github.com/casey/just) - runner zadań

### Instalacja
```bash
# Klonowanie repozytorium
git clone https://github.com/MikolajSzawerda/25L-zum.git
cd 25L-zum

# Instalacja zależności
just init

# Alternatywnie bez just:
uv sync --all-extras
```

### Uruchamianie Eksperymentów

```bash
# Eksperyment 1: Preprocessing i wektoryzacja
uv run scripts/run_preprocessing_study.py

# Eksperyment 2: Porównanie algorytmów  
uv run scripts/run_algorithm_comparison.py

# Eksperyment 3: Redukcja wymiarowości
uv run scripts/run_dimensionality_study.py

# Pomiary wydajności
uv run scripts/measure_time.py

# Ewaluacja modeli
uv run scripts/evaluate_models.py
```

### Analiza w Jupyter
```bash
# Uruchomienie Jupyter
uv run jupyter lab

# W notebooku:
%load_ext autoreload
%autoreload 2
```

## 📊 Kluczowe Wyniki

### Najlepsze Konfiguracje

1. **Dla maksymalnej jakości**: SVM RBF + TF-IDF unigram (F1 ≈ 0.991)
2. **Dla produkcji**: XGBoost + TF-IDF unigram (F1 ≈ 0.985, szybki trening)
3. **Dla urządzeń mobilnych**: Naive Bayes + BoW (F1 ≈ 0.97, bardzo szybki)
4. **Z redukcją wymiarowości**: SVM Linear + TF-IDF bigram + PCA(200) (F1 ≈ 0.98, 2x szybszy)

### Preprocessing Pipeline
```python
# Optymalna konfiguracja
preprocessing = {
    'lowercase': True,
    'remove_punctuation': True, 
    'remove_numbers': True,
    'remove_stopwords': True,
    'stemming': False  # Pogarsza wyniki!
}
```

## 🛠️ Narzędzia Pomocnicze

```bash
# Formatowanie kodu
just lint

# Czyszczenie cache
just clean

# Testy (placeholder)
just test
```

## 📈 Zależności

Główne biblioteki:
- **scikit-learn** - algorytmy ML i preprocessing
- **xgboost** - gradient boosting
- **nltk** - przetwarzanie języka naturalnego
- **gensim** - Word2Vec embeddings
- **pandas** - manipulacja danych
- **seaborn** - wizualizacje
- **jupyter** - interaktywna analiza

## 📄 Dokumentacja

Pełne sprawozdanie z eksperymentów znajduje się w `docs/main.tex` i zawiera:
- Szczegółową metodologię badań
- Wykresy i tabele wyników
- Analizę statystyczną
- Wnioski i rekomendacje

## 👥 Autorzy

- Anna Schäfer
- Mikołaj Szawerda

---

**Projekt realizowany w ramach kursu Zaawansowane Uczenie Maszynowe (ZUM)**

