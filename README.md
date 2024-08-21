# Multilingual Language Identification - IIT Delhi COL772

A lean, surprisingly strong **language ID** pipeline using character/word n-gram TF-IDF features and Multinomial Naive Bayes. Built for **COL772 (Natural Language Processing)** at IIT Delhi.

## Why this works

Language identity lives in orthographic and short n-gram statistics. Instead of a heavy neural encoder, this system:

1. Light text cleanup (strip digits / punctuation noise)
2. Builds a **TF-IDF** matrix over word unigrams + bigrams (`ngram_range=(1,2)`)
3. Trains **Multinomial Naive Bayes** with a tiny Laplace smoother (`alpha=5e-5`)
4. Serializes `model.joblib` + `vectorizer.joblib` for fast offline inference

## Usage

```bash
# Train
python final.py train /path/to/data_dir /path/to/save_dir

# Infer
python final.py test /path/to/save_dir /path/to/test.json predictions.txt
```

Expected data JSON items: `{ "text": "...", "langid": "..." }` for train; `{ "text": "..." }` for test.

## Dependencies

```
scikit-learn
joblib
```

## Design choices

| Choice | Rationale |
|---|---|
| TF-IDF + NB | Extremely fast train/infer, strong on short multilingual text |
| `(1,2)` word n-grams | Captures function-word patterns without huge vocab explosion |
| Custom tokenizer `r'\S+'` | Keeps whitespace-separated tokens across scripts |
| Aggressive special-char strip | Stabilizes features across noisy social/web text |

## Course

**COL772 - Natural Language Processing**, IIT Delhi  
Assignment 1: Language identification

## License

Coursework / educational use.
