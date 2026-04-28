# NLP-project-archelec

# Measuring Thematic Conformism in French Campaign Manifestos

This project studies **thematic conformism** in French legislative campaign manifestos (*professions de foi*) between 1981 and 1993.

The objective is to measure how much individual candidates deviate from the average discourse of their political family using topic modeling and distance metrics.

---

## 📊 Methodology

1. **Preprocessing**
   - Merge metadata (CSV) 
   - Language filtering (French only)
   - Political family mapping
   - Tokenization and lemmatization (spaCy)

2. **Topic Modeling**
   - TF-IDF vectorization
   - Non-negative Matrix Factorization (NMF)
   - Selection of optimal number of topics using UMass coherence

3. **Conformism Scores**
   - Compute party centroids (leave-one-out)
   - Measure distance between candidates and their party:
     - Cosine distance
     - KL-divergence

---

## 📦 Dependencies
Python 3.10+
pandas
numpy
scikit-learn
scipy
spaCy (fr_core_news_md)
nltk
matplotlib / seaborn

Install spaCy model:
python -m spacy download fr_core_news_md



## Notes
GPU is not required (CPU is sufficient)
KL-divergence is sensitive to small values → smoothing applied
Results may vary slightly due to random initialization (fixed seed used)


##  Output
Clean dataset: data/processed/archelec_clean.csv
Conformism scores dataset
Figures for analysis
Final report


