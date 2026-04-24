# src/conformisme.py

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from scipy.stats import entropy

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/processed/archelec_clean.csv")

# Convert tokens back
import ast
df['tokens'] = df['tokens'].apply(ast.literal_eval)

docs = [" ".join(tokens) for tokens in df['tokens']]

# =========================
# TF-IDF
# =========================
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(docs)

# =========================
# NMF
# =========================
n_topics = 10
nmf = NMF(n_components=n_topics, random_state=42)
W = nmf.fit_transform(X)

# NORMALISATION
W = W / W.sum(axis=1, keepdims=True)

# =========================
# PARTY PROFILES
# =========================
df_topics = pd.DataFrame(W)
df_topics['famille'] = df['famille']

party_profiles = df_topics.groupby('famille').mean()

# =========================
# DISTANCES
# =========================
def kl_divergence(p, q):
    return entropy(p, q)

def cosine_distance(p, q):
    return cosine(p, q)

kl_scores = []
cos_scores = []

for i, row in df_topics.iterrows():
    party = row['famille']
    p = row.drop('famille').values
    q = party_profiles.loc[party].values

    kl_scores.append(kl_divergence(p, q))
    cos_scores.append(cosine_distance(p, q))

df['kl_divergence'] = kl_scores
df['cosine_distance'] = cos_scores

# =========================
# SAVE
# =========================
df.to_csv("data/processed/conformisme_scores.csv", index=False)

print("Done")