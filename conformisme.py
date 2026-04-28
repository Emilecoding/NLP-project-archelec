
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import json
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

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
tfidf = TfidfVectorizer(max_features=2000)
X = tfidf.fit_transform(docs)

# =========================
# NMF
# =========================
n_topics = 10
nmf = NMF(n_components=n_topics, random_state=42)
W = nmf.fit_transform(X)

tokenized_docs = [doc.split() for doc in docs]
gensim_dict    = Dictionary(tokenized_docs)
gensim_corpus  = [gensim_dict.doc2bow(doc) for doc in tokenized_docs]

umass_scores = {}
for k in [5, 7, 9, 11, 13, 15]:
    nmf = NMF(n_components=k, random_state=42)
    nmf.fit(X)

# Récupère les top-20 mots par topic
    feature_names = tfidf.get_feature_names_out()
    topics = [
        [feature_names[i] for i in topic.argsort()[:-21:-1]]
        for topic in nmf.components_
    ]
    
    cm = CoherenceModel(
        topics=topics,
        corpus=gensim_corpus,
        dictionary=gensim_dict,
        coherence="u_mass"
    )
    umass_scores[k] = cm.get_coherence()
    print(f"K={k} → UMass={umass_scores[k]:.4f}")

with open("data/processed/umass_scores.json", "w") as f:
    json.dump(umass_scores, f)

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
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    p = p / p.sum() #normalize
    q = q / q.sum()
    
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