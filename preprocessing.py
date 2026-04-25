import os
import re
import glob
import pandas as pd
from langdetect import detect, LangDetectException
import spacy
from tqdm import tqdm

# =========================
# CONFIG
# =========================
CSV_PATH = "data/raw/archelect_search.csv"
TEXT_DIRS = {
    1981: "data/raw/txt_1981/",
    1988: "data/raw/txt_1988/",
    1993: "data/raw/txt_1993/",
}
OUTPUT_PATH = "data/processed/archelec_clean.csv"

ELECTIONS = [1981, 1988, 1993]
MIN_TOKENS = 20

# =========================
# LOAD NLP MODEL
# =========================
nlp = spacy.load("fr_core_news_md", disable=["ner", "parser"])

# =========================
# POLITICAL MAPPING
# =========================
FAMILLE_POLITIQUE = {
    'Parti socialiste': 'Left',
    'Rassemblement pour la République': 'Right',
    'Front national': 'Far right',
    'Les Verts': 'Greens',
    'Lutte ouvrière': 'Far left',
}

def get_famille(soutien):
    if pd.isna(soutien) or soutien == 'non mentionné':
        return 'Other / Unknown'
    for parti in soutien.split(';'):
        parti = parti.strip()
        for key, famille in FAMILLE_POLITIQUE.items():
            if key.lower() in parti.lower():  # matching partiel
                return famille
    return 'Other / Unknown'

# =========================
# CLEAN OCR
# =========================
def clean_ocr(text):
    text = re.sub(r'Sciences Po.*', '', text)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# LANGUAGE FILTER
# =========================
def is_french(text):
    try:
        return detect(text[:500]) == 'fr'
    except LangDetectException:
        return False

# =========================
# TOKENIZATION
# =========================
def tokenize(text):
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop and len(token) > 2
    ]

# =========================
# MAIN PIPELINE
# =========================
def main():
    print("Loading CSV")
    meta = pd.read_csv(CSV_PATH)

    meta['annee'] = pd.to_datetime(meta['date'], errors='coerce').dt.year
    meta = meta[meta['annee'].isin(ELECTIONS)]

    print("Loading texts")
    texts = {}
    for year, directory in TEXT_DIRS.items():
        for fpath in glob.glob(os.path.join(directory, "*.txt")):
            doc_id = os.path.splitext(os.path.basename(fpath))[0]
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                texts[doc_id] = f.read()

    print("Merging")
    df = meta[meta['id'].isin(texts.keys())].copy()
    matched = meta['id'].isin(texts.keys())
    print(f"Documents matchés : {matched.sum()} / {len(meta)}")


    df['text_raw'] = df['id'].map(texts)

    print("Cleaning OCR")
    df['text_clean'] = df['text_raw'].apply(clean_ocr)

    print("Filtering language")
    df = df[df['text_clean'].apply(is_french)]

    print("Mapping political families")
    df['famille'] = df['titulaire-soutien'].apply(get_famille)
    df = df[df['famille'] != 'Other / Unknown']

    print("Tokenizing")
    tqdm.pandas()
    df['tokens'] = df['text_clean'].progress_apply(tokenize)
    df['n_tokens'] = df['tokens'].apply(len)

    df = df[df['n_tokens'] >= MIN_TOKENS]

    print(f"Final dataset size: {len(df)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()