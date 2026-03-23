# helpers/auto_synonym_engine.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# === WordNet safe import (one block) ===
try:
    from nltk.corpus import wordnet as wn
    import nltk
    try:
        wn.synsets("example")
    except LookupError:
        nltk.download("wordnet", quiet=True)
except Exception:
    wn = None  # fallback αν κάτι πάει στραβά
# =======================================


class AutoSynonymEngine:
    def __init__(self):
        self.vec = None
        self.matrix = None
        self.feature_names = None
        self.texts = []

    def fit(self, texts):
        """Train TF-IDF model on corpus texts."""
        if not texts:
            self.vec = None
            self.matrix = None
            self.feature_names = None
            self.texts = []
            return
        self.texts = texts
        self.vec = TfidfVectorizer(stop_words="english")
        self.matrix = self.vec.fit_transform(texts)
        self.feature_names = np.array(self.vec.get_feature_names_out())

    def similar_terms(self, term, top_n=5):
        """Find words with similar usage in the corpus."""
        if self.vec is None or self.feature_names is None:
            return []
        hits = np.where(self.feature_names == term)[0]
        if hits.size == 0:
            return []
        idx = int(hits[0])
        term_vec = self.matrix[:, idx].T
        sims = cosine_similarity(term_vec, self.matrix.T).flatten()
        top_idx = sims.argsort()[-top_n-1:-1][::-1]
        return self.feature_names[top_idx].tolist()

    def wordnet_synonyms(self, term: str) -> list[str]:
        """Return WordNet synonyms (if available)."""
        if wn is None:
            return []
        syns = set()
        for syn in wn.synsets(term):
            for lemma in syn.lemmas():
                syns.add(lemma.name().replace("_", " "))
        return list(syns)

    def expand_query(self, query, max_terms=8):
        """Combine original + WordNet + corpus-similar terms."""
        terms = set()
        for token in query.lower().split():
            terms.add(token)
            terms.update(self.wordnet_synonyms(token))
            terms.update(self.similar_terms(token))
        return list(terms)[:max_terms]

