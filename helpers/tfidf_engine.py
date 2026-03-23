# helpers/tfidf_engine.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfEngine:
    def __init__(self):
        self.vec = None
        self.uris = []
        self.matrix = None
        self.texts = []

    def fit(self, uri_text_pairs):
        self.uris = [u for u, _ in uri_text_pairs]
        self.texts = [t for _, t in uri_text_pairs]
        self.vec = TfidfVectorizer(stop_words="english")
        self.matrix = self.vec.fit_transform(self.texts)

    def search(self, query, top_k=8):
        if not self.vec or not self.uris:
            return []
        qv = self.vec.transform([query])
        sims = cosine_similarity(qv, self.matrix).ravel()
        idx = sims.argsort()[::-1][:top_k]
        out = []
        for i in idx:
            out.append({
                "uri": self.uris[i],
                "snippet": (self.texts[i] or "")[:400],
                "score": float(sims[i])
            })
        return out
