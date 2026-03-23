# helpers/semantic_rerank.py
import os, numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

# Load once, global (CPU-friendly)
MODEL_NAME = os.getenv("MCP_SEM_MODEL", "all-MiniLM-L6-v2")
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model

def embed(texts: List[str]):
    model = get_model()
    return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

def rerank_semantic(query: str, candidates: List[Dict],
                    top_k: int = 8, alpha: float = 0.5) -> List[Dict]:
    """
    query: string
    candidates: list[{ 'uri', 'snippet', 'score' (lexical 0..1) }]
    fusion: final = alpha * semantic + (1-alpha) * lexical
    """
    if not candidates:
        return []

    # Embed query + all snippets
    query_emb = embed([query])[0]
    snippets = [c["snippet"] for c in candidates]
    cand_emb = embed(snippets)

    # Semantic cosine scores 0..1
    semantic_scores = util.cos_sim(query_emb, cand_emb).cpu().numpy().flatten()

    fused = []
    for c, sem in zip(candidates, semantic_scores):
        fused_score = (alpha * float(sem)) + ((1 - alpha) * float(c["score"]))
        fused.append({
            **c,
            "lexical_score": c["score"],
            "semantic_score": round(float(sem), 3),
            "score": round(fused_score, 3),
        })

    # Sort desc & trim
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:top_k]
