from rapidfuzz import fuzz
from helpers.text_utils import normalize_text

def smart_fuzzy_score(query: str, text: str) -> float:
    q, t = normalize_text(query), normalize_text(text)
    s1 = fuzz.partial_ratio(q, t)
    s2 = fuzz.token_sort_ratio(q, t)
    s3 = fuzz.token_set_ratio(q, t)
    return 0.4*s1 + 0.4*s2 + 0.2*s3

def text_similarity(a: str, b: str) -> float:
    return fuzz.token_set_ratio(normalize_text(a), normalize_text(b))

def mmr_rerank(results, lam: float = 0.65, top_k: int = 8):
    if not results:
        return results
    picked = [results[0]]
    cand = results[1:]
    while cand and len(picked) < top_k:
        best_i, best_score = 0, -1e9
        for i, r in enumerate(cand):
            rel = r["score"]
            div = max(text_similarity(r["snippet"], p["snippet"]) for p in picked)
            mmr = lam*rel - (1-lam)*div
            if mmr > best_score:
                best_i, best_score = i, mmr
        picked.append(cand.pop(best_i))
    return picked
