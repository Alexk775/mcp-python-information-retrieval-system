import re, unicodedata

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def token_estimate(text: str) -> int:
    """Ρεαλιστικό estimation tokens για LLM budget."""
    if not text:
        return 1
    words = text.split()
    chars = len(text)
    avg_chars = chars / max(len(words), 1)
    return int(chars / (avg_chars * 0.37))
