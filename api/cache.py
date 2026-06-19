from cachetools import TTLCache
import hashlib
import json

# Max 256 cached responses, each expires after 1 hour
_cache = TTLCache(maxsize=256, ttl=3600)

def make_cache_key(question: str, top_k: int, source_filter: str | None) -> str:
    normalized = question.strip().lower()
    raw = json.dumps({"q": normalized, "k": top_k, "f": source_filter}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()

def get_cached(key: str):
    return _cache.get(key)

def set_cached(key: str, value):
    _cache[key] = value

def cache_stats() -> dict:
    return {"size": len(_cache), "maxsize": _cache.maxsize}