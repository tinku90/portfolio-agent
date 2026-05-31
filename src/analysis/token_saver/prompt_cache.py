"""
Prompt optimization cache — zero-cost deduplication.

Keyed by SHA-256(aggressiveness + prompt). LRU eviction at configurable size.
Thread-safe for async FastAPI handlers (GIL covers dict operations).
"""

import hashlib
from collections import OrderedDict

_MAX_SIZE = 500   # entries kept in memory


class _LRUCache:
    def __init__(self, maxsize: int = _MAX_SIZE) -> None:
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._maxsize = maxsize
        self._hits   = 0
        self._misses = 0

    # ── key ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _key(prompt: str, aggressiveness: str) -> str:
        raw = f"{aggressiveness}:{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    # ── public API ───────────────────────────────────────────────────────────
    def get(self, prompt: str, aggressiveness: str) -> str | None:
        key = self._key(prompt, aggressiveness)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, prompt: str, aggressiveness: str, result: str) -> None:
        key = self._key(prompt, aggressiveness)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = result
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)   # evict oldest (LRU)
            self._cache[key] = result

    def invalidate(self, prompt: str, aggressiveness: str) -> None:
        key = self._key(prompt, aggressiveness)
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()
        self._hits = self._misses = 0

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size":     len(self._cache),
            "capacity": self._maxsize,
            "hits":     self._hits,
            "misses":   self._misses,
            "hit_rate": round(self._hits / total * 100, 1) if total else 0.0,
        }


# Module-level singleton
_cache = _LRUCache()


def cache_get(prompt: str, aggressiveness: str) -> str | None:
    """Return cached optimized prompt, or None on miss."""
    return _cache.get(prompt, aggressiveness)


def cache_set(prompt: str, aggressiveness: str, result: str) -> None:
    """Store an optimized prompt in the cache."""
    _cache.set(prompt, aggressiveness, result)


def cache_stats() -> dict:
    """Return hit/miss stats for the /stats endpoint."""
    return _cache.stats()


def cache_clear() -> None:
    """Flush the entire cache (used in tests or admin reset)."""
    _cache.clear()
