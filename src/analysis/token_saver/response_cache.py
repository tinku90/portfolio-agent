# -*- coding: utf-8 -*-
"""
LLM response cache — serves identical requests from disk, zero API cost.

Keyed by SHA-256 of (messages + response_format + temperature + max_tokens).
Because the uploaded document text and fetched news are part of the message
content, the key changes whenever the actual inputs change — so re-running
analysis on the SAME stock with the SAME documents returns instantly, but a
new document or fresh news produces a cache miss and a real call.

Disk-backed (survives Streamlit restarts) + in-memory LRU.
"""
import os, json, hashlib, threading
from collections import OrderedDict
from pathlib import Path

_CACHE_FILE = Path(__file__).resolve().parents[3] / "data" / "llm_response_cache.json"
_MAX_ENTRIES = 500
_lock   = threading.Lock()
_cache  = OrderedDict()
_loaded = False
_hits   = 0
_misses = 0


def _load():
    global _loaded
    if _loaded:
        return
    try:
        if _CACHE_FILE.exists():
            data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            _cache.update(data)
    except Exception:
        pass
    _loaded = True


def _persist():
    try:
        _CACHE_FILE.parent.mkdir(exist_ok=True)
        tmp = str(_CACHE_FILE) + ".tmp"
        Path(tmp).write_text(json.dumps(_cache, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, _CACHE_FILE)
    except Exception:
        pass


def make_key(messages, response_format, temperature, max_tokens) -> str:
    payload = json.dumps(
        {"m": messages, "rf": bool(response_format), "t": temperature, "mx": max_tokens},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def get(key: str):
    global _hits, _misses
    _load()
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            _hits += 1
            return _cache[key]
    _misses += 1
    return None


def set(key: str, value: str):
    _load()
    with _lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)
    _persist()


def clear():
    global _hits, _misses
    with _lock:
        _cache.clear()
        _hits = _misses = 0
    _persist()


def stats() -> dict:
    total = _hits + _misses
    return {
        "entries":  len(_cache),
        "hits":     _hits,
        "misses":   _misses,
        "hit_rate": round(_hits / total * 100, 1) if total else 0.0,
    }
