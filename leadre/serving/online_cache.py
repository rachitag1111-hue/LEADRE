"""
Latency-Sensitive Service Module & User Candidate Cache.
Implements Section 4.4.1:
- Sub-10ms ad candidate retrieval from pre-computed cache
- User Server querying & async interaction logging
"""

import time
from typing import List, Dict, Optional, Any


class UserCandidateCache:
    """
    In-memory / Redis cache storing pre-computed generative retrieval ad lists per user.
    Enables the online Mixer to respond within <10ms SLA.
    """
    def __init__(self, default_ttl_seconds: int = 86400):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl_seconds
        self.hits = 0
        self.misses = 0

    def get_candidates(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Fast lookup (<1ms in-memory, <5ms in Redis).
        """
        entry = self._cache.get(user_id)
        if entry is None:
            self.misses += 1
            return None

        if time.time() > entry["expires_at"]:
            del self._cache[user_id]
            self.misses += 1
            return None

        self.hits += 1
        return entry["candidates"]

    def set_candidates(self, user_id: str, candidates: List[Dict[str, Any]], ttl: Optional[int] = None):
        """
        Sets pre-computed candidate ads from the nearline generative retrieval worker.
        """
        expires_at = time.time() + (ttl or self.default_ttl)
        self._cache[user_id] = {
            "candidates": candidates,
            "updated_at": time.time(),
            "expires_at": expires_at
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "cached_users": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


class OnlineMixerService:
    """
    Simulates the online Mixer and User Server (Section 4.4.1 Inference).
    """
    def __init__(self, cache: UserCandidateCache):
        self.cache = cache

    def handle_request(self, user_id: str) -> Dict[str, Any]:
        """
        Handles real-time online request:
        1. Checks cache for pre-computed LEADRE candidates.
        2. Measures latency (ensures SLA < 10ms).
        """
        t0 = time.perf_counter()
        candidates = self.cache.get_candidates(user_id)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "user_id": user_id,
            "candidates": candidates or [],
            "found_in_cache": candidates is not None,
            "latency_ms": round(latency_ms, 3)
        }
