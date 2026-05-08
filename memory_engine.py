"""
Memory engine: working, episodic, and lightweight semantic recall.

Encoding attaches salience from reward magnitude and emotional arousal. Recall
uses weighted overlap of cues with stored tags plus recency decay.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple


def _utc_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class MemoryTrace:
    """Single memory trace with searchable metadata."""

    trace_id: str
    ts_ms: int
    content: str
    tags: Tuple[str, ...]
    salience: float
    emotion_tag: str
    embedding_hint: Tuple[float, ...] = field(default_factory=tuple)


class MemoryEngine:
    """
    Tiered memory:
      - working: bounded deque
      - episodic: list with cap
      - semantic: keyword -> aggregate weight (very small scale prototype)
    """

    def __init__(self, *, working_capacity: int, episodic_max: int) -> None:
        self._working: Deque[MemoryTrace] = deque(maxlen=max(8, working_capacity))
        self._episodic: List[MemoryTrace] = []
        self._episodic_max = max(100, episodic_max)
        self._semantic_weights: Dict[str, float] = {}
        self._seq = 0

    @property
    def working(self) -> List[MemoryTrace]:
        return list(self._working)

    @property
    def episodic(self) -> List[MemoryTrace]:
        return list(self._episodic)

    def store_trace(
        self,
        *,
        content: str,
        tags: Sequence[str],
        salience: float,
        emotion_tag: str,
    ) -> MemoryTrace:
        """Write to working memory and conditionally promote to episodic."""
        self._seq += 1
        trace_id = f"mem_{self._seq}"
        trace = MemoryTrace(
            trace_id=trace_id,
            ts_ms=_utc_ms(),
            content=content,
            tags=tuple(sorted({t.lower() for t in tags})),
            salience=max(0.0, float(salience)),
            emotion_tag=emotion_tag.lower(),
        )
        self._working.append(trace)

        if trace.salience >= 0.35 or "important" in trace.tags:
            self._episodic.append(trace)
            if len(self._episodic) > self._episodic_max:
                self._episodic.sort(key=lambda t: t.salience, reverse=True)
                self._episodic = self._episodic[: self._episodic_max]

        # Semantic aggregation: strengthen tag co-occurrence weights.
        for t in trace.tags:
            self._semantic_weights[t] = self._semantic_weights.get(t, 0.0) + trace.salience
        return trace

    def recall_by_cue(
        self,
        cues: Sequence[str],
        *,
        top_k: int = 5,
        emotion_bias: Optional[str] = None,
    ) -> List[MemoryTrace]:
        """Retrieve episodic traces ranked by cue overlap and recency."""
        cue_set = {c.lower() for c in cues if c}
        if not self._episodic:
            return []

        now = _utc_ms()
        scored: List[Tuple[float, MemoryTrace]] = []
        for tr in self._episodic:
            overlap = len(cue_set.intersection(set(tr.tags)))
            base = overlap * 1.0 + 0.35 * tr.salience
            recency = 1.0 / (1.0 + (now - tr.ts_ms) / 60_000.0)
            score = base + 0.55 * recency
            if emotion_bias and tr.emotion_tag == emotion_bias.lower():
                score += 0.25
            scored.append((score, tr))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[: max(1, top_k)]]

    def novelty_hint_for_object(self, object_id: str, label: str) -> float:
        """
        Return a novelty score in [0,1] based on absence from recent memory.

        Used by attention as a lightweight familiarity proxy.
        """
        key_tokens = {object_id.lower(), label.lower()}
        recent = list(self._working)[-16:]
        hits = 0
        for tr in recent:
            if key_tokens.intersection(set(tr.tags)):
                hits += 1
        if hits == 0:
            return 0.75
        if hits < 3:
            return 0.45
        return 0.15

    def consolidate_batch(self, *, salience_threshold: float, max_prune: int) -> Dict[str, int]:
        """
        Prune weak episodic traces and compress semantic weights.

        Returns counts for observability.
        """
        before = len(self._episodic)
        self._episodic = [t for t in self._episodic if t.salience >= salience_threshold]
        pruned = before - len(self._episodic)

        # Enforce episodic capacity, optionally pruning extra aggressively during consolidation.
        target_keep = max(1, self._episodic_max - max(0, int(max_prune)))
        target_keep = min(self._episodic_max, target_keep)
        if len(self._episodic) > target_keep:
            self._episodic.sort(key=lambda t: (t.salience, t.ts_ms), reverse=True)
            pruned += len(self._episodic) - target_keep
            self._episodic = self._episodic[:target_keep]

        # Semantic decay
        decay = 0.85
        for k in list(self._semantic_weights.keys()):
            self._semantic_weights[k] *= decay
            if self._semantic_weights[k] < 1e-3:
                del self._semantic_weights[k]

        return {"pruned": pruned, "remaining": len(self._episodic)}

    def context_hash(self, parts: Iterable[str]) -> str:
        """Stable hash for persisting decision context references."""
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8"))
            h.update(b"|")
        return h.hexdigest()[:32]

    def trace_to_dict(self, tr: MemoryTrace) -> Dict[str, Any]:
        return {
            "traceId": tr.trace_id,
            "tsMs": tr.ts_ms,
            "content": tr.content,
            "tags": list(tr.tags),
            "salience": tr.salience,
            "emotionTag": tr.emotion_tag,
        }

    def export_snapshot(self) -> str:
        """Serialize a small JSON snapshot for debugging (not full graph DB)."""
        payload = {
            "working": [self.trace_to_dict(t) for t in self._working],
            "episodic": [self.trace_to_dict(t) for t in self._episodic[-200:]],
            "semanticTop": sorted(self._semantic_weights.items(), key=lambda kv: kv[1], reverse=True)[:50],
        }
        return json.dumps(payload, separators=(",", ":"))
