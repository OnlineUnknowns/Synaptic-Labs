"""
Sleep consolidation engine: offline-style batch jobs for memory pruning and
synaptic normalization. Invoked when the orchestrator detects low activity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from memory_engine import MemoryEngine
from synapse_engine import SynapseEngine


@dataclass
class ConsolidationReport:
    """Summary of consolidation effects."""

    memory_pruned: int
    memory_remaining: int
    synapse_l2: float
    synapse_max_abs: float


class SleepConsolidationEngine:
    """Runs consolidation passes; deterministic given engine state."""

    def __init__(self, *, salience_floor: float = 0.12, max_extra_prune: int = 64) -> None:
        self._salience_floor = float(salience_floor)
        self._max_extra_prune = int(max_extra_prune)

    def run(
        self,
        *,
        memory: MemoryEngine,
        synapse: SynapseEngine,
    ) -> ConsolidationReport:
        mem_stats = memory.consolidate_batch(
            salience_threshold=self._salience_floor,
            max_prune=self._max_extra_prune,
        )
        synapse.normalize_full()
        synapse.decay_eligibility()
        l2, mx = synapse.weight_norms()
        return ConsolidationReport(
            memory_pruned=int(mem_stats.get("pruned", 0)),
            memory_remaining=int(mem_stats.get("remaining", 0)),
            synapse_l2=float(l2),
            synapse_max_abs=float(mx),
        )

    def report_to_dict(self, r: ConsolidationReport) -> Dict[str, Any]:
        return {
            "memoryPruned": r.memory_pruned,
            "memoryRemaining": r.memory_remaining,
            "synapseL2": r.synapse_l2,
            "synapseMaxAbs": r.synapse_max_abs,
        }
