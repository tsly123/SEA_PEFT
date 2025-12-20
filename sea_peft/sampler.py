"""Audit sampling utilities with coverage and debt tracking."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence


class AuditSampler:
    def __init__(
        self,
        units: Iterable[str],
        coverage_target: int = 3,
        debt_horizon: int = 6,
        seed: int = 0,
        active_units: Iterable[str] | None = None,
    ) -> None:
        self.units = list(units)
        self.coverage_target = coverage_target
        self.debt_horizon = debt_horizon
        self.probe_counts: Dict[str, int] = {u: 0 for u in self.units}
        self.debt_counters: Dict[str, int] = {u: 0 for u in self.units}
        self.rng = random.Random(seed)
        self.active_units: List[str] = list(active_units or [])
        self.last_selection: List[str] = []

    def record(self, sampled_units: Sequence[str]) -> None:
        sampled = set(sampled_units)
        self.last_selection = list(sampled_units)
        for unit in self.units:
            if unit in sampled:
                self.probe_counts[unit] += 1
                self.debt_counters[unit] = 0
            else:
                self.debt_counters[unit] += 1

    def update_active_units(self, active_units: Iterable[str]) -> None:
        self.active_units = list(active_units)

    def sample(
        self,
        k: int,
        uncertainty: Dict[str, float] | None = None,
        uniform_only: bool = False,
    ) -> List[str]:
        forced = [u for u, debt in self.debt_counters.items() if debt >= self.debt_horizon]
        self.rng.shuffle(forced)
        selection: List[str] = forced[: min(len(forced), k)]

        remaining = max(k - len(selection), 0)
        if remaining == 0:
            return selection[:k]

        active_set = set(self.active_units)
        pool_active = [u for u in self.units if u in active_set and u not in selection]
        pool_inactive = [u for u in self.units if u not in active_set and u not in selection]

        target_active = int(round(remaining * 0.9)) # default 0.3
        target_inactive = remaining - target_active

        if target_active > len(pool_active):
            target_inactive += target_active - len(pool_active)
            target_active = len(pool_active)
        if target_inactive > len(pool_inactive):
            target_active += target_inactive - len(pool_inactive)
            target_inactive = len(pool_inactive)

        def _pick(pool: List[str], target: int) -> List[str]:
            if target <= 0 or not pool:
                return []
            if uniform_only or not uncertainty:
                self.rng.shuffle(pool)
                return pool[:target]
            coverage_sorted = sorted(
                pool,
                key=lambda u: (self.probe_counts[u], -self.debt_counters[u]),
            )
            quota = min(len(pool), max(1, target // 2))
            picks = coverage_sorted[:quota]
            remaining_local = target - len(picks)
            if remaining_local > 0:
                leftover = [u for u in pool if u not in picks]
                leftover.sort(key=lambda u: uncertainty.get(u, 0.0), reverse=True)
                picks.extend(leftover[:remaining_local])
            return picks[:target]

        active_picks = _pick(pool_active, target_active)
        selection.extend(active_picks)

        inactive_picks = _pick([u for u in pool_inactive if u not in selection], target_inactive)
        selection.extend(inactive_picks)

        remaining = k - len(selection)
        if remaining <= 0:
            return selection[:k]

        residual_pool = [u for u in self.units if u not in selection]
        if uniform_only or not uncertainty:
            self.rng.shuffle(residual_pool)
            selection.extend(residual_pool[:remaining])
            return selection[:k]

        residual_pool.sort(key=lambda u: uncertainty.get(u, 0.0), reverse=True)
        selection.extend(residual_pool[:remaining])
        return selection[:k]

    def coverage_snapshot(self) -> Dict[str, int]:
        return dict(self.probe_counts)


__all__ = ["AuditSampler"]
