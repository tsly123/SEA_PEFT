"""Greedy Dice-first allocator with guard cycles and hysteresis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .units import PEFTUnit


@dataclass
class AllocationStats:
    density: float
    value: float
    weight: float
    uncertainty: float = 0.0


@dataclass
class AllocationDiagnostics:
    """Phase 2: Diagnostics returned by allocator for export metadata."""
    selected_units: List[str]
    density_map: Dict[str, float]
    uncertainty_map: Dict[str, float]
    param_fraction: float
    total_params: float


@dataclass
class CandidateRow:
    unit: PEFTUnit
    density: float
    per_param_value: float
    total_value: float
    weight: float
    uncertainty: float


class DiceFirstAllocator:
    def __init__(
        self,
        units: Iterable[PEFTUnit],
        total_params: int,
        budget_fraction: float,
        guard_cycles: int = 3,
        hysteresis_down: float = 0.02,
        beta_uncertainty: float = 0.5,
        *,
        use_promotion_fsm: bool = False,
        fsm_promotion_votes: int = 3,
        fsm_demotion_votes: int = 3,
    ) -> None:
        self.units: List[PEFTUnit] = list(units)
        self._unit_map: Dict[str, PEFTUnit] = {unit.name: unit for unit in self.units}
        self.total_params = float(max(total_params, 1))
        self.budget = budget_fraction * self.total_params
        self.guard_cycles = guard_cycles
        self.hysteresis_down = hysteresis_down
        self.beta_uncertainty = beta_uncertainty
        self._stats: Dict[str, AllocationStats] = {}
        self.use_promotion_fsm = use_promotion_fsm
        self.fsm_promotion_votes = max(1, fsm_promotion_votes)
        self.fsm_demotion_votes = max(1, fsm_demotion_votes)

        if self.use_promotion_fsm:
            for unit in self.units:
                unit.state.promotion_votes = 0
                unit.state.demotion_votes = 0

    def _decay_guards(self) -> None:
        for unit in self.units:
            if unit.state.guard > 0:
                unit.state.guard -= 1

    def _selected_params(self, selected: Set[PEFTUnit]) -> float:
        return sum(unit.params for unit in selected)

    def _locked_active(self) -> List[PEFTUnit]:
        locked: List[PEFTUnit] = []
        for unit in self.units:
            if unit.state.active and unit.state.guard > 0:
                locked.append(unit)
        return locked

    def get_diagnostics(
        self,
        selected: Set[PEFTUnit],
        predictions: Dict[str, float],
        uncertainties: Dict[str, float],
    ) -> AllocationDiagnostics:
        """Phase 2: Return diagnostics for export metadata."""
        density_map = {}
        uncertainty_map = {}
        for unit in selected:
            density_map[unit.name] = predictions.get(unit.name, 0.0)
            uncertainty_map[unit.name] = uncertainties.get(unit.name, 0.0)

        selected_params = self._selected_params(selected)
        param_fraction = selected_params / self.total_params if self.total_params else 0.0

        return AllocationDiagnostics(
            selected_units=[u.name for u in selected],
            density_map=density_map,
            uncertainty_map=uncertainty_map,
            param_fraction=param_fraction,
            total_params=selected_params,
        )

    def allocate(
        self,
        step: int,
        predictions: Dict[str, float],
        uncertainties: Dict[str, float],
    ) -> Set[PEFTUnit]:
        self._decay_guards()
        locked = self._locked_active()
        selected, stats = self._select_projected_units(
            predictions,
            uncertainties,
            locked=set(locked),
            prefer_active=True,
            respect_activation_thresholds=True,
        )
        self._stats = stats
        return selected

    def apply(
        self,
        desired: Set[PEFTUnit],
    ) -> Tuple[List[PEFTUnit], List[PEFTUnit], List[Tuple[PEFTUnit, Tuple[str, str, int, int]]]]:
        activated: List[PEFTUnit] = []
        deactivated: List[PEFTUnit] = []
        transitions: List[Tuple[PEFTUnit, Tuple[str, str, int, int]]] = []

        for unit in self.units:
            target_active = unit in desired
            currently_active = unit.state.active

            if unit.state.guard > 0 and target_active != currently_active:
                continue

            transition: Optional[Tuple[str, str, int, int]] = None

            if self.use_promotion_fsm and target_active != currently_active:
                if target_active:
                    unit.state.promotion_votes += 1
                    unit.state.demotion_votes = 0
                    if unit.state.promotion_votes < self.fsm_promotion_votes:
                        continue
                    transition = (
                        "inactive",
                        "active",
                        self.fsm_promotion_votes,
                        unit.state.promotion_votes,
                    )
                    unit.state.promotion_votes = 0
                else:
                    unit.state.demotion_votes += 1
                    unit.state.promotion_votes = 0
                    if unit.state.demotion_votes < self.fsm_demotion_votes:
                        continue
                    transition = (
                        "active",
                        "inactive",
                        self.fsm_demotion_votes,
                        unit.state.demotion_votes,
                    )
                    unit.state.demotion_votes = 0

            if target_active == currently_active:
                if self.use_promotion_fsm:
                    unit.state.promotion_votes = 0
                    unit.state.demotion_votes = 0
                continue

            if target_active:
                unit.activate()
                unit.state.guard = self.guard_cycles
                activated.append(unit)
            else:
                unit.deactivate()
                unit.state.guard = self.guard_cycles
                deactivated.append(unit)

            if transition is not None:
                unit.state.last_transition = transition
                transitions.append((unit, transition))

        return activated, deactivated, transitions

    def active_params(self) -> float:
        return sum(unit.param_count() for unit in self.units)

    def project_candidate_names(
        self,
        candidate_names: Sequence[str],
        predictions: Dict[str, float],
        uncertainties: Dict[str, float],
        *,
        prefer_active: bool = True,
        respect_activation_thresholds: bool = True,
        budget_override: Optional[float] = None,
    ) -> Tuple[Set[str], Dict[str, AllocationStats]]:
        subset = [
            self._unit_map[name]
            for name in candidate_names
            if name in self._unit_map
        ]
        selected, stats = self._select_projected_units(
            predictions,
            uncertainties,
            locked=set(),
            prefer_active=prefer_active,
            respect_activation_thresholds=respect_activation_thresholds,
            budget_override=budget_override if budget_override is not None else self.budget,
            subset=subset,
        )
        return {unit.name for unit in selected}, stats

    def _build_candidate_rows(
        self,
        predictions: Dict[str, float],
        uncertainties: Dict[str, float],
        subset: Optional[Iterable[PEFTUnit]] = None,
    ) -> Dict[Tuple[str, str], List[CandidateRow]]:
        rows: Dict[Tuple[str, str], List[CandidateRow]] = {}
        universe = subset if subset is not None else self.units
        for unit in universe:
            weight = float(unit.params)
            if weight <= 0:
                continue
            density = predictions.get(unit.name, 0.0)
            uncertainty = uncertainties.get(unit.name, 0.0)
            per_param_value = density + self.beta_uncertainty * uncertainty
            total_value = per_param_value * weight
            key = (getattr(unit, "site_instance", unit.site.name), unit.family)
            rows.setdefault(key, []).append(
                CandidateRow(
                    unit=unit,
                    density=density,
                    per_param_value=per_param_value,
                    total_value=total_value,
                    weight=weight,
                    uncertainty=uncertainty,
                )
            )
        return rows

    def _project_one_per_site(
        self,
        rows: Dict[Tuple[str, str], List[CandidateRow]],
        *,
        prefer_active: bool,
    ) -> List[CandidateRow]:
        projected: List[CandidateRow] = []
        for candidates in rows.values():
            if not candidates:
                continue
            if prefer_active:
                projected.append(
                    max(
                        candidates,
                        key=lambda row: (row.per_param_value, row.unit.state.active),
                    )
                )
                continue
            projected.append(max(candidates, key=lambda row: row.per_param_value))
        return projected

    def _select_projected_units(
        self,
        predictions: Dict[str, float],
        uncertainties: Dict[str, float],
        *,
        locked: Set[PEFTUnit],
        prefer_active: bool,
        respect_activation_thresholds: bool,
        budget_override: Optional[float] = None,
        subset: Optional[Iterable[PEFTUnit]] = None,
    ) -> Tuple[Set[PEFTUnit], Dict[str, AllocationStats]]:
        rows = self._build_candidate_rows(predictions, uncertainties, subset=subset)
        projected_rows = self._project_one_per_site(rows, prefer_active=prefer_active)
        selected: Set[PEFTUnit] = set(locked)
        stats: Dict[str, AllocationStats] = {}
        budget_limit = budget_override if budget_override is not None else self.budget

        for row in sorted(projected_rows, key=lambda item: item.per_param_value, reverse=True):
            unit = row.unit
            if unit in selected:
                continue
            currently_active = unit.state.active

            if respect_activation_thresholds:
                if currently_active and row.per_param_value < -self.hysteresis_down:
                    continue
                if not currently_active and row.per_param_value <= 0.0:
                    continue

            projected_weight = self._selected_params(selected) + row.weight
            if projected_weight > budget_limit and not currently_active:
                continue

            selected.add(unit)
            stats[unit.name] = AllocationStats(
                density=row.density,
                value=row.total_value,
                weight=row.weight,
                uncertainty=row.uncertainty,
            )

        total = self._selected_params(selected)
        if total > budget_limit:
            removable = [u for u in selected if u not in locked]
            removable.sort(
                key=lambda u: stats.get(u.name, AllocationStats(0.0, 0.0, 0.0)).density
            )
            for unit in removable:
                selected.remove(unit)
                stats.pop(unit.name, None)
                total = self._selected_params(selected)
                if total <= budget_limit:
                    break

        return selected, stats


__all__ = ["DiceFirstAllocator", "AllocationDiagnostics"]
from typing import Dict, Iterable, List, Set, Tuple
from .units import PEFTUnit
