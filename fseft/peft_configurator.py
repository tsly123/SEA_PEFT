"""Utilities to apply exported SA-PEFT candidates onto a fresh model."""

from __future__ import annotations

from typing import Iterable

from .config_loader import Candidate, ConfigLoadError


def apply_candidate_to_model(model, candidate: Candidate) -> None:
    """Activate/deactivate PEFT units on ``model`` according to ``candidate``."""

    registry = getattr(model, "unit_registry", None)
    if registry is None:
        raise ConfigLoadError("Model does not expose a unit_registry; call set_model_peft first")

    units_by_name = {}
    for unit in registry:
        units_by_name[unit.name] = unit
        if unit.state.active:
            unit.deactivate()

    unknown = [name for name in candidate.units if name not in units_by_name]
    if unknown:
        raise ConfigLoadError(
            f"Candidate '{candidate.label}' references unknown units: {', '.join(unknown)}"
        )

    for name in candidate.units:
        unit = units_by_name[name]
        if not unit.state.active:
            unit.activate()
            unit.state.guard = 0
            unit.state.promotion_votes = 0
            unit.state.demotion_votes = 0

    # Ensure the registry stored on the model remains iterable (tuple for safety).
    model.unit_registry = tuple(registry)


__all__ = ["apply_candidate_to_model"]
