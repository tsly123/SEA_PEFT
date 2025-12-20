"""Helpers for loading SA-PEFT final configuration exports (YAML / CSV)."""

from __future__ import annotations

import csv
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


class ConfigLoadError(RuntimeError):
    """Raised when exported SA-PEFT configuration files cannot be parsed."""


@dataclass
class Candidate:
    """Represents a single exported SA-PEFT configuration."""

    rank: int
    label: str
    source: str = ""
    units: Sequence[str] = field(default_factory=list)
    dice: float = 0.0
    cvar: Optional[float] = None
    param_fraction: Optional[float] = None
    param_count: Optional[int] = None
    unit_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise unit ordering for stable comparisons / logging.
        self.units = tuple(sorted(self.units))
        if self.unit_count is None:
            self.unit_count = len(self.units)


def load_candidates(path: str | Path) -> List[Candidate]:
    """Load all exported candidates from a YAML or CSV file."""

    file_path = Path(path)
    if not file_path.exists():
        raise ConfigLoadError(f"Configuration file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _load_candidates_from_yaml(file_path)
    if suffix == ".csv":
        return _load_candidates_from_csv(file_path)
    raise ConfigLoadError(f"Unsupported configuration format: {file_path.suffix}")


def select_candidate(candidates: Iterable[Candidate], config_id: int) -> Candidate:
    """Pick a candidate by rank (1-indexed)."""

    for cand in candidates:
        if cand.rank == config_id:
            return cand
    raise ConfigLoadError(f"No candidate with rank={config_id}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_candidates_from_yaml(path: Path) -> List[Candidate]:
    if yaml is None:
        raise ConfigLoadError("PyYAML is required to load YAML configuration files")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict) or "candidates" not in data:
        raise ConfigLoadError(f"Unexpected YAML structure in {path}")

    candidates: List[Candidate] = []
    for raw in data.get("candidates", []):
        candidates.append(_candidate_from_mapping(raw, path))

    return sorted(candidates, key=lambda c: c.rank)


def _load_candidates_from_csv(path: Path) -> List[Candidate]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ConfigLoadError(f"CSV file has no header: {path}")

        candidates: List[Candidate] = []
        for row in reader:
            candidates.append(_candidate_from_csv_row(row, path))

    return sorted(candidates, key=lambda c: c.rank)


def _candidate_from_mapping(raw: Any, path: Path) -> Candidate:
    if not isinstance(raw, dict):
        raise ConfigLoadError(f"Invalid candidate entry in {path}: {raw!r}")

    try:
        rank = int(raw["rank"])
        label = str(raw.get("label", f"candidate_{rank}"))
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ConfigLoadError(f"Missing required rank/label in {path}: {raw}") from exc

    units = raw.get("units") or raw.get("units_list") or []
    if isinstance(units, (set, tuple)):
        units = list(units)
    if not isinstance(units, list):
        raise ConfigLoadError(f"Candidate {label} has invalid units payload: {units!r}")

    metadata = dict(raw)
    metadata.pop("units", None)
    metadata.pop("units_list", None)
    metadata.pop("rank", None)
    metadata.pop("label", None)

    return Candidate(
        rank=rank,
        label=label,
        source=str(raw.get("source", "")),
        units=[str(u) for u in units],
        dice=_coerce_float(raw.get("dice")),
        cvar=_coerce_optional_float(raw.get("cvar")),
        param_fraction=_coerce_optional_float(raw.get("param_fraction")),
        param_count=_coerce_optional_int(raw.get("param_count")),
        unit_count=_coerce_optional_int(raw.get("unit_count")) or len(units),
        metadata=metadata,
    )


def _candidate_from_csv_row(row: Dict[str, str], path: Path) -> Candidate:
    if "rank" not in row or not row["rank"]:
        raise ConfigLoadError(f"CSV row missing rank in {path}: {row}")

    units_col = row.get("units") or row.get("units_list") or ""
    units = [u.strip() for u in units_col.split(";") if u.strip()]

    metadata = dict(row)
    for key in ("rank", "label", "source", "dice", "cvar", "param_fraction", "param_count", "units"):
        metadata.pop(key, None)

    return Candidate(
        rank=int(row["rank"]),
        label=row.get("label", f"candidate_{row['rank']}").strip(),
        source=row.get("source", "").strip(),
        units=units,
        dice=_coerce_float(row.get("dice")),
        cvar=_coerce_optional_float(row.get("cvar")),
        param_fraction=_coerce_optional_float(row.get("param_fraction")),
        param_count=_coerce_optional_int(row.get("param_count")),
        unit_count=_coerce_optional_int(row.get("unit_count")) or len(units),
        metadata=metadata,
    )


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive default
        return 0.0


def _coerce_optional_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):  # pragma: no cover - defensive default
        return None


def _coerce_optional_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):  # pragma: no cover - defensive default
        return None


__all__ = ["Candidate", "ConfigLoadError", "load_candidates", "select_candidate"]
