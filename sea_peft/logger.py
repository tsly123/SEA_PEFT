"""CSV logging utilities for SA-PEFT episodes."""

from __future__ import annotations

import csv
import datetime as dt
import os
import pathlib
import sys
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence
from tqdm.auto import tqdm as _tqdm_factory


class _ProgressWrapper:
    """Thin adapter that wraps iterables with tqdm when enabled."""

    def __init__(
        self,
        enabled: bool,
        factory: Optional[Callable[..., Iterable]],
        total: int,
        description: str,
        leave: bool,
    ) -> None:
        self._enabled = enabled and factory is not None
        self._factory = factory
        self._total = total
        self._description = description
        self._leave = leave
        self._bar = None

    def __call__(self, iterable: Iterable):
        if not self._enabled or self._factory is None:
            return iterable
        self._bar = self._factory(
            iterable,
            total=self._total,
            desc=self._description,
            leave=self._leave,
        )
        return self._bar

    def close(self) -> None:
        if self._bar is not None and hasattr(self._bar, "close"):
            self._bar.close()
        self._bar = None


class CSVLogger:
    def __init__(self, path: pathlib.Path, fieldnames: Iterable[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            return

        with self.path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])

        if not existing_header:
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            return

        missing = [field for field in self.fieldnames if field not in existing_header]
        if not missing:
            self.fieldnames = existing_header
            return

        with self.path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        upgraded_header = existing_header + missing
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=upgraded_header)
            writer.writeheader()
            for row in rows:
                for key in missing:
                    row.setdefault(key, "")
                writer.writerow(row)
        self.fieldnames = upgraded_header

    def write(self, row: Mapping[str, object]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class EpisodeLogger:
    def __init__(
        self,
        root: Optional[pathlib.Path] = None,
        run_name: Optional[str] = None,
        *,
        disable_progress: bool = False,
        force_progress: bool = False,
    ) -> None:
        if root is None:
            root = pathlib.Path("logs")
        if run_name is None:
            run_name = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.root = pathlib.Path(root) / run_name
        self.root.mkdir(parents=True, exist_ok=True)

        env_setting = os.getenv("SA_PEFT_PROGRESS", "").strip().lower()
        env_disable = os.getenv("SA_PEFT_DISABLE_PROGRESS", "").strip().lower()
        env_force = os.getenv("SA_PEFT_FORCE_PROGRESS", "").strip().lower()

        def _truthy(value: str) -> bool:
            return value in {"1", "true", "yes", "on"}

        def _falsey(value: str) -> bool:
            return value in {"0", "false", "no", "off"}

        force_from_env = _truthy(env_force) or _truthy(env_setting)
        disable_from_env = _falsey(env_setting) or _truthy(env_disable)

        self.force_progress = force_progress or force_from_env
        self.disable_progress = disable_progress or (disable_from_env and not self.force_progress)
        self.is_interactive = sys.stdout.isatty() or sys.stderr.isatty()
        progress_available = _tqdm_factory is not None

        if self.force_progress and progress_available:
            self.progress_enabled = True
        elif self.disable_progress or not progress_available:
            self.progress_enabled = False
        else:
            self.progress_enabled = self.is_interactive

        self._progress_factory = _tqdm_factory if self.progress_enabled else None
        if self.progress_enabled and _tqdm_factory is not None:
            self._printer = _tqdm_factory.write
        else:
            self._printer = print

        self.refinement_root = self.root / "refinement"
        self.refinement_root.mkdir(parents=True, exist_ok=True)

        self.training = CSVLogger(
            self.root / "training.csv",
            ["step", "phase", "loss", "dice", "lr", "param_percent"],
        )
        self.audits = CSVLogger(
            self.root / "audits.csv",
            [
                "step",
                "unit",
                "site_instance",
                "topology",
                "level",
                "dice_base",
                "delta_dice",
                "density_raw",
                "utility_smoothed",
                "param_count",
                "validation_sample",
            ],
        )
        self.allocations = CSVLogger(
            self.root / "allocations.csv",
            ["step", "active_params", "budget", "selected_units"],
        )
        self.coverage = CSVLogger(self.root / "coverage.csv", ["step", "unit", "probes"])
        self.final_selection = CSVLogger(
            self.root / "final_selection.csv",
            [
                "rank",
                "label",
                "source",
                "dice",
                "param_fraction",
                "param_count",
                "unit_count",
                "qbuf_batches",
                "projection_delta",
                "projection_valid",
                "history_dice",
                "units",
            ],
        )
        self.fsm_transitions = CSVLogger(
            self.root / "fsm_transitions.csv",
            [
                "step",
                "unit",
                "site",
                "from_state",
                "to_state",
                "votes_required",
                "votes_recorded",
            ],
        )
        self.refinement_ranks = CSVLogger(
            self.root / "refinement_ranks.csv",
            [
                "rank",
                "label",
                "source",
                "dice",
                "param_fraction",
                "param_count",
                "unit_count",
                "config_path",
            ],
        )
        self.lr_restarts = CSVLogger(
            self.root / "lr_restarts.csv",
            [
                "step",
                "phase",
                "scheduler_type",
                "restart_count",
                "T_cur",
                "T_i",
                "adapter_lr_min",
                "adapter_lr_max",
                "adapter_lr_mean",
            ],
        )

    def log_training(
        self,
        step: int,
        phase: str,
        loss: float,
        dice: float,
        lr: float,
        param_percent: float,
    ) -> None:
        self.training.write(
            {
                "step": step,
                "phase": phase,
                "loss": loss,
                "dice": dice,
                "lr": lr,
                "param_percent": param_percent,
            }
        )

    def log_audit(
        self,
        step: int,
        unit: str,
        site_instance: str,
        topology: str,
        level: str,
        dice_base: float,
        delta_dice: float,
        density_raw: float,
        utility_smoothed: float,
        params: int,
        validation_sample: str = "",
    ) -> None:
        self.audits.write(
            {
                "step": step,
                "unit": unit,
                "site_instance": site_instance,
                "topology": topology,
                "level": level,
                "dice_base": dice_base,
                "delta_dice": delta_dice,
                "density_raw": density_raw,
                "utility_smoothed": utility_smoothed,
                "param_count": params,
                "validation_sample": validation_sample,
            }
        )

    def log_allocation(
        self,
        step: int,
        active_params: float,
        budget: float,
        selected: Iterable[str],
    ) -> None:
        self.allocations.write(
            {
                "step": step,
                "active_params": active_params,
                "budget": budget,
                "selected_units": ",".join(selected),
            }
        )

    def log_coverage(self, step: int, probes: Dict[str, int]) -> None:
        for unit, count in probes.items():
            self.coverage.write({"step": step, "unit": unit, "probes": count})

    def log_final_selection(
        self,
        rank: int,
        label: str,
        source: str = "",
        dice: float = 0.0,
        param_fraction: float = 0.0,
        param_count: Optional[float] = None,
        unit_count: int = 0,
        qbuf_batches: Optional[int] = None,
        projection_delta: Optional[float] = None,
        projection_valid: Optional[bool] = None,
        history_dice: Optional[float] = None,
        units: Sequence[str] = (),
    ) -> None:
        row = {
            "rank": rank,
            "label": label,
            "source": source,
            "dice": dice,
            "param_fraction": param_fraction,
            "param_count": "" if param_count is None else param_count,
            "unit_count": unit_count,
            "qbuf_batches": "" if qbuf_batches is None else qbuf_batches,
            "projection_delta": "" if projection_delta is None else projection_delta,
            "projection_valid": "" if projection_valid is None else int(projection_valid),
            "history_dice": "" if history_dice is None else history_dice,
            "units": ";".join(units),
        }
        self.final_selection.write(row)

    def progress(self, total: int, description: str, *, transient: bool = True) -> _ProgressWrapper:
        return _ProgressWrapper(
            enabled=self.progress_enabled,
            factory=self._progress_factory,
            total=total,
            description=description,
            leave=not transient,
        )

    def console(self, message: str) -> None:
        self._printer(message)

    def log_fsm_transition(
        self,
        *,
        step: int,
        unit: str,
        site: str,
        from_state: str,
        to_state: str,
        votes_required: int,
        votes_recorded: int,
    ) -> None:
        self.fsm_transitions.write(
            {
                "step": step,
                "unit": unit,
                "site": site,
                "from_state": from_state,
                "to_state": to_state,
                "votes_required": votes_required,
                "votes_recorded": votes_recorded,
            }
        )

    def refinement_rank_dir(self, rank: int) -> pathlib.Path:
        path = self.refinement_root / f"refinement_rank_{rank}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def log_refinement_rank(
        self,
        *,
        rank: int,
        label: str,
        source: str,
        dice: float,
        param_fraction: Optional[float],
        param_count: Optional[int],
        unit_count: int,
        config_path: pathlib.Path,
    ) -> None:
        self.refinement_ranks.write(
            {
                "rank": rank,
                "label": label,
                "source": source,
                "dice": dice,
                "param_fraction": "" if param_fraction is None else param_fraction,
                "param_count": "" if param_count is None else param_count,
                "unit_count": unit_count,
                "config_path": str(config_path),
            }
        )

    def log_lr_restart(
        self,
        *,
        step: int,
        phase: str,
        scheduler_type: str,
        restart_count: int,
        T_cur: int,
        T_i: int,
        adapter_lr_min: float,
        adapter_lr_max: float,
        adapter_lr_mean: float,
    ) -> None:
        self.lr_restarts.write(
            {
                "step": step,
                "phase": phase,
                "scheduler_type": scheduler_type,
                "restart_count": restart_count,
                "T_cur": T_cur,
                "T_i": T_i,
                "adapter_lr_min": adapter_lr_min,
                "adapter_lr_max": adapter_lr_max,
                "adapter_lr_mean": adapter_lr_mean,
            }
        )


__all__ = ["EpisodeLogger"]
