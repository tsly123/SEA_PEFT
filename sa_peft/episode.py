"""Episode runner for SA-PEFT search."""

from __future__ import annotations

import copy
import csv
import itertools
import json
import pathlib
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Set

import torch
import yaml
from torch import nn

from fseft.datasets.dataset import get_loader
from fseft.utils import load_model, set_model_peft
from models.configs import get_model_config
from utils.losses import BinaryDice3D
from utils.misc import set_seeds
from utils.scheduler import LinearWarmupCosineAnnealingLR, CosineAnnealingWarmRestarts

from .allocator import DiceFirstAllocator
from .logger import EpisodeLogger
from .manifest import Manifest, load_manifest
from .sampler import AuditSampler
from .units import PEFTUnit, UnitRegistry
from .utility import UtilityPredictor


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

@dataclass
class EpisodeConfig:
    warmup_steps: int = 200
    episode_steps: int = 6000
    refinement_steps: int = 200
    audit_cadence: int = 200
    allocation_cadence: int = 400
    audit_sample_size: int = 0
    audit_fraction: float = 0.20
    coverage_target: int = 5
    debt_horizon: int = 6
    param_budget: float = 0.01
    beta_uncertainty: float = 0.5
    guard_cycles: int = 3
    hysteresis_down: float = 0.02
    use_promotion_fsm: bool = True
    fsm_promotion_votes: int = 3
    fsm_demotion_votes: int = 3
    fsm_log_transitions: bool = False
    top_k: int = 5
    max_single_swaps: int = 5
    # LR scheduler configuration
    use_adapter_lr_restarts: bool = True
    adapter_restart_steps: Optional[int] = None  # Defaults to audit_cadence if None


class SAPEFTEpisode:
    def __init__(
        self,
        args,
        manifest_path: Optional[str] = None,
        config: Optional[EpisodeConfig] = None,
        run_name: Optional[str] = None,
        log_root: Optional[pathlib.Path] = None,
        disable_progress: Optional[bool] = None,
        force_progress: Optional[bool] = None,
    ) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seeds(42, use_cuda=self.device.type == "cuda")
        self.manifest: Manifest = load_manifest(manifest_path)
        self.config = config or EpisodeConfig()
        log_root_arg = log_root or getattr(self.args, "log_root", None)
        if log_root_arg is None:
            out_path = getattr(self.args, "out_path", None)
            if out_path is not None:
                log_root_arg = pathlib.Path(out_path) / "logs"
            else:
                log_root_arg = pathlib.Path("logs")
        else:
            log_root_arg = pathlib.Path(log_root_arg)

        disable_flag = getattr(self.args, "disable_progress", False)
        if disable_progress is not None:
            disable_flag = disable_progress

        force_flag = getattr(self.args, "force_progress", False)
        if force_progress is not None:
            force_flag = force_progress

        self.logger = EpisodeLogger(
            root=log_root_arg,
            run_name=run_name,
            disable_progress=disable_flag,
            force_progress=force_flag,
        )
        self.args.log_root = log_root_arg
        self.args.disable_progress = disable_flag
        self.args.force_progress = force_flag
        self.phase = "warmup"
        get_model_config(self.args)
        self.model = self._build_model()
        if args.lr is not None:
            args.adapt_hp["lr"] = args.lr

        self.registry = UnitRegistry(self.manifest, self.model)
        self.units: List[PEFTUnit] = list(self.registry)
        # Expose registry so retraining scripts can reuse configurations (Phase 3)
        self.model.unit_registry = self.units
        self.unit_lookup: Dict[str, PEFTUnit] = {unit.name: unit for unit in self.units}
        self.total_params = _count_parameters(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.adapt_hp["lr"], weight_decay=0.0)

        # Classifier scheduler (existing LinearWarmupCosineAnnealingLR)
        self.scheduler = LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=max(1, self.config.warmup_steps // 10),
            max_epochs=self.config.episode_steps + self.config.refinement_steps,
            warmup_start_lr=self.args.adapt_hp["lr"] / max(1, self.config.warmup_steps // 10),
        )

        # Adapter scheduler (CosineAnnealingWarmRestarts) - will be initialized after warmup
        self.adapter_scheduler: Optional[CosineAnnealingWarmRestarts] = None
        self.adapter_param_group_indices: List[int] = []
        self.train_loader = get_loader(self.args)
        self.train_iter = itertools.cycle(self.train_loader)
        self.criterion = BinaryDice3D()
        self.query_buffer: Deque[Dict[str, torch.Tensor]] = deque(maxlen=4)
        self.audit_sample_size = self.config.audit_sample_size or max(1, int(round(len(self.units) * self.config.audit_fraction)))
        ema_decay = 0.95 if getattr(self.args, "k", 1) == 1 else 0.93
        self.sampler = AuditSampler(
            [unit.name for unit in self.units],
            coverage_target=self.config.coverage_target,
            debt_horizon=self.config.debt_horizon,
        )
        self.utility = UtilityPredictor([unit.name for unit in self.units], ema_decay=ema_decay)
        self.allocator = DiceFirstAllocator(
            self.units,
            total_params=self.total_params,
            budget_fraction=self.config.param_budget,
            guard_cycles=self.config.guard_cycles,
            hysteresis_down=self.config.hysteresis_down,
            beta_uncertainty=self.config.beta_uncertainty,
            use_promotion_fsm=self.config.use_promotion_fsm,
            fsm_promotion_votes=self.config.fsm_promotion_votes,
            fsm_demotion_votes=self.config.fsm_demotion_votes,
        )
        self.audit_counter = 0
        self._tracked_params = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        self.model.to(self.device)
        # Phase 2: Allocation history tracking
        self.hist_sets: Deque[Set[str]] = deque(maxlen=self.config.top_k)
        self.hist_metrics: Deque[float] = deque(maxlen=self.config.top_k)
        self.validation_buffer: List[Dict[str, torch.Tensor]] = []
        # Deterministic warmup sequence
        self.warmup_sequence: List[PEFTUnit] = []
        self.warmup_sequence_index: int = 0
        # Final config bookkeeping
        self._final_config_candidates: List[Dict] = []
        self._canonical_final_configs_path: Optional[pathlib.Path] = None

    def _build_model(self) -> nn.Module:
        model = load_model(self.args)
        model = set_model_peft(model, self.args)
        return model

    def _ensure_optimizer_params(self, unit: PEFTUnit) -> None:
        params = [p for p in unit.parameters() if p.requires_grad]
        if not params:
            return
        for param in params:
            if id(param) not in self._tracked_params:
                lr = self.args.adapt_hp["lr"]
                if unit.family == "adaptformer" and self.phase == "warmup":
                    lr *= 0.5

                # Mark this param group as an adapter group
                is_adapter = unit.family in {"lora", "adapter", "adaptformer", "ia3"}
                group_idx = len(self.optimizer.param_groups)
                self.optimizer.add_param_group({
                    "params": [param],
                    "lr": lr,
                    "is_adapter": is_adapter,
                })
                self._tracked_params.add(id(param))

                # Track adapter param group indices
                if is_adapter and group_idx not in self.adapter_param_group_indices:
                    self.adapter_param_group_indices.append(group_idx)

    def _next_batch(self) -> Dict[str, torch.Tensor]:
        batch = next(self.train_iter)
        self.query_buffer.append(
            {
                "image": batch["image"].detach().cpu(),
                "label": batch["label"].detach().cpu(),
            }
        )
        return batch

    def _forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["image"].to(self.device).float()
        y = batch["label"].to(self.device).float()
        logits = self.model(x)
        if self.args.objective == "binary":
            pred = torch.sigmoid(logits)
        else:
            pred = torch.softmax(logits, dim=1)
        return self.criterion(pred, y)

    def _validation_batches(self) -> Iterable[Dict[str, torch.Tensor]]:
        """Generate fresh validation batches from stored metadata."""
        entries = getattr(self.args, "validation_entries", [])
        transform = getattr(self.args, "validation_transform", None)
        if not entries or transform is None:
            return
        for entry in entries:
            sample = transform(entry)
            if isinstance(sample, dict):
                samples = [sample]
            elif isinstance(sample, (list, tuple)):
                samples = list(sample)
            else:
                raise TypeError(
                    "validation_transform must return a dict or a list/tuple of dicts"
                )

            for idx, item in enumerate(samples):
                if not isinstance(item, dict):
                    raise TypeError(
                        "validation_transform returned non-dict element; expected dict with 'image' and 'label'"
                    )

                name = entry.get("name", "unknown")
                if len(samples) > 1:
                    name = f"{name}_crop{idx}"

                yield {
                    "image": item["image"].unsqueeze(0).cpu(),
                    "label": item["label"].unsqueeze(0).cpu(),
                    "name": name,
                }

    def _train_step(self, step: int, phase: str) -> float:
        batch = self._next_batch()
        loss = self._forward(batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Advance schedulers
        self.scheduler.step()
        if phase != "warmup" and self.adapter_scheduler is not None:
            self.adapter_scheduler.step()

            # Log restart events *after* stepping, so T_cur reflects the new state
            if self.adapter_scheduler.T_cur == 0:
                adapter_lrs = [
                    self.optimizer.param_groups[idx]["lr"]
                    for idx in self.adapter_param_group_indices
                ]
                if adapter_lrs:
                    restart_period = self.config.adapter_restart_steps or self.config.audit_cadence
                    restart_period = max(1, restart_period)
                    restart_count = ((step - self.config.warmup_steps) // restart_period) + 1
                    self.logger.log_lr_restart(
                        step=step,
                        phase=phase,
                        scheduler_type="CosineAnnealingWarmRestarts",
                        restart_count=restart_count,
                        T_cur=self.adapter_scheduler.T_cur,
                        T_i=self.adapter_scheduler.T_i,
                        adapter_lr_min=min(adapter_lrs),
                        adapter_lr_max=max(adapter_lrs),
                        adapter_lr_mean=sum(adapter_lrs) / len(adapter_lrs),
                    )
        dice_val = 1.0 - loss.item()
        lr = self.optimizer.param_groups[0]["lr"]
        param_percent = self.registry.active_param_count() / self.total_params
        self.logger.log_training(step, phase, loss.item(), dice_val, lr, param_percent)
        return loss.item()

    def _build_warmup_sequence(self) -> List[PEFTUnit]:
        """Build deterministic warmup sequence grouped by block -> family -> level."""
        sequence: List[PEFTUnit] = []
        units_by_block = self.registry.units_by_block()
        # Sort blocks for determinism
        sorted_blocks = sorted(units_by_block.keys())

        for block_name in sorted_blocks:
            block_units = units_by_block[block_name]
            # Group by family
            by_family: Dict[str, List[PEFTUnit]] = {}
            for unit in block_units:
                by_family.setdefault(unit.family, []).append(unit)

            # Sort families for determinism
            for family in sorted(by_family.keys()):
                family_units = by_family[family]
                # Sort by level_index for determinism
                family_units.sort(key=lambda u: u.level_index)
                # Add only the first unit per family per block (lowest level)
                if family_units:
                    sequence.append(family_units[0])

        return sequence

    def _activate_warmup_units(self) -> List[PEFTUnit]:
        """Deterministic round-robin warmup activation with wrap-around."""
        if not self.warmup_sequence:
            return []

        # Get next unit in sequence (with wrap-around)
        unit = self.warmup_sequence[self.warmup_sequence_index]
        self.warmup_sequence_index = (self.warmup_sequence_index + 1) % len(self.warmup_sequence)

        # Activate unit
        if not unit.state.active:
            unit.activate()
            self._ensure_optimizer_params(unit)

        return [unit]

    def warmup(self) -> None:
        # Phase 2: Capture validation buffer at episode start
        self.validation_buffer = list(self._validation_batches())
        # Build deterministic warmup sequence
        self.warmup_sequence = self._build_warmup_sequence()
        self.warmup_sequence_index = 0

        self.model.train()
        progress = self.logger.progress(self.config.warmup_steps, "Warmup")
        progress_iterator = progress(range(1, self.config.warmup_steps + 1)) if callable(progress) else range(1, self.config.warmup_steps + 1)
        for step in progress_iterator:
            active_units = self._activate_warmup_units()
            self._train_step(step, phase="warmup")
            for unit in active_units:
                unit.deactivate()
        for group in self.optimizer.param_groups:
            group["lr"] = self.args.adapt_hp["lr"]

        # Initialize adapter scheduler after warmup if enabled
        if self.config.use_adapter_lr_restarts and self.adapter_param_group_indices:
            restart_period = self.config.adapter_restart_steps or self.config.audit_cadence
            restart_period = max(1, restart_period)
            self.adapter_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=restart_period,
                T_mult=1,
                eta_min=0.0,
                param_group_indices=self.adapter_param_group_indices,
            )
            self.logger.console(f"→ Adapter LR scheduler initialized with restart period={restart_period}")

        self.phase = "search"
        progress.close()
        self.logger.console("✓ Warmup complete")

    def _evaluate_batches(self, batches: Iterable[Dict[str, torch.Tensor]]) -> float:
        self.model.eval()
        scores: List[float] = []
        with torch.no_grad():
            for batch in batches:
                x = batch["image"].to(self.device).float()
                y = batch["label"].to(self.device).float()
                logits = self.model(x)
                if self.args.objective == "binary":
                    pred = torch.sigmoid(logits)
                else:
                    pred = torch.softmax(logits, dim=1)
                scores.append(float(1.0 - self.criterion(pred, y).item()))
        self.model.train()
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _audit(self, step: int) -> None:
        self.audit_counter += 1
        uniform_only = self.audit_counter <= 3
        uncertainties = self.utility.residuals()
        selection = self.sampler.sample(self.audit_sample_size, uncertainties, uniform_only=uniform_only)
        if not selection:
            return
        # Phase 2: Use validation buffer instead of regenerating batches
        audit_batches = self.validation_buffer if self.validation_buffer else list(self._validation_batches())
        if not audit_batches:
            return
        validation_sample_names = ",".join([batch.get("name", "unknown") for batch in audit_batches])
        dice_base = self._evaluate_batches(audit_batches)
        for unit_name in selection:
            unit = self.unit_lookup[unit_name]
            prev_active = unit.state.active
            prev_guard = unit.state.guard

            if prev_active:
                dice_on = dice_base
                unit.deactivate()
                dice_off = self._evaluate_batches(audit_batches)
                unit.activate()
                unit.state.guard = prev_guard
                delta_dice = dice_on - dice_off
            else:
                unit.activate()
                dice_on = self._evaluate_batches(audit_batches)
                unit.deactivate()
                delta_dice = dice_on - dice_base
                dice_off = dice_base

            density_raw = delta_dice / float(max(unit.params, 1))
            # density_raw = delta_dice
            smoothed = self.utility.update(unit.name, density_raw)
            self.logger.log_audit(
                step=step,
                unit=unit.name,
                site_instance=getattr(unit, "site_instance", unit.site.name),
                topology=unit.topology,
                level=unit.level.name,
                dice_base=dice_base,
                delta_dice=delta_dice,
                density_raw=density_raw,
                utility_smoothed=smoothed,
                params=unit.params,
                validation_sample=validation_sample_names,
            )
        self.sampler.record(selection)
        self.logger.log_coverage(step, self.sampler.coverage_snapshot())

    def _collect_predictions(self, step: int) -> Dict[str, float]:
        predictions: Dict[str, float] = {}
        for unit in self.units:
            predictions[unit.name] = self.utility.predict(unit.name)
        return predictions

    def search(self) -> None:
        self.model.train()
        progress = self.logger.progress(self.config.episode_steps, "Search")
        progress_iterator = progress(range(1, self.config.episode_steps + 1)) if callable(progress) else range(1, self.config.episode_steps + 1)
        for step in progress_iterator:
            global_step = step + self.config.warmup_steps
            self._train_step(global_step, phase="search")
            if step % self.config.audit_cadence == 0:
                self._audit(global_step)
                self.logger.console(f"→ Audit complete at step {global_step}")
            if step % self.config.allocation_cadence == 0:
                preds = self._collect_predictions(global_step)
                uncertainties = self.utility.residuals()
                desired = self.allocator.allocate(step, preds, uncertainties)
                activated, _, transitions = self.allocator.apply(desired)
                active_names = [unit.name for unit in self.units if unit.state.active]
                self.sampler.update_active_units(active_names)
                active_frac = self.allocator.active_params() / self.total_params if self.total_params else 0.0
                selected_names = [unit.name for unit in sorted(desired, key=lambda u: u.name)]
                self.logger.log_allocation(global_step, active_frac, self.config.param_budget, selected_names)
                self.logger.console(
                    f"[alloc step {global_step}] active params {active_frac:.2%} budget {self.config.param_budget:.2%}"
                )
                if self.config.use_promotion_fsm:
                    for unit, transition in transitions:
                        from_state, to_state, votes_required, votes_recorded = transition
                        self.logger.log_fsm_transition(
                            step=global_step,
                            unit=unit.name,
                            site=unit.site.name,
                            from_state=from_state,
                            to_state=to_state,
                            votes_required=votes_required,
                            votes_recorded=votes_recorded,
                        )
                # Phase 2: Track allocation history
                self.hist_sets.append(set(selected_names))
                # Evaluate current allocation on validation buffer to track metrics
                if self.validation_buffer:
                    dice_metric = self._evaluate_batches(self.validation_buffer)
                    self.hist_metrics.append(dice_metric)
                for unit in activated:
                    self._ensure_optimizer_params(unit)
        progress.close()
        self.logger.console("✓ Search complete")

    def _select_final_configs(self) -> List[Dict]:
        """Phase 2: Execute candidate sweep and return evaluation metadata for all candidates.

        Returns ranked list of all candidates with metadata, where top-K are the best,
        but guard-free and single-swap candidates are included even if outside top-K.
        """
        candidates: List[Dict] = []

        # Stage 1: Add last K allocations from history
        for idx, (unit_set, dice_metric) in enumerate(zip(self.hist_sets, self.hist_metrics)):
            candidates.append({
                "label": f"hist_{idx}",
                "units": unit_set,
                "source": "history",
                "history_dice": dice_metric,
            })

        # Stage 2: Guard-free re-solve (disable guards/hysteresis/FSM)
        if self.hist_sets:
            # Get current predictions and uncertainties
            preds = {unit.name: self.utility.predict(unit.name) for unit in self.units}
            uncertainties = self.utility.residuals()

            # Create a temporary allocator without guards for clean greedy selection
            guard_free_allocator = DiceFirstAllocator(
                self.units,
                total_params=self.total_params,
                budget_fraction=self.config.param_budget,
                guard_cycles=0,  # Disable guards
                hysteresis_down=0.0,  # Disable hysteresis
                beta_uncertainty=self.config.beta_uncertainty,
            )

            # Run allocation without training-time mechanisms
            guard_free_set = guard_free_allocator.allocate(0, preds, uncertainties)
            guard_free_names = set(unit.name for unit in guard_free_set)

            candidates.append({
                "label": "guard_free",
                "units": guard_free_names,
                "source": "guard_free",
                "history_dice": None,
            })

        # Stage 3: Single-swap neighbors (up to max_single_swaps)
        if self.hist_sets:
            most_recent = self.hist_sets[-1]
            preds = {unit.name: self.utility.predict(unit.name) for unit in self.units}

            # Build excluded and included lists sorted by robust score (using utility predictions)
            excluded_units = [(name, preds.get(name, 0.0)) for name in [u.name for u in self.units] if name not in most_recent]
            included_units = [(name, preds.get(name, 0.0)) for name in most_recent]

            excluded_units.sort(key=lambda x: x[1], reverse=True)  # Highest score first
            included_units.sort(key=lambda x: x[1])  # Lowest score first

            swap_count = 0
            for excluded_name, _ in excluded_units:
                if swap_count >= self.config.max_single_swaps:
                    break

                excluded_unit = self.unit_lookup[excluded_name]

                for included_name, _ in included_units:
                    included_unit = self.unit_lookup[included_name]

                    # Check if swap is valid (budget and site exclusivity)
                    swap_set = (most_recent - {included_name}) | {excluded_name}

                    # Check budget constraint
                    swap_params = sum(self.unit_lookup[name].params for name in swap_set)
                    if swap_params > self.config.param_budget * self.total_params:
                        continue

                    # Check site exclusivity (one variant per site per family)
                    valid_swap = True
                    site_family_map: Dict[tuple, str] = {}
                    for name in swap_set:
                        unit = self.unit_lookup[name]
                        key = (unit.site.name, unit.family)
                        if key in site_family_map:
                            valid_swap = False
                            break
                        site_family_map[key] = name

                    if valid_swap:
                        candidates.append({
                            "label": f"swap_{swap_count}",
                            "units": swap_set,
                            "source": "single_swap",
                            "history_dice": None,
                        })
                        swap_count += 1
                        break  # Found valid swap for this excluded unit

        # Stage 4: Evaluate all candidates on validation buffer
        for candidate in candidates:
            units_set = candidate["units"]

            # Temporarily set gates to match candidate configuration
            original_states = {}
            for unit in self.units:
                original_states[unit.name] = unit.state.active
                if unit.name in units_set:
                    if not unit.state.active:
                        unit.activate()
                else:
                    if unit.state.active:
                        unit.deactivate()

            # Evaluate on validation buffer
            dice_score = self._evaluate_batches(self.validation_buffer) if self.validation_buffer else 0.0
            param_count = sum(self.unit_lookup[name].params for name in units_set)
            param_fraction = param_count / self.total_params if self.total_params else 0.0

            # Calculate sum of utilities (robust scores) for the candidate
            utility_sum = sum(self.utility.predict(name) for name in units_set)

            # Calculate vote-rate sum (frequency in last K allocations)
            vote_rate_sum = 0.0
            for name in units_set:
                vote_count = sum(1 for hist_set in self.hist_sets if name in hist_set)
                vote_rate_sum += vote_count / len(self.hist_sets) if self.hist_sets else 0.0

            # Restore original gates
            for unit in self.units:
                if original_states[unit.name]:
                    if not unit.state.active:
                        unit.activate()
                else:
                    if unit.state.active:
                        unit.deactivate()

            # Store evaluation results
            candidate["dice"] = dice_score
            candidate["param_fraction"] = param_fraction
            candidate["param_count"] = param_count
            candidate["utility_sum"] = utility_sum
            candidate["vote_rate_sum"] = vote_rate_sum
            candidate["unit_count"] = len(units_set)
            candidate["units_list"] = sorted(units_set)
            candidate.setdefault("history_dice", dice_score)

            budget_params = self.config.param_budget * self.total_params if self.total_params else 0.0
            delta = param_count - budget_params
            candidate["projection_delta"] = delta
            candidate["projection_valid"] = delta <= 1e-6

        # Stage 5: Rank candidates using tie-breakers (Dice → param_fraction → vote_rate → utility_sum)
        # Higher Dice is better, lower param_fraction is better (tie), higher vote_rate is better (tie)
        candidates.sort(
            key=lambda c: (
                -c["dice"],  # Higher Dice is better (negate for ascending sort)
                c["param_fraction"],  # Lower param fraction is better
                -c["vote_rate_sum"],  # Higher vote rate is better
                -c["utility_sum"],  # Higher utility sum is better
            )
        )

        # Add rank to each candidate
        for rank, candidate in enumerate(candidates, start=1):
            candidate["rank"] = rank

        return candidates

    def _serialize_candidate_for_export(self, candidate: Dict) -> Dict:
        """Return a JSON/YAML-safe copy of a candidate."""
        serializable = dict(candidate)
        units_list = candidate.get("units_list")
        if units_list is None:
            source_units = candidate.get("units", set())
            units_list = sorted(source_units)
            serializable["units_list"] = list(units_list)
        else:
            # Ensure list instance
            serializable["units_list"] = list(units_list)

        # Ensure units are serialized as a list
        serializable["units"] = list(serializable["units_list"])
        return serializable

    def _write_canonical_final_configs(self, candidates: List[Dict]) -> pathlib.Path:
        """Write canonical final_configs.yaml at the logger root."""
        canonical_path = self.export_final_configs_yaml(candidates, output_dir=self.logger.root)
        return canonical_path

    def ensure_canonical_final_configs(self) -> pathlib.Path:
        """Ensure the canonical final_configs.yaml exists; recreate if missing."""
        canonical_path = self.logger.root / "final_configs.yaml"
        if canonical_path.exists():
            return canonical_path
        if not self._final_config_candidates:
            raise RuntimeError("No final configuration data available to rebuild canonical final_configs.yaml")
        self.logger.console("⚠ Canonical final_configs.yaml missing; rebuilding export.")
        canonical_path = self._write_canonical_final_configs(self._final_config_candidates)
        self._canonical_final_configs_path = canonical_path
        return canonical_path

    def export_final_configs_yaml(self, candidates: List[Dict], output_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
        """Phase 2: Export all candidate configurations to YAML.

        Exports all candidates with their metadata, including guard-free and single-swap
        candidates even if outside top-K.
        """
        if output_dir is None:
            output_dir = self.logger.root

        output_path = output_dir / "final_configs.yaml"

        # Prepare export data
        export_data = {
            "top_k": self.config.top_k,
            "budget_fraction": self.config.param_budget,
            "total_params": self.total_params,
            "candidates": []
        }

        for candidate in candidates:
            serial = self._serialize_candidate_for_export(candidate)
            export_data["candidates"].append({
                "rank": serial["rank"],
                "label": serial["label"],
                "source": serial["source"],
                "dice": float(serial["dice"]),
                "param_fraction": float(serial["param_fraction"]),
                "param_count": int(serial["param_count"]),
                "unit_count": int(serial["unit_count"]),
                "utility_sum": float(serial["utility_sum"]),
                "vote_rate_sum": float(serial["vote_rate_sum"]),
                "units": serial["units"],
            })

        # Write YAML
        with output_path.open("w", encoding="utf-8") as f:
            yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)

        return output_path

    def export_final_configs_csv(self, candidates: List[Dict], output_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
        """Phase 2: Export all candidate configurations to CSV.

        Exports all candidates with their metadata, including guard-free and single-swap
        candidates even if outside top-K.
        """
        if output_dir is None:
            output_dir = self.logger.root

        output_path = output_dir / "final_configs.csv"

        # Write CSV
        with output_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "rank", "label", "source", "dice", "param_fraction", "param_count",
                "unit_count", "utility_sum", "vote_rate_sum", "units"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for candidate in candidates:
                writer.writerow({
                    "rank": candidate["rank"],
                    "label": candidate["label"],
                    "source": candidate["source"],
                    "dice": candidate["dice"],
                    "param_fraction": candidate["param_fraction"],
                    "param_count": candidate["param_count"],
                    "unit_count": candidate["unit_count"],
                    "utility_sum": candidate["utility_sum"],
                    "vote_rate_sum": candidate["vote_rate_sum"],
                    "units": ";".join(candidate["units_list"]),
                })

        return output_path

    def refine(self) -> None:
        self.logger.console("Starting refinement phase: main_peft...")
        self.logger.console("Delegate refinement to main_fseft.process using canonical final_configs.yaml.")
        from fseft.config_loader import ConfigLoadError
        import main_fseft

        config_yaml = self.logger.root / "final_configs.yaml"
        if not config_yaml.exists():
            msg = f"Canonical configuration file not found: {config_yaml}"
            self.logger.console(f"✗ {msg}")
            raise ConfigLoadError(msg)

        args_copy = copy.copy(self.args)
        args_copy.config_file = str(config_yaml)
        args_copy.config_id = getattr(self.args, "refinement_rank_id", 1)
        args_copy.epochs = getattr(self.args, "refinement_epochs", self.config.refinement_steps)

        self.logger.console(
            f"→ Delegating refinement to main_fseft.process with config={config_yaml}, "
            f"rank={args_copy.config_id}, epochs={args_copy.epochs}"
        )

        main_fseft.process(args_copy)

        # self.logger.console("Starting refinement phase: cycle step...")
        # self.phase = "refine"
        # self.model.train()
        # progress = self.logger.progress(self.config.refinement_steps, "Refine")
        # progress_iterator = progress(range(1, self.config.refinement_steps + 1)) if callable(progress) else range(1, self.config.refinement_steps + 1)
        # for idx in progress_iterator:
        #     step = self.config.warmup_steps + self.config.episode_steps + idx
        #     self._train_step(step, phase="refine")
        # if hasattr(progress, "close"):
        #     progress.close()

        self.logger.console("✓ Refinement complete")

    def run(self) -> None:
        """Phase 2: Run full SA-PEFT episode with final config selection and export."""
        self.warmup()
        self.search()

        # Phase 2: Execute candidate sweep before refinement
        if self.hist_sets:
            print("Phase 2: Executing candidate sweep for final configuration selection...")
            candidates = self._select_final_configs()

            # Log all candidates to final_selection.csv via logger
            for candidate in candidates:
                self.logger.log_final_selection(
                    rank=candidate["rank"],
                    label=candidate["label"],
                    source=candidate.get("source", ""),
                    dice=candidate.get("dice", 0.0),
                    param_fraction=candidate.get("param_fraction", 0.0),
                    param_count=candidate.get("param_count"),
                    unit_count=candidate.get("unit_count", 0),
                    qbuf_batches=len(self.query_buffer),
                    projection_delta=candidate.get("projection_delta"),
                    projection_valid=candidate.get("projection_valid"),
                    history_dice=candidate.get("history_dice"),
                    units=candidate.get("units_list", []),
                )

            # Export to YAML and CSV
            self._final_config_candidates = candidates
            canonical_yaml_path = self._write_canonical_final_configs(candidates)
            self._canonical_final_configs_path = canonical_yaml_path
            print(f"Phase 2: Exported canonical final configs to {canonical_yaml_path}")

            yaml_root = self.logger.root / "final_configs"
            yaml_root.mkdir(parents=True, exist_ok=True)
            yaml_path = self.export_final_configs_yaml(candidates, output_dir=yaml_root)
            csv_path = self.export_final_configs_csv(candidates, output_dir=yaml_root)
            print(f"Phase 2: Exported final configs to {yaml_path} and {csv_path}")

            # Apply the best candidate (rank 1) for refinement
            if candidates:
                best_candidate = candidates[0]
                best_units = best_candidate["units"]
                print(f"Phase 2: Applying best candidate '{best_candidate['label']}' "
                      f"(Dice={best_candidate['dice']:.4f}, {best_candidate['unit_count']} units)")

                # Set gates to match best configuration
                for unit in self.units:
                    if unit.name in best_units:
                        if not unit.state.active:
                            unit.activate()
                            self._ensure_optimizer_params(unit)
                    else:
                        if unit.state.active:
                            unit.deactivate()

            refinement_dir = self.logger.refinement_rank_dir(best_candidate["rank"])
            rank_yaml = refinement_dir / "config.yaml"
            rank_csv = refinement_dir / "config.csv"
            rank_json = refinement_dir / "config.json"

            # Write per-rank files for consumption by refinement pipeline
            serial_best = self._serialize_candidate_for_export(best_candidate)
            with rank_yaml.open("w", encoding="utf-8") as f_yaml:
                yaml.dump({"candidate": serial_best}, f_yaml, sort_keys=False)
            with rank_csv.open("w", newline="", encoding="utf-8") as f_csv:
                fieldnames = [
                    "rank",
                    "label",
                    "source",
                    "dice",
                    "param_fraction",
                    "param_count",
                    "unit_count",
                    "units",
                ]
                writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "rank": serial_best["rank"],
                        "label": serial_best["label"],
                        "source": serial_best["source"],
                        "dice": serial_best["dice"],
                        "param_fraction": serial_best["param_fraction"],
                        "param_count": serial_best["param_count"],
                        "unit_count": serial_best["unit_count"],
                        "units": ";".join(serial_best["units"]),
                    }
                )
            with rank_json.open("w", encoding="utf-8") as f_json:
                json.dump(serial_best, f_json, indent=2)

            self.logger.log_refinement_rank(
                rank=best_candidate["rank"],
                label=best_candidate["label"],
                source=best_candidate["source"],
                dice=best_candidate["dice"],
                param_fraction=best_candidate.get("param_fraction"),
                param_count=best_candidate.get("param_count"),
                unit_count=best_candidate.get("unit_count", 0),
                config_path=rank_yaml,
            )

        self.refine()


__all__ = ["SAPEFTEpisode", "EpisodeConfig"]
