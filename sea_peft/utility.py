"""Utility smoothing for SA-PEFT units."""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List


@dataclass
class UtilityState:
    ema: float = 0.0
    initialized: bool = False
    residual: float = 0.0


class UtilityPredictor:
    def __init__(self, units: Iterable[str],
                 ema_decay: float = 0.93,
                 history: int = 5,
                 lambda_iqr: float = 0.25) -> None:
        self.decay = ema_decay
        self.history_len = max(history, 1)
        self.history: Dict[str, Deque[float]] = {u: deque(maxlen=self.history_len) for u in units}
        self.state: Dict[str, UtilityState] = {u: UtilityState() for u in units}
        self.predicted_measured_pairs: List[tuple[float, float]] = []
        self.lambda_iqr = lambda_iqr

    def update(self, unit: str, density: float) -> float:
        window = self.history[unit]
        info = self.state[unit]
        if info.initialized:
            self.predicted_measured_pairs.append((info.ema, density))
            if len(self.predicted_measured_pairs) > 1000:
                self.predicted_measured_pairs = self.predicted_measured_pairs[-1000:]
        if not info.initialized:
            info.ema = density
            info.initialized = True
        else:
            info.ema = self.decay * info.ema + (1.0 - self.decay) * density

        window.append(info.ema)
        robust_value = self.robust_score(unit, self.lambda_iqr)
        info.robust = robust_value
        info.residual = abs(info.ema - robust_value)
        return robust_value

    def predict(self, unit: str) -> float:
        state = self.state[unit]
        return state.robust if state.initialized else state.ema

    def residuals(self) -> Dict[str, float]:
        return {unit: state.residual for unit, state in self.state.items()}

    def spearman(self) -> float:
        if len(self.predicted_measured_pairs) < 2:
            return 0.0

        predicted = [p for p, _ in self.predicted_measured_pairs]
        measured = [m for _, m in self.predicted_measured_pairs]

        def rank(values: List[float]) -> List[float]:
            order = sorted(range(len(values)), key=lambda idx: values[idx])
            ranks = [0.0] * len(values)
            for r, idx in enumerate(order, 1):
                ranks[idx] = float(r)
            return ranks

        predicted_ranks = rank(predicted)
        measured_ranks = rank(measured)
        n = len(predicted_ranks)
        diff_sq = sum((pr - mr) ** 2 for pr, mr in zip(predicted_ranks, measured_ranks))
        return 1.0 - (6.0 * diff_sq) / (n * (n**2 - 1)) if n > 1 else 0.0

    def robust_score(self, unit: str, lambda_iqr: float = 0.25) -> float:
        window = list(self.history[unit])
        if not window or self.history_len <= 1:
            return self.state[unit].ema
        if len(window) == 1:
            return window[0]

        median = statistics.median(window)
        if self.history_len > 1:
            try:
                quartiles = statistics.quantiles(window, n=4, method="inclusive")
                q1, q3 = quartiles[0], quartiles[2]
            except statistics.StatisticsError:
                sorted_window = sorted(window)
                mid = len(sorted_window) // 2
                lower = sorted_window[:mid]
                upper = sorted_window[mid:]
                q1 = statistics.median(lower) if lower else median
                q3 = statistics.median(upper) if upper else median

            iqr = max(0.0, q3 - q1)
            return median - lambda_iqr * iqr
        else:
            return median


__all__ = ["UtilityPredictor"]
