"""Algorithm/objective-aware campaign completion forecasting.

This module estimates *whole campaign* completion time, not tqdm's ETA for one
currently running job.  It learns runtime separately by benchmark/algorithm/
objective, blends completed-job history with live progress, and then simulates
how the remaining queue will occupy the configured process slots.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import math
import statistics
from typing import DefaultDict, Iterable, List, Mapping, Optional, Sequence, Tuple

JobKey = Tuple[str, str, str]  # (benchmark, algorithm, objective)


def _valid_seconds(values: Iterable[float]) -> List[float]:
    clean: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0.0:
            clean.append(number)
    return clean


def robust_median(values: Iterable[float]) -> Optional[float]:
    clean = _valid_seconds(values)
    if not clean:
        return None
    return float(statistics.median(clean))


def blend_runtime(
    historical_seconds: Optional[float],
    live_seconds: Optional[float],
    historical_weight: float = 0.70,
) -> Optional[float]:
    """Blend completed history and current-job projection into one estimate."""
    if historical_seconds is None:
        return live_seconds
    if live_seconds is None:
        return historical_seconds
    w = min(max(float(historical_weight), 0.0), 1.0)
    return w * historical_seconds + (1.0 - w) * live_seconds


class RuntimeModel:
    """Hierarchical runtime predictor.

    Fallback order is intentionally conservative and transparent:
      1. exact benchmark + algorithm + objective
      2. benchmark + algorithm
      3. algorithm across benchmarks
      4. global observed runtime
    """

    def __init__(self, historical_weight: float = 0.70):
        self.historical_weight = float(historical_weight)
        self._history_exact: DefaultDict[JobKey, List[float]] = defaultdict(list)

    def add_completed(self, key: JobKey, duration_seconds: float) -> None:
        clean = _valid_seconds([duration_seconds])
        if clean:
            self._history_exact[key].append(clean[0])

    def extend_completed(self, key: JobKey, durations: Iterable[float]) -> None:
        self._history_exact[key].extend(_valid_seconds(durations))

    def historical_count(self) -> int:
        return sum(len(values) for values in self._history_exact.values())

    def _history_levels(self, key: JobKey):
        benchmark, algorithm, _objective = key
        exact = list(self._history_exact.get(key, []))
        benchmark_algorithm = [
            d
            for (fun, algo, _obj), values in self._history_exact.items()
            if fun == benchmark and algo == algorithm
            for d in values
        ]
        algorithm_only = [
            d
            for (_fun, algo, _obj), values in self._history_exact.items()
            if algo == algorithm
            for d in values
        ]
        global_values = [d for values in self._history_exact.values() for d in values]
        return [exact, benchmark_algorithm, algorithm_only, global_values]

    @staticmethod
    def _live_levels(key: JobKey, live: Mapping[JobKey, Sequence[float]]):
        benchmark, algorithm, _objective = key
        exact = list(live.get(key, []))
        benchmark_algorithm = [
            d
            for (fun, algo, _obj), values in live.items()
            if fun == benchmark and algo == algorithm
            for d in values
        ]
        algorithm_only = [
            d
            for (_fun, algo, _obj), values in live.items()
            if algo == algorithm
            for d in values
        ]
        global_values = [d for values in live.values() for d in values]
        return [exact, benchmark_algorithm, algorithm_only, global_values]

    def predict_total_seconds(
        self,
        key: JobKey,
        live_total_projections: Mapping[JobKey, Sequence[float]],
    ) -> Optional[float]:
        history_levels = self._history_levels(key)
        live_levels = self._live_levels(key, live_total_projections)
        for history_values, live_values in zip(history_levels, live_levels):
            history = robust_median(history_values)
            live = robust_median(live_values)
            estimate = blend_runtime(history, live, self.historical_weight)
            if estimate is not None:
                return float(estimate)
        return None


def schedule_makespan_seconds(
    active_remaining_seconds: Sequence[float],
    pending_durations_seconds: Sequence[float],
    concurrency: int,
) -> Optional[float]:
    """Estimate when the final job finishes under greedy process-slot scheduling."""
    workers = max(int(concurrency), 1)
    active = _valid_seconds(active_remaining_seconds)
    pending = _valid_seconds(pending_durations_seconds)

    if not active and not pending:
        return 0.0

    # Running jobs already occupy worker slots. Remaining slots are available now.
    slots = active[:workers] + [0.0] * max(0, workers - len(active))
    if not slots:
        slots = [0.0] * workers

    # Match ProcessPoolExecutor's practical behaviour: next queued job goes to the
    # first worker that becomes available.
    for duration in pending:
        slot_index = min(range(len(slots)), key=slots.__getitem__)
        slots[slot_index] += duration

    return float(max(slots)) if slots else 0.0


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or seconds < 0:
        return "warming up"
    total_seconds = int(round(float(seconds)))
    if total_seconds < 60:
        return f"{total_seconds}s"
    total_minutes, sec = divmod(total_seconds, 60)
    days, remainder = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {sec:02d}s"


def finish_timestamp(remaining_seconds: Optional[float], now: Optional[datetime] = None) -> str:
    if remaining_seconds is None or not math.isfinite(float(remaining_seconds)):
        return "warming up"
    if now is None:
        now = datetime.now().astimezone()
    finish = now + timedelta(seconds=max(0.0, float(remaining_seconds)))
    zone = finish.tzname() or "local"
    return finish.strftime(f"%Y-%m-%d %H:%M:%S {zone}")
