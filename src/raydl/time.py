from contextlib import contextmanager
from time import perf_counter
from typing import Any, Callable, Optional

__all__ = ["Timer", "running_timer"]


class Timer:
    def __init__(self, average: bool = False):
        self._average = average

        self.reset()

    def reset(self, *args: Any) -> "Timer":
        """Reset the timer to zero."""
        self._t0 = perf_counter()
        self.total = 0.0
        self.step_count = 0.0
        self.running = True

        self._last_check_time = 0.0
        self._last_step_time = 0.0

        self._last_step_value = 0.0
        self._last_step_elapsed = 0.0

        return self

    def pause(self, *args: Any) -> None:
        """Pause the current running timer."""
        if self.running:
            self.total += self._elapsed()
            self.running = False

    def resume(self, *args: Any) -> None:
        """Resume the current running timer."""
        if not self.running:
            self.running = True
            self._t0 = perf_counter()

    def since_start(self, do_not_update_check_time=False) -> float:
        total = self.total
        if self.running:
            total += self._elapsed()
        if not do_not_update_check_time:
            self._last_check_time = total
        return total

    def since_last_check(self) -> float:
        last_check_time = self._last_check_time
        return self.since_start() - last_check_time

    def value(self) -> float:
        total = self.since_start()
        denominator = max(self.step_count, 1.0) if self._average else 1.0
        return total / denominator

    def step(self, *args: Any) -> None:
        """Increment the timer."""
        self.step_count += 1.0

        total = self.since_start(do_not_update_check_time=True)
        self._last_step_elapsed = total - self._last_step_value
        self._last_step_value = total

    def _elapsed(self) -> float:
        return perf_counter() - self._t0


@contextmanager
def running_timer(echo_func: Optional[Callable] = lambda x: print(f"{x:.4f}"), average=False):
    timer = Timer(average=average)
    try:
        yield timer
    finally:
        if echo_func is not None:
            echo_func(timer.value())
