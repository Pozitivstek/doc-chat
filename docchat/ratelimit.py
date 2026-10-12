"""Скользящее окно: не больше N событий на ключ за период."""
import time
from collections import defaultdict, deque
from typing import Callable

CLEANUP_EVERY = 1000


class RateLimiter:
    def __init__(self, max_events: int, window: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        self._max = max_events
        self._window = window
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._calls = 0

    def allow(self, key: str) -> bool:
        now = self._clock()
        self._calls += 1
        if self._calls % CLEANUP_EVERY == 0:
            self._forget_idle(now)
        events = self._events[key]
        while events and now - events[0] >= self._window:
            events.popleft()
        if len(events) >= self._max:
            return False
        events.append(now)
        return True

    def _forget_idle(self, now: float) -> None:
        idle = [k for k, ev in self._events.items() if not ev or now - ev[-1] >= self._window]
        for key in idle:
            del self._events[key]
