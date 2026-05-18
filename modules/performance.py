"""
Performance monitoring for the RAG system.

Tracks retrieval latencies and LLM Time-To-First-Token (TTFT) across queries,
providing percentile-based statistics for observability.

Integration hints:
    - In vector_db.py search() and hybrid_search(): wrap with @track_latency("retrieval")
    - In llm_handler.py _invoke_llm(): wrap with @track_latency("llm")
"""

import time
from typing import Any, Callable, Dict, List
from functools import wraps

from modules.logger import get_logger


logger = get_logger(__name__)


class PerformanceMonitor:
    """Collects and reports performance metrics for RAG pipeline operations."""

    def __init__(self) -> None:
        """Initialize internal storage for latency and TTFT metrics."""
        self._timers: Dict[str, float] = {}
        self._search_latencies: List[float] = []
        self._llm_ttft_values: List[float] = []

    def start_timer(self, name: str) -> None:
        """Start timing a named operation.

        Args:
            name: Identifier for the timed operation.
        """
        self._timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        """Stop timing a named operation and record the elapsed duration.

        Args:
            name: Identifier matching a previously started timer.

        Returns:
            Elapsed time in seconds (float).  Returns 0.0 if the timer
            was never started.
        """
        start = self._timers.pop(name, None)
        if start is None:
            return 0.0
        elapsed = time.perf_counter() - start
        return elapsed

    def record_search_latency(self, latency_seconds: float) -> None:
        """Record a retrieval latency data point.

        Args:
            latency_seconds: Latency in seconds for a retrieval operation.
        """
        self._search_latencies.append(latency_seconds)

    def record_llm_ttft(self, ttft_seconds: float) -> None:
        """Record a Time-To-First-Token measurement for the LLM.

        Args:
            ttft_seconds: TTFT in seconds for an LLM invocation.
        """
        self._llm_ttft_values.append(ttft_seconds)

    @staticmethod
    def _percentile(data: list, p: float) -> float:
        """Compute the p-th percentile of a numeric list.

        Args:
            data: List of numeric values (must be sortable).
            p: Percentile value (0–100).

        Returns:
            The p-th percentile value, or 0.0 if *data* is empty.
        """
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    def get_stats(self) -> Dict[str, Any]:
        """Compute percentile statistics for all recorded metrics.

        Returns:
            A dictionary containing:
                - retrieval_p50 / retrieval_p95 / retrieval_p99
                - llm_ttft_p50 / llm_ttft_p95 / llm_ttft_p99
                - total_queries
        """
        stats: Dict[str, Any] = {
            "retrieval_p50": self._percentile(self._search_latencies, 50),
            "retrieval_p95": self._percentile(self._search_latencies, 95),
            "retrieval_p99": self._percentile(self._search_latencies, 99),
            "llm_ttft_p50": self._percentile(self._llm_ttft_values, 50),
            "llm_ttft_p95": self._percentile(self._llm_ttft_values, 95),
            "llm_ttft_p99": self._percentile(self._llm_ttft_values, 99),
            "total_queries": len(self._search_latencies),
        }
        return stats

    def log_summary(self) -> None:
        """Log a formatted summary of all performance statistics."""
        stats = self.get_stats()
        summary_lines = [
            "=" * 60,
            "Performance Monitor Summary",
            "=" * 60,
            f"Total queries:            {stats['total_queries']}",
            "-" * 60,
            "Retrieval latency:",
            f"  P50: {stats['retrieval_p50']:.4f}s",
            f"  P95: {stats['retrieval_p95']:.4f}s",
            f"  P99: {stats['retrieval_p99']:.4f}s",
            "-" * 60,
            "LLM TTFT:",
            f"  P50: {stats['llm_ttft_p50']:.4f}s",
            f"  P95: {stats['llm_ttft_p95']:.4f}s",
            f"  P99: {stats['llm_ttft_p99']:.4f}s",
            "=" * 60,
        ]
        for line in summary_lines:
            logger.info(line)

    def reset(self) -> None:
        """Clear all recorded metrics and active timers."""
        self._timers.clear()
        self._search_latencies.clear()
        self._llm_ttft_values.clear()


_monitor = PerformanceMonitor()


def get_monitor() -> PerformanceMonitor:
    """Return the module-level singleton PerformanceMonitor instance."""
    return _monitor


def track_latency(name: str) -> Callable:
    """Decorator factory that records the latency of a function call.

    The elapsed wall-clock time of every invocation is recorded via
    ``get_monitor().record_search_latency(elapsed)``.

    Usage::

        @track_latency("retrieval")
        def search(...):
            ...

        @track_latency("llm")
        def _invoke_llm(...):
            ...

    Args:
        name: Label used when recording the latency data point.

    Returns:
        A decorator that wraps the target function with timing logic.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            monitor = get_monitor()
            monitor.start_timer(name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = monitor.stop_timer(name)
                if elapsed > 0:
                    monitor.record_search_latency(elapsed)
        return wrapper
    return decorator
