"""
Knowledge service statistics tracking.
  - only in memory at the moment but can be imroved later
  - curntly doesn't carry from run to run either, it's just to get live stats
"""
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class KnowledgeServiceStats:
    """Statistics tracker for knowledge service queue operations."""
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _start_time: float | None = field(default=None, init=False)

    # Totals
    _total_added: int = field(default=0, init=False)
    _total_processed: int = field(default=0, init=False)

    # Sliding window tracking (timestamp, count)
    _added_timestamps: deque = field(default_factory=deque, init=False)
    _processed_timestamps: deque = field(default_factory=deque, init=False)

    # Configurable window size in seconds
    rate_window_seconds: int = 5

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._start_time = time.time()
            self._total_added = 0
            self._total_processed = 0
            self._added_timestamps.clear()
            self._processed_timestamps.clear()

    def record_added(self, count: int = 1) -> None:
        """Record messages added to the queue."""
        with self._lock:
            now = time.time()
            self._total_added += count
            self._added_timestamps.append((now, count))
            self._cleanup_old_timestamps(self._added_timestamps, now)

    def record_processed(self, count: int = 1) -> None:
        """Record messages processed from the queue."""
        with self._lock:
            now = time.time()
            self._total_processed += count
            self._processed_timestamps.append((now, count))
            self._cleanup_old_timestamps(self._processed_timestamps, now)

    def _cleanup_old_timestamps(self, timestamps: deque, now: float) -> None:
        """Remove timestamps older than the window."""
        cutoff = now - self.rate_window_seconds
        while timestamps and timestamps[0][0] < cutoff:
            timestamps.popleft()

    def _calculate_rate_from_timestamps(self, timestamps: deque, now: float) -> float:
        """Calculate the rate per second from timestamps within the window.

        Note: This method assumes the caller holds the lock and does NOT modify the deque.
        """
        cutoff = now - self.rate_window_seconds

        # Sum only timestamps within the window (without modifying the deque)
        total_in_window = sum(count for ts, count in timestamps if ts >= cutoff)

        if total_in_window == 0:
            return 0.0

        return total_in_window / self.rate_window_seconds

    def get_stats(self) -> dict:
        """Get current statistics snapshot."""
        with self._lock:
            now = time.time()

            # Cleanup old timestamps first
            self._cleanup_old_timestamps(self._added_timestamps, now)
            self._cleanup_old_timestamps(self._processed_timestamps, now)

            # Calculate rates (these don't modify the deques)
            add_rate = self._calculate_rate_from_timestamps(self._added_timestamps, now)
            process_rate = self._calculate_rate_from_timestamps(self._processed_timestamps, now)

            # Calculate pending (added - processed)
            pending = max(0, self._total_added - self._total_processed)

            # Calculate ETA based on current rates
            eta_seconds: float | None = None
            queue_growing = add_rate > process_rate

            if pending > 0 and process_rate > 0:
                if add_rate == 0:
                    # No more items being added, just draining
                    eta_seconds = pending / process_rate
                elif process_rate > add_rate:
                    # We're processing faster than adding (catching up)
                    net_rate = process_rate - add_rate
                    eta_seconds = pending / net_rate
                else:
                    # Queue is growing - show ETA assuming adding stops now
                    # This gives user an idea of how long to drain current backlog
                    eta_seconds = pending / process_rate
            if pending == 0:
                eta_seconds = 0.0

            elapsed = now - self._start_time if self._start_time else 0

            return {
                "total_added": self._total_added,
                "total_processed": self._total_processed,
                "pending_in_queue": pending,
                "queue_growing": queue_growing,
                "add_rate_per_second": round(add_rate, 2),
                "process_rate_per_second": round(process_rate, 2),
                "rate_window_seconds": self.rate_window_seconds,
                "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
                "eta_formatted": self._format_eta(eta_seconds),
                "eta_note": "ETA assumes no new items added" if queue_growing else None,
                "elapsed_seconds": round(elapsed, 1),
            }

    def set_rate_window(self, seconds: Literal[5, 10]) -> None:
        """Set the rate calculation window (5 or 10 seconds)."""
        with self._lock:
            self.rate_window_seconds = seconds

    @staticmethod
    def _format_eta(eta_seconds: float | None) -> str | None:
        """Format ETA in human readable format."""
        if eta_seconds is None:
            return None
        if eta_seconds == 0:
            return "Complete"

        hours, remainder = divmod(int(eta_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"
