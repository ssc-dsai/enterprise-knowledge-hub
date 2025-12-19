"""Thread-safe run state management for knowledge operations."""
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class RunState:
    """Thread-safe run state tracker."""
    lock: Lock = field(default_factory=Lock)
    running: bool = False

    def try_start(self) -> bool:
        """Attempt to start a run. Returns True if started, False if already running."""
        with self.lock:
            if self.running:
                return False
            self.running = True
            return True

    def stop(self) -> None:
        """Mark the run as stopped."""
        with self.lock:
            self.running = False

    def is_running(self) -> bool:
        """Check if a run is in progress."""
        with self.lock:
            return self.running
