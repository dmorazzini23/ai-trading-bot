from __future__ import annotations

# AI-AGENT-REF: test facade for legacy process manager imports
class ProcessManager:
    """Minimal process manager for tests."""

    def acquire_process_lock(self, path: str) -> bool:  # pragma: no cover
        return True

    def find_python_processes(self) -> list[dict]:  # pragma: no cover
        return []

    def _is_trading_process(self, proc: dict) -> bool:  # pragma: no cover
        return False

    def check_multiple_instances(self) -> dict:  # pragma: no cover
        return {"total_instances": 1, "multiple_instances": False, "recommendations": []}
