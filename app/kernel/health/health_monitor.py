from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from app.kernel.module_manager import ModuleManager


class HealthMonitor:
    """Consolida o estado operacional do PHK Studio."""

    def __init__(self, module_manager: ModuleManager) -> None:
        self._module_manager = module_manager
        self._started_at = datetime.now(UTC)

    def snapshot(self) -> dict[str, Any]:
        now = datetime.now(UTC)

        modules = [
            asdict(module)
            for module in self._module_manager.list_modules()
        ]

        failed_modules = [
            module
            for module in modules
            if module["status"] == "failed"
        ]

        return {
            "status": "degraded" if failed_modules else "healthy",
            "started_at": self._started_at.isoformat(),
            "checked_at": now.isoformat(),
            "uptime_seconds": int(
                (now - self._started_at).total_seconds()
            ),
            "module_summary": self._module_manager.summary(),
            "modules": modules,
        }