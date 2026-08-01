from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModuleStatus(StrEnum):
    REGISTERED = "registered"
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(slots=True)
class KernelModule:
    name: str
    version: str
    description: str
    status: ModuleStatus = ModuleStatus.REGISTERED
    error: str | None = None


class ModuleManager:
    """Registro e acompanhamento dos módulos do PHK Studio."""

    def __init__(self) -> None:
        self._modules: dict[str, KernelModule] = {}

    def register(self, module: KernelModule) -> None:
        if module.name in self._modules:
            raise ValueError(
                f"O módulo '{module.name}' já está registrado."
            )

        self._modules[module.name] = module

    def mark_running(self, module_name: str) -> None:
        module = self.get(module_name)
        module.status = ModuleStatus.RUNNING
        module.error = None

    def mark_failed(
        self,
        module_name: str,
        error: Exception | str,
    ) -> None:
        module = self.get(module_name)
        module.status = ModuleStatus.FAILED
        module.error = str(error)

    def mark_stopped(self, module_name: str) -> None:
        module = self.get(module_name)
        module.status = ModuleStatus.STOPPED

    def get(self, module_name: str) -> KernelModule:
        try:
            return self._modules[module_name]
        except KeyError as exc:
            raise KeyError(
                f"Módulo não registrado: {module_name}"
            ) from exc

    def list_modules(self) -> list[KernelModule]:
        return list(self._modules.values())

    def summary(self) -> dict[str, int]:
        result = {status.value: 0 for status in ModuleStatus}

        for module in self._modules.values():
            result[module.status.value] += 1

        return result