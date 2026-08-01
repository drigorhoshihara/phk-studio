from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.kernel.events.event_bus import EventBus, KernelEvent
from app.kernel.health.health_monitor import HealthMonitor
from app.kernel.module_manager import KernelModule, ModuleManager


@dataclass(slots=True)
class KernelContext:
    settings: Settings
    event_bus: EventBus
    module_manager: ModuleManager
    health_monitor: HealthMonitor


async def bootstrap_kernel() -> KernelContext:
    settings = get_settings()
    event_bus = EventBus()
    module_manager = ModuleManager()

    modules = (
        KernelModule(
            name="phk-core",
            version=settings.app_version,
            description="Núcleo e configurações do PHK Studio.",
        ),
        KernelModule(
            name="phk-shield",
            version="0.1.0",
            description="Segurança defensiva e auditoria.",
        ),
        KernelModule(
            name="phk-observatory",
            version="0.1.0",
            description="Telemetria, métricas e diagnóstico.",
        ),
    )

    for module in modules:
        module_manager.register(module)
        module_manager.mark_running(module.name)

    health_monitor = HealthMonitor(module_manager)

    await event_bus.publish(
        KernelEvent(
            name="kernel.started",
            payload={
                "application": settings.app_name,
                "version": settings.app_version,
            },
        )
    )

    return KernelContext(
        settings=settings,
        event_bus=event_bus,
        module_manager=module_manager,
        health_monitor=health_monitor,
    )


async def shutdown_kernel(context: KernelContext) -> None:
    await context.event_bus.publish(
        KernelEvent(
            name="kernel.stopping",
            payload={
                "application": context.settings.app_name,
            },
        )
    )

    for module in context.module_manager.list_modules():
        context.module_manager.mark_stopped(module.name)