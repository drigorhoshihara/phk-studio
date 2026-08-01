from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4


EventHandler = Callable[["KernelEvent"], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class KernelEvent:
    """Evento interno imutável do PHK Kernel."""

    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "phk-kernel"
    event_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )


class EventBus:
    """Barramento assíncrono de eventos internos."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(
        self,
        event_name: str,
        handler: EventHandler,
    ) -> None:
        if handler not in self._handlers[event_name]:
            self._handlers[event_name].append(handler)

    def unsubscribe(
        self,
        event_name: str,
        handler: EventHandler,
    ) -> None:
        handlers = self._handlers.get(event_name, [])

        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: KernelEvent) -> None:
        handlers = list(self._handlers.get(event.name, []))

        for handler in handlers:
            await handler(event)

    def subscriber_count(self, event_name: str) -> int:
        return len(self._handlers.get(event_name, []))