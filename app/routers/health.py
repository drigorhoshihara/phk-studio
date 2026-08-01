from typing import Any

from fastapi import APIRouter, Request


router = APIRouter(
    prefix="/health",
    tags=["System"],
)


@router.get("")
async def health_check(request: Request) -> dict[str, Any]:
    kernel = request.app.state.kernel
    snapshot = kernel.health_monitor.snapshot()

    return {
        "application": kernel.settings.app_name,
        "version": kernel.settings.app_version,
        **snapshot,
    }