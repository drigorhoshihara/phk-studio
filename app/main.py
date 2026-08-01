from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.kernel.bootstrap import (
    KernelContext,
    bootstrap_kernel,
    shutdown_kernel,
)
from app.routers import health, pages


BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    kernel = await bootstrap_kernel()
    application.state.kernel = kernel

    yield

    await shutdown_kernel(kernel)


def create_application() -> FastAPI:
    application = FastAPI(
        title="PHK Studio",
        description="Scientific Content Production Platform",
        version="0.3.0",
        lifespan=lifespan,
    )

    application.mount(
        "/static",
        StaticFiles(directory=BASE_DIR / "static"),
        name="static",
    )

    application.include_router(health.router)
    application.include_router(pages.router)

    return application


app = create_application()