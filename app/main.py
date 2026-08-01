from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import health, pages


BASE_DIR = Path(__file__).resolve().parent


def create_application() -> FastAPI:
    application = FastAPI(
        title="PHK Studio",
        description="Scientific Content Production Platform",
        version="0.2.0",
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