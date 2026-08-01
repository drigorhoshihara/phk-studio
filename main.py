from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="PHK Studio",
    version="0.1.0",
    description="Scientific Content Production Platform",
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": "PHK Studio",
            "version": "0.1.0",
            "tagline": "Da evidência ao conhecimento.",
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "application": "PHK Studio",
        "version": "0.1.0",
    }