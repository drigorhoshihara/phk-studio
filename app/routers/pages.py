from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parents[1]

templates = Jinja2Templates(
    directory=BASE_DIR / "templates"
)

router = APIRouter(
    tags=["Pages"],
)


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": "PHK Studio",
            "version": "0.2.0",
            "tagline": "Da evidência ao conhecimento.",
        },
    )