from fastapi import APIRouter


router = APIRouter(
    prefix="/health",
    tags=["System"],
)


@router.get("")
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "application": "PHK Studio",
        "version": "0.2.0",
        "security_module": "PHK Shield initialized",
    }