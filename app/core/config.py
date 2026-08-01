from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configurações centrais do PHK Studio."""

    app_name: str = Field(default="PHK Studio")
    app_version: str = Field(default="0.3.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=True)

    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)

    database_url: str = Field(
        default="sqlite:///./data/phk_studio.db"
    )

    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    shield_enabled: bool = Field(default=True)
    quarantine_path: Path = Field(
        default=PROJECT_ROOT / "quarantine"
    )
    security_log_path: Path = Field(
        default=PROJECT_ROOT / "logs" / "security"
    )

    secret_key: str = Field(
        default="development-key-change-before-production"
    )
    allowed_hosts: str = Field(
        default="127.0.0.1,localhost"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PHK_",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def allowed_hosts_list(self) -> list[str]:
        return [
            host.strip()
            for host in self.allowed_hosts.split(",")
            if host.strip()
        ]

    def create_required_directories(self) -> None:
        directories = (
            PROJECT_ROOT / "data",
            PROJECT_ROOT / "logs",
            self.security_log_path,
            self.quarantine_path,
            PROJECT_ROOT / "backups",
        )

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.create_required_directories()
    return settings