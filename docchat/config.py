"""Настройки приложения: из окружения / .env, в тестах — напрямую."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = ""
    model: str = "claude-opus-5-5"
    effort: str = "medium"            # low | medium | high | xhigh | max
    fallbacks: bool = True            # серверный фолбэк при отказе классификатора
    db_path: str | Path = ROOT / "data" / "docchat.db"
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pages: int = 300
    top_k: int = 6                    # сколько фрагментов отдаём модели
    chat_rate_limit: int = 20         # вопросов в минуту на одну сессию
    ttl_hours: float = 24.0           # сколько живут загруженные документы
    cookie_secure: bool = False       # включить за HTTPS

    @property
    def has_llm(self) -> bool:
        return bool(self.anthropic_api_key)


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "off", "no", "")


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    return Settings(
        anthropic_api_key=(os.getenv("ANTHROPIC_API_KEY") or "").strip(),
        model=os.getenv("CLAUDE_MODEL", "claude-opus-5-5").strip(),
        effort=os.getenv("CLAUDE_EFFORT", "medium").strip(),
        fallbacks=_flag("CLAUDE_FALLBACKS", True),
        db_path=Path(os.getenv("DB_PATH") or ROOT / "data" / "docchat.db"),
        max_upload_bytes=int(float(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024),
        max_pages=int(os.getenv("MAX_PAGES", "300")),
        ttl_hours=float(os.getenv("TTL_HOURS", "24")),
        cookie_secure=_flag("COOKIE_SECURE", False),
    )
