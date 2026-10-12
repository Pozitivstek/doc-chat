"""Запуск: python -m docchat  (HOST / PORT — из окружения)."""
import logging
import os

import uvicorn

from .app import create_app
from .config import load_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    mode = f"Claude ({settings.model})" if settings.has_llm else "поиск без LLM (ANTHROPIC_API_KEY не задан)"
    logging.getLogger("docchat").info("режим ответов: %s", mode)
    uvicorn.run(create_app(settings), host=os.getenv("HOST", "127.0.0.1"),
                port=int(os.getenv("PORT", "8000")), log_level="info")


main()
