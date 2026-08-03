"""Доступ до промптових файлів.

Промпти живуть у .md-файлах, а не в рядках Python, з двох причин: їх редагує
людина без страху зламати код, і вони версіонуються окремо — саме в них осідають
висновки з рецензій готових фільмів.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

__all__ = ["load", "load_many", "PROMPTS_DIR"]

PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def load(name: str) -> str:
    """Читає промпт за іменем без розширення: load("uk_rules")."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"немає промпта {name!r} у {PROMPTS_DIR}")
    return path.read_text(encoding="utf-8").strip()


def load_many(*names: str, separator: str = "\n\n---\n\n") -> str:
    return separator.join(load(name) for name in names)
