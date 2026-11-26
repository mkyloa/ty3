"""Завантаження config.toml із дефолтами."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "claude-cli"
    name: str = "claude-opus-5-5"
    effort: str = "high"
    effort_light: str = "medium"
    max_tokens: int = 32000
    search_budget_usd: float = 1.5


@dataclass(frozen=True)
class SubtitleConfig:
    max_chars_per_line: int = 42
    max_lines: int = 2
    max_cps: float = 17.0
    min_duration_ms: int = 833
    max_duration_ms: int = 7000
    dialogue_dash: str = "—"
    quote_open: str = "«"
    quote_close: str = "»"


@dataclass(frozen=True)
class SegmentConfig:
    scene_gap_ms: int = 2500
    max_cues_per_chunk: int = 50
    max_chars_per_chunk: int = 6000
    context_before: int = 5
    context_after: int = 5


@dataclass(frozen=True)
class MediaConfig:
    # 0 = обчислити з розміру файлу. ffmpeg читає контейнер цілком, тому час
    # залежить від розміру й швидкості носія, а не від тривалості фільму.
    extract_timeout_s: int = 0
    # Консервативна нижня оцінка швидкості носія (МБ/с) для розрахунку ліміту
    slow_media_mb_s: float = 6.0


@dataclass(frozen=True)
class RunConfig:
    jobs: int = 1
    batch: bool = False


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def _build(cls: type, data: dict[str, Any]):
    known = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: Path | None = None) -> Config:
    """Читає config.toml. Відсутній файл — не помилка, беруться дефолти."""
    data: dict[str, Any] = {}
    if path is None:
        candidate = Path.cwd() / "config.toml"
        path = candidate if candidate.exists() else None
    if path is not None and path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))

    return Config(
        model=_build(ModelConfig, _section(data, "model")),
        subtitle=_build(SubtitleConfig, _section(data, "subtitle")),
        segment=_build(SegmentConfig, _section(data, "segment")),
        media=_build(MediaConfig, _section(data, "media")),
        run=_build(RunConfig, _section(data, "run")),
    )
