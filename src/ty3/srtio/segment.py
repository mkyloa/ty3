"""Сегментація на сцени й чанки.

Чому сцени, а не просто «кожні 50 реплик»: контекст перекладу скидається на межі
сцени природним чином, тому розрив чанка по межі сцени коштує якості майже
нічого, а розрив посеред діалогу — коштує дорого (модель не бачить, чим репліка
закінчилася і кому відповідає).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import SegmentConfig
from .parse import Cue

__all__ = ["Scene", "Chunk", "build_scenes", "build_chunks", "build_windows"]


@dataclass
class Scene:
    number: int
    cues: list[Cue]

    @property
    def start_ms(self) -> int:
        return self.cues[0].start_ms

    @property
    def end_ms(self) -> int:
        return self.cues[-1].end_ms

    @property
    def char_count(self) -> int:
        return sum(cue.char_count for cue in self.cues)


@dataclass
class Chunk:
    number: int
    cues: list[Cue]
    context_before: list[Cue] = field(default_factory=list)
    context_after: list[Cue] = field(default_factory=list)
    scene_numbers: list[int] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(cue.char_count for cue in self.cues)

    @property
    def indices(self) -> list[int]:
        return [cue.index for cue in self.cues]


def build_scenes(cues: list[Cue], gap_ms: int) -> list[Scene]:
    """Ріже на сцени за паузою між репліками."""
    if not cues:
        return []
    scenes: list[Scene] = []
    current: list[Cue] = [cues[0]]
    for previous, cue in zip(cues, cues[1:]):
        if cue.start_ms - previous.end_ms >= gap_ms:
            scenes.append(Scene(number=len(scenes) + 1, cues=current))
            current = [cue]
        else:
            current.append(cue)
    scenes.append(Scene(number=len(scenes) + 1, cues=current))
    return scenes


def _split_oversized(scene: Scene, cfg: SegmentConfig) -> list[list[Cue]]:
    """Сцена сама більша за чанк — ріжемо її по найбільших внутрішніх паузах."""
    cues = scene.cues
    if len(cues) <= cfg.max_cues_per_chunk and scene.char_count <= cfg.max_chars_per_chunk:
        return [cues]

    parts: list[list[Cue]] = []
    current: list[Cue] = []
    current_chars = 0
    for cue in cues:
        would_overflow = current and (
            len(current) + 1 > cfg.max_cues_per_chunk
            or current_chars + cue.char_count > cfg.max_chars_per_chunk
        )
        if would_overflow:
            parts.append(current)
            current, current_chars = [], 0
        current.append(cue)
        current_chars += cue.char_count
    if current:
        parts.append(current)
    return parts


def build_chunks(cues: list[Cue], cfg: SegmentConfig) -> list[Chunk]:
    """Групує сцени в чанки, не розриваючи сцену без потреби.

    Кожен чанк отримує read-only контекст: репліки до і після нього. Контекст
    потрібен моделі, щоб зрозуміти, на що відповідає перша репліка чанка й куди
    веде остання, але перекладати його не треба.
    """
    scenes = build_scenes(cues, cfg.scene_gap_ms)
    if not scenes:
        return []

    groups: list[tuple[list[Cue], list[int]]] = []
    current: list[Cue] = []
    current_scenes: list[int] = []
    current_chars = 0

    for scene in scenes:
        for part in _split_oversized(scene, cfg):
            part_chars = sum(cue.char_count for cue in part)
            fits = current and (
                len(current) + len(part) <= cfg.max_cues_per_chunk
                and current_chars + part_chars <= cfg.max_chars_per_chunk
            )
            if fits:
                current.extend(part)
                current_chars += part_chars
                if scene.number not in current_scenes:
                    current_scenes.append(scene.number)
            else:
                if current:
                    groups.append((current, current_scenes))
                current, current_scenes, current_chars = list(part), [scene.number], part_chars
    if current:
        groups.append((current, current_scenes))

    by_index = {cue.index: position for position, cue in enumerate(cues)}
    chunks: list[Chunk] = []
    for number, (group, scene_numbers) in enumerate(groups, start=1):
        first = by_index[group[0].index]
        last = by_index[group[-1].index]
        chunks.append(
            Chunk(
                number=number,
                cues=group,
                context_before=cues[max(0, first - cfg.context_before) : first],
                context_after=cues[last + 1 : last + 1 + cfg.context_after],
                scene_numbers=scene_numbers,
            )
        )
    return chunks


def build_windows(cues: list[Cue], size: int, overlap: int) -> list[tuple[list[Cue], list[Cue]]]:
    """Вікна з перекриттям для проходу атрибуції мовців.

    Повертає пари (контекст, вікно): контекст — попередні репліки, які модель
    читає, але не описує; вікно — репліки, для яких треба дати відповідь.
    """
    if size <= 0:
        raise ValueError("size мусить бути додатним")
    windows: list[tuple[list[Cue], list[Cue]]] = []
    for start in range(0, len(cues), size):
        window = cues[start : start + size]
        if not window:
            break
        windows.append((cues[max(0, start - overlap) : start], window))
    return windows
