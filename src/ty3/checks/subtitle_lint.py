"""Технічні норми субтитрування: швидкість читання, довжина, розмітка, тайминги.

Ці перевірки — не «покращення», а межа профпридатності файлу. Субтитр, який
неможливо прочитати за час показу, не виконує своєї функції, навіть якщо переклад
блискучий.
"""

from __future__ import annotations

import re

from ..config import SubtitleConfig
from ..srtio.parse import Cue, CueKind
from .base import Issue, Severity

__all__ = [
    "check_cue",
    "check_all",
    "check_timings_preserved",
    "count_tags",
]

_TAG = re.compile(r"</?[a-zA-Z][^>]*>|\{\\[^}]*\}")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’.-]{2,}")


def count_tags(text: str) -> list[str]:
    """Список тегів у порядку появи — для порівняння джерела й перекладу."""
    return _TAG.findall(text)


def check_cue(
    cue: Cue,
    uk_lines: list[str],
    cfg: SubtitleConfig,
    *,
    allowed_latin: set[str] | None = None,
) -> list[Issue]:
    """Перевіряє одну перекладену репліку. `uk_lines` — уже готові рядки з розміткою."""
    issues: list[Issue] = []
    plain_lines = [_TAG.sub("", line).strip() for line in uk_lines]
    plain_lines = [line for line in plain_lines if line]
    flat = " ".join(plain_lines)

    if not flat:
        if cue.is_translatable:
            issues.append(
                Issue(cue.index, "EMPTY_TRANSLATION", Severity.ERROR, "переклад порожній")
            )
        return issues

    # --- швидкість читання ---
    seconds = cue.duration_ms / 1000
    if seconds > 0:
        cps = len(flat) / seconds
        if cps > cfg.max_cps:
            issues.append(
                Issue(
                    cue.index,
                    "CPS_HIGH",
                    Severity.ERROR,
                    f"{cps:.1f} симв./с перевищує норму {cfg.max_cps:.0f}",
                    f"{len(flat)} символів за {seconds:.2f} с",
                )
            )

    # --- геометрія ---
    for number, line in enumerate(plain_lines, start=1):
        if len(line) > cfg.max_chars_per_line:
            issues.append(
                Issue(
                    cue.index,
                    "LINE_TOO_LONG",
                    Severity.ERROR,
                    f"рядок {number}: {len(line)} симв. > {cfg.max_chars_per_line}",
                    line,
                )
            )
    if len(plain_lines) > cfg.max_lines:
        issues.append(
            Issue(
                cue.index,
                "TOO_MANY_LINES",
                Severity.ERROR,
                f"{len(plain_lines)} рядки > {cfg.max_lines}",
            )
        )

    # --- тривалість: це властивість джерела, виправити не можемо, але знати треба ---
    if cue.duration_ms < cfg.min_duration_ms:
        issues.append(
            Issue(
                cue.index,
                "DURATION_SHORT",
                Severity.WARNING,
                f"{cue.duration_ms} мс < {cfg.min_duration_ms} мс — тайминг джерела",
            )
        )
    if cue.duration_ms > cfg.max_duration_ms:
        issues.append(
            Issue(
                cue.index,
                "DURATION_LONG",
                Severity.INFO,
                f"{cue.duration_ms} мс > {cfg.max_duration_ms} мс — тайминг джерела",
            )
        )

    # --- цілісність розмітки ---
    source_tags = [tag for line in cue.lines for tag in count_tags(line.render())]
    output_tags = [tag for line in uk_lines for tag in count_tags(line)]
    if _normalize_tags(source_tags) != _normalize_tags(output_tags):
        issues.append(
            Issue(
                cue.index,
                "TAG_MISMATCH",
                Severity.WARNING,
                "набір розмітки не збігається з джерелом",
                f"джерело: {source_tags} → переклад: {output_tags}",
            )
        )

    # --- недоперекладене ---
    if cue.kind in (CueKind.DIALOGUE, CueKind.SDH):
        allowed = {word.casefold() for word in (allowed_latin or set())}
        leftovers = [
            word
            for word in (_trim(match) for match in _LATIN_WORD.findall(flat))
            if word and word.casefold() not in allowed
        ]
        if leftovers:
            issues.append(
                Issue(
                    cue.index,
                    "LATIN_LEFTOVER",
                    Severity.WARNING,
                    "латиниця в перекладі — можливо, не перекладено",
                    ", ".join(sorted(set(leftovers))),
                )
            )
    return issues


def _trim(word: str) -> str:
    """Знімає кінцеву пунктуацію: «FBI.» — це те саме слово, що «FBI»."""
    return word.strip(".'’-")


def _normalize_tags(tags: list[str]) -> list[str]:
    """Порівнюємо типи тегів, а не їхні атрибути: <font color=…> може змінитися."""
    normalized = []
    for tag in tags:
        match = re.match(r"</?([a-zA-Z][a-zA-Z0-9]*)", tag)
        if match:
            normalized.append(match.group(1).lower())
        else:
            normalized.append(tag)  # {\anN} порівнюємо як є
    return sorted(normalized)


def check_all(
    cues: list[Cue],
    rendered: dict[int, list[str]],
    cfg: SubtitleConfig,
    *,
    allowed_latin: set[str] | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    for cue in cues:
        lines = rendered.get(cue.index)
        if lines is None:
            if cue.is_translatable:
                issues.append(
                    Issue(cue.index, "NOT_TRANSLATED", Severity.ERROR, "репліку пропущено")
                )
            continue
        issues.extend(check_cue(cue, lines, cfg, allowed_latin=allowed_latin))
    return issues


def check_timings_preserved(source_srt: str, output_srt: str) -> list[Issue]:
    """Найважливіша перевірка файлу: тайминги мусять бути ті самі, рядок у рядок."""
    pattern = re.compile(r"^.*-->.*$", re.MULTILINE)
    source = pattern.findall(source_srt.replace("\r\n", "\n"))
    output = pattern.findall(output_srt.replace("\r\n", "\n"))

    if len(source) != len(output):
        return [
            Issue(
                0,
                "CUE_COUNT_CHANGED",
                Severity.ERROR,
                f"кількість реплик змінилася: {len(source)} → {len(output)}",
            )
        ]
    issues = []
    for position, (before, after) in enumerate(zip(source, output), start=1):
        if before.strip() != after.strip():
            issues.append(
                Issue(
                    position,
                    "TIMING_CHANGED",
                    Severity.ERROR,
                    "тайминг змінено",
                    f"{before.strip()!r} → {after.strip()!r}",
                )
            )
    return issues
