"""Парсер SRT, який зберігає тайминги побайтово і не втрачає розмітку.

Свідомо не використовуємо готову бібліотеку: нам потрібні три гарантії, яких
жодна з них не дає одночасно —

1. рядок таймкоду повертається у вихідний файл **точно тим самим рядком** (разом
   із можливими координатами `X1:.. X2:..`), бо зміщені тайминги — це брак;
2. розмітка (`<i>`, `{\\an8}`, `♪`) відокремлена від тексту, щоб перекладати
   чистий текст, а потім повернути розмітку на місце;
3. кожна репліка класифікована (діалог / напис / SDH / пісня), бо політика
   перекладу для них різна.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

__all__ = ["Cue", "CueKind", "TextLine", "parse_srt", "parse_timestamp", "SrtParseError"]


class SrtParseError(ValueError):
    """Файл не схожий на SRT або блок безнадійно поламаний."""


class CueKind(str, Enum):
    DIALOGUE = "dialogue"
    FORCED = "forced"  # напис/екранний текст — маркується позиціюванням {\anN}
    SDH = "sdh"  # [DOOR CREAKS], (WHISPERING)
    LYRICS = "lyrics"  # ♪ ... ♪


# Теги розмітки
_ASS_TAG = re.compile(r"\{\\[^}]*\}")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_FULL_WRAP = re.compile(r"^((?:<[a-zA-Z][^>]*>)+)(.*?)((?:</[a-zA-Z][^>]*>)+)$", re.DOTALL)
_MUSIC = "♪"

# "JOHN:", "- MAN (V.O.):", "NARRATOR :"
_SPEAKER_LABEL = re.compile(
    r"""^\s*
    (?P<name>[A-Z][A-Z0-9'’.\- ]{1,24}?)      # ім'я капсом
    (?:\s*\((?P<paren>[^)]{1,20})\))?          # (V.O.), (CONT'D)
    \s*:\s+                                     # двокрапка + пробіл
    """,
    re.VERBOSE,
)
_DIALOGUE_DASH = re.compile(r"^\s*[-–—]\s*")
_SDH_ONLY = re.compile(r"^\s*[\[(](?P<inner>[^\])]*)[\])]\s*$")
_TIMING_LINE = re.compile(
    r"^(?P<start>\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(?P<end>\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})"
)
_TIMESTAMP = re.compile(r"^(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})$")


def parse_timestamp(value: str) -> int:
    """`00:01:02,500` → мілісекунди."""
    m = _TIMESTAMP.match(value.strip())
    if not m:
        raise SrtParseError(f"некоректний таймкод: {value!r}")
    hours, minutes, seconds, millis = m.groups()
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(millis.ljust(3, "0"))
    )


@dataclass
class TextLine:
    """Один рядок репліки: чистий текст окремо від розмітки."""

    text: str
    prefix: str = ""  # {\an8}, ♪ — усе, що стоїть перед текстом
    suffix: str = ""  # ♪ у кінці
    wrap_tags: list[str] = field(default_factory=list)  # ['<i>'] — обгортка всього рядка
    partial_markup: bool = False  # розмітка покривала частину рядка → обгортаємо весь
    dialogue_dash: bool = False  # рядок починався з діалогового тире

    def render(self, text: str | None = None, dash: str = "—") -> str:
        """Збирає рядок назад, підставляючи перекладений текст."""
        body = self.text if text is None else text
        for tag in reversed(self.wrap_tags):
            closing = "</" + re.match(r"<([a-zA-Z][a-zA-Z0-9]*)", tag).group(1) + ">"
            body = f"{tag}{body}{closing}"
        if self.dialogue_dash:
            body = f"{dash} {body}"
        return f"{self.prefix}{body}{self.suffix}"


@dataclass
class Cue:
    """Одна репліка субтитрів."""

    index: int
    start_ms: int
    end_ms: int
    timing_line: str  # вихідний рядок таймкоду — повертається у файл без змін
    lines: list[TextLine]
    kind: CueKind = CueKind.DIALOGUE
    speaker_hint: str | None = None  # з мітки "JOHN:"
    speaker_note: str | None = None  # з дужок мітки: V.O., CONT'D
    all_caps: bool = False
    source_block: str = ""  # оригінальний блок цілком (для діагностики)

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def text(self) -> str:
        """Чистий текст репліки з переносами рядків."""
        return "\n".join(line.text for line in self.lines)

    @property
    def flat_text(self) -> str:
        """Чистий текст в один рядок — для підрахунку символів."""
        return " ".join(line.text for line in self.lines).strip()

    @property
    def char_count(self) -> int:
        return len(self.flat_text)

    @property
    def cps(self) -> float:
        seconds = self.duration_ms / 1000
        return self.char_count / seconds if seconds > 0 else float("inf")

    def char_budget(self, max_cps: float, max_lines: int, max_chars_per_line: int) -> int:
        """Скільки символів можна вмістити в цю репліку за нормами."""
        by_time = int((self.duration_ms / 1000) * max_cps)
        by_space = max_lines * max_chars_per_line
        return max(1, min(by_time, by_space))

    @property
    def is_translatable(self) -> bool:
        return bool(self.flat_text)


def _split_markup(raw: str) -> TextLine:
    """Відокремлює розмітку від тексту одного рядка."""
    line = raw

    dash = False
    if _DIALOGUE_DASH.match(line):
        line = _DIALOGUE_DASH.sub("", line, count=1)
        dash = True

    prefix_parts: list[str] = []
    for tag in _ASS_TAG.findall(line):
        prefix_parts.append(tag)
    line = _ASS_TAG.sub("", line)

    # Нотні знаки — маркери пісні, стоять по краях
    suffix = ""
    stripped = line.strip()
    if stripped.startswith(_MUSIC):
        prefix_parts.append(_MUSIC + " ")
        line = stripped[len(_MUSIC) :].lstrip()
    stripped = line.strip()
    if stripped.endswith(_MUSIC):
        suffix = " " + _MUSIC
        line = stripped[: -len(_MUSIC)].rstrip()

    # Діалогове тире могло стояти після {\anN}
    if not dash and _DIALOGUE_DASH.match(line):
        line = _DIALOGUE_DASH.sub("", line, count=1)
        dash = True

    wrap_tags: list[str] = []
    partial = False
    match = _FULL_WRAP.match(line.strip())
    if match and not _HTML_TAG.search(match.group(2)):
        opening = re.findall(r"<[a-zA-Z][^>]*>", match.group(1))
        closing = re.findall(r"</[a-zA-Z][^>]*>", match.group(3))
        if len(opening) == len(closing):
            wrap_tags = opening
            line = match.group(2)
    if _HTML_TAG.search(line):
        # Розмітка покриває лише частину рядка — після перекладу її не можна
        # позиціювати точно, тому обгортаємо весь рядок і позначаємо факт.
        partial = True
        for tag in re.findall(r"<[a-zA-Z][^>]*>", line):
            if tag not in wrap_tags:
                wrap_tags.append(tag)
        line = _HTML_TAG.sub("", line)

    return TextLine(
        text=line.strip(),
        prefix="".join(prefix_parts),
        suffix=suffix,
        wrap_tags=wrap_tags,
        partial_markup=partial,
        dialogue_dash=dash,
    )


def _extract_speaker(lines: list[TextLine]) -> tuple[str | None, str | None]:
    """Знімає мітку мовця з першого рядка, повертає (ім'я, примітка)."""
    if not lines:
        return None, None
    match = _SPEAKER_LABEL.match(lines[0].text)
    if not match:
        return None, None
    name = match.group("name").strip().rstrip(".")
    # Мітка мусить бути справді капсом, інакше це звичайне речення з двокрапкою
    if not name or name != name.upper() or not any(c.isalpha() for c in name):
        return None, None
    lines[0].text = lines[0].text[match.end() :].lstrip()
    return name, match.group("paren")


def _classify(lines: list[TextLine]) -> CueKind:
    if any(_MUSIC in line.prefix or _MUSIC in line.suffix for line in lines):
        return CueKind.LYRICS
    if any(r"\an" in line.prefix for line in lines):
        return CueKind.FORCED
    texts = [line.text for line in lines if line.text]
    if texts and all(_SDH_ONLY.match(t) for t in texts):
        return CueKind.SDH
    return CueKind.DIALOGUE


def _parse_block(block: str, fallback_index: int) -> Cue | None:
    """Розбирає один блок SRT. Повертає None для порожнього блоку."""
    raw_lines = block.split("\n")
    # прибираємо порожні рядки на краях, всередині зберігаємо
    while raw_lines and not raw_lines[0].strip():
        raw_lines.pop(0)
    while raw_lines and not raw_lines[-1].strip():
        raw_lines.pop()
    if not raw_lines:
        return None

    cursor = 0
    index = fallback_index
    if raw_lines[0].strip().isdigit() and len(raw_lines) > 1:
        index = int(raw_lines[0].strip())
        cursor = 1

    if cursor >= len(raw_lines):
        return None
    timing_line = raw_lines[cursor].strip()
    timing = _TIMING_LINE.match(timing_line)
    if not timing:
        # Деякі файли не мають рядка з індексом — спробуємо перший рядок як таймкод
        if cursor == 1:
            timing_line = raw_lines[0].strip()
            timing = _TIMING_LINE.match(timing_line)
            cursor = 0
            index = fallback_index
        if not timing:
            raise SrtParseError(f"блок #{fallback_index} без рядка таймкоду: {block!r}")
    cursor += 1

    lines = [_split_markup(line) for line in raw_lines[cursor:] if line.strip()]
    speaker, note = _extract_speaker(lines)
    lines = [line for line in lines if line.text or line.prefix or line.suffix]

    return Cue(
        index=index,
        start_ms=parse_timestamp(timing.group("start")),
        end_ms=parse_timestamp(timing.group("end")),
        timing_line=timing_line,
        lines=lines,
        kind=_classify(lines),
        speaker_hint=speaker,
        speaker_note=note,
        all_caps=bool(
            (joined := " ".join(l.text for l in lines).strip())
            and any(c.isalpha() for c in joined)
            and joined == joined.upper()
        ),
        source_block=block,
    )


def parse_srt(content: str) -> list[Cue]:
    """Розбирає вміст .srt на репліки.

    Стійкий до: BOM, CRLF/CR, непослідовних індексів, відсутніх індексів,
    зайвих порожніх рядків, координат після таймкоду.
    """
    text = content.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n{2,}", text)

    cues: list[Cue] = []
    for position, block in enumerate(blocks, start=1):
        if not block.strip():
            continue
        cue = _parse_block(block, fallback_index=len(cues) + 1)
        if cue is not None:
            cues.append(cue)

    if not cues:
        raise SrtParseError("не знайдено жодної репліки — це точно SRT?")
    return cues


def parse_file(path) -> list[Cue]:
    """Читає файл, пробуючи utf-8, потім cp1251 (типово для старих укр./рос. релізів)."""
    from pathlib import Path

    data = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return parse_srt(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    raise SrtParseError(f"не вдалося визначити кодування {path}")
