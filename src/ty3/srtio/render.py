"""Збирання SRT назад: тайминги 1:1, розмітка на місці, українська типографіка.

Розрив рядків робить код, а не модель: це механічна робота з чіткими правилами
(баланс довжин, не відривати прийменник від іменника, різати по пунктуації),
і детермінований алгоритм тут надійніший за LLM.
"""

from __future__ import annotations

import re

from ..config import SubtitleConfig
from .parse import Cue

__all__ = [
    "render_srt",
    "render_bilingual",
    "layout_cue",
    "wrap_text",
    "normalize_typography",
    "format_timestamp",
]

# Службові слова, які не можна лишати в кінці рядка — вони «тягнуться» до наступного
_CLINGING = {
    # прийменники
    "в", "у", "з", "зі", "із", "на", "до", "від", "при", "під", "над", "за", "по",
    "про", "для", "без", "крім", "між", "через", "біля", "після", "перед", "о", "об",
    "поза", "серед", "проти", "щодо", "попри", "коло", "край", "потому",
    # сполучники та частки
    "і", "й", "та", "а", "але", "чи", "або", "ні", "не", "як", "що", "щоб", "бо",
    "аби", "хоч", "хоча", "мов", "наче", "ніби", "то", "це", "цей", "ця", "той",
    "моя", "мій", "твій", "твоя", "наш", "ваш", "його", "її", "їх", "усі", "всі",
}
# Слова, перед якими розрив природний
_BREAK_BEFORE = {
    "і", "й", "та", "а", "але", "бо", "що", "щоб", "як", "коли", "якщо", "хоч",
    "тому", "проте", "однак", "адже", "аби", "поки", "доки", "ніж",
}
_STRONG_PUNCT = ".!?…:;"
_WEAK_PUNCT = ","

_APOSTROPHE = re.compile(r"(?<=[а-яіїєґА-ЯІЇЄҐa-zA-Z])'(?=[а-яіїєґА-ЯІЇЄҐa-zA-Z])")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:…])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_SPACED_HYPHEN = re.compile(r"(?<=\S) [-–] (?=\S)")


def format_timestamp(ms: int) -> str:
    ms = max(0, ms)
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def normalize_typography(text: str, cfg: SubtitleConfig) -> str:
    """Приводить текст до української типографічної норми."""
    out = text.replace(" ", " ")
    out = out.replace("...", "…").replace("..", "…") if "..." in out else out
    out = _APOSTROPHE.sub("’", out)
    out = _SPACED_HYPHEN.sub(f" {cfg.dialogue_dash} ", out)

    # Парні лапки: перша " → відкрита, друга → закрита
    if '"' in out:
        pieces = out.split('"')
        rebuilt = pieces[0]
        for position, piece in enumerate(pieces[1:]):
            rebuilt += (cfg.quote_open if position % 2 == 0 else cfg.quote_close) + piece
        out = rebuilt

    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    out = _MULTI_SPACE.sub(" ", out)
    return out.strip()


def _split_score(left: str, right: str) -> float:
    """Чим менше, тим кращий розрив."""
    score = abs(len(left) - len(right)) * 1.0

    last_char = left[-1] if left else ""
    if last_char in _STRONG_PUNCT:
        score -= 30
    elif last_char == _WEAK_PUNCT:
        score -= 18

    last_word = re.sub(r"[^\w’-]", "", left.split()[-1].lower()) if left.split() else ""
    if last_word in _CLINGING:
        score += 60  # службове слово в кінці рядка — найгірше, що можна зробити

    first_word = re.sub(r"[^\w’-]", "", right.split()[0].lower()) if right.split() else ""
    if first_word in _BREAK_BEFORE:
        score -= 12

    # Верхній рядок коротший за нижній читається краще, ніж навпаки
    if len(left) > len(right):
        score += 4
    return score


def wrap_text(text: str, max_chars: int, max_lines: int = 2) -> list[str]:
    """Ріже текст на рядки за нормами субтитрування.

    Якщо вписатися неможливо — повертає найкращий варіант із перевищенням;
    порушення потім піймає subtitle_lint і воно потрапить у звіт.
    """
    text = text.strip()
    if not text:
        return [""]
    if len(text) <= max_chars or max_lines <= 1:
        return [text]

    words = text.split()
    if len(words) == 1:
        return [text]

    if max_lines == 2:
        best: tuple[float, str, str] | None = None
        for cut in range(1, len(words)):
            left = " ".join(words[:cut])
            right = " ".join(words[cut:])
            score = _split_score(left, right)
            if len(left) > max_chars or len(right) > max_chars:
                score += 1000 + max(len(left), len(right)) - max_chars
            if best is None or score < best[0]:
                best = (score, left, right)
        assert best is not None
        return [best[1], best[2]]

    # Більше двох рядків: жадібне заповнення з тими самими штрафами на краях
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and len(candidate) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
        if len(lines) == max_lines - 1:
            break
    remaining = words[sum(len(l.split()) for l in lines) :]
    if remaining:
        lines.append(" ".join(remaining))
    return lines or [text]


def _layout(cue: Cue, uk_text: str, cfg: SubtitleConfig) -> list[str]:
    """Повертає готові рядки репліки з поверненою розміткою."""
    dash_lines = [line for line in cue.lines if line.dialogue_dash]
    uk_text = normalize_typography(uk_text, cfg)

    # Діалог двох мовців в одній репліці: модель зберігає розриви, ми їх шануємо
    if len(dash_lines) >= 2 and "\n" in uk_text:
        parts = [p.strip() for p in uk_text.split("\n") if p.strip()]
        out: list[str] = []
        for position, part in enumerate(parts):
            template = cue.lines[position] if position < len(cue.lines) else cue.lines[-1]
            wrapped = wrap_text(part, cfg.max_chars_per_line, 1)
            out.append(template.render(wrapped[0], cfg.dialogue_dash))
        return out

    flat = " ".join(part.strip() for part in uk_text.split("\n") if part.strip())
    template = cue.lines[0] if cue.lines else None
    budget = cfg.max_chars_per_line
    if template is not None and template.dialogue_dash:
        budget -= 2  # тире плюс пробіл їдять місце
    wrapped = wrap_text(flat, budget, cfg.max_lines)

    if template is None:
        return wrapped
    rendered = [template.render(wrapped[0], cfg.dialogue_dash)]
    for extra in wrapped[1:]:
        # Продовження тієї самої реплики: та сама обгортка, без тире та префіксів
        continuation = type(template)(
            text=extra,
            prefix="",
            suffix=template.suffix if extra is wrapped[-1] else "",
            wrap_tags=list(template.wrap_tags),
            partial_markup=template.partial_markup,
            dialogue_dash=False,
        )
        rendered.append(continuation.render(extra, cfg.dialogue_dash))
    if len(rendered) > 1 and template.suffix:
        rendered[0] = template.render(wrapped[0], cfg.dialogue_dash).replace(
            template.suffix, "", 1
        )
    return rendered


def layout_cue(cue: Cue, uk_text: str, cfg: SubtitleConfig) -> list[str]:
    """Рядки, які реально побачить глядач — потрібні лінтерам до запису файлу."""
    return _layout(cue, uk_text, cfg)


def render_srt(
    cues: list[Cue],
    translations: dict[int, str],
    cfg: SubtitleConfig,
    labels: dict[int, str] | None = None,
) -> str:
    """Збирає SRT. Індекси та рядки таймкодів беруться з джерела без змін.

    `labels` — українські мітки мовця для реплік, де в джерелі було «MARCUS:»
    (такі мітки є в SDH-субтитрах і їх треба перекладати, а не викидати).
    """
    labels = labels or {}
    blocks: list[str] = []
    for cue in cues:
        uk = translations.get(cue.index)
        if uk is None or not uk.strip():
            body_lines = [line.render(None, cfg.dialogue_dash) for line in cue.lines]
        else:
            label = labels.get(cue.index)
            if label:
                uk = f"{label}: {uk.lstrip()}"
            body_lines = _layout(cue, uk, cfg)
        body = "\n".join(line for line in body_lines if line.strip() or not body_lines)
        blocks.append(f"{cue.index}\n{cue.timing_line}\n{body}")
    return "\n\n".join(blocks) + "\n"


def render_bilingual(
    cues: list[Cue], translations: dict[int, str], cfg: SubtitleConfig
) -> str:
    """UA зверху, EN курсивом знизу — для рецензування, не для перегляду."""
    blocks: list[str] = []
    for cue in cues:
        uk = (translations.get(cue.index) or "").strip()
        uk_lines = _layout(cue, uk, cfg) if uk else ["—"]
        en = cue.flat_text.replace("\n", " ")
        body = "\n".join(uk_lines + [f"<i>{en}</i>"])
        blocks.append(f"{cue.index}\n{cue.timing_line}\n{body}")
    return "\n\n".join(blocks) + "\n"
