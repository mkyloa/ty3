"""Прохід ③: автопошук контексту фільму.

Два виклики, а не один: структурований вивід і серверний вебпошук погано
поєднуються в одному запиті у всіх провайдерів (Gemini і Anthropic це прямо
забороняють). Тому спершу вільний пошук, потім окремий виклик-екстрактор у схему.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .. import prompts
from ..model.backends.base import Prompt, Tools
from ..model.runner import Runner
from ..model.schemas import Dossier
from ..srtio.parse import Cue

__all__ = ["guess_title_from_filename", "research", "load_dossier"]

# Сміття з імен релізів, яке треба зчистити, перш ніж шукати фільм
_RELEASE_NOISE = re.compile(
    r"\b("
    r"\d{3,4}p|[xh]\.?26[45]|hevc|aac\d?|ac3|dts|ddp?5\.1|\d+bit|"
    r"web-?dl|web-?rip|web|dl|bluray|blu-?ray|bdrip|dvdrip|hdrip|hdtv|remux|"
    r"uhd|hdr\d*|dv|amzn|nf|hulu|dsnp|atvp|max|proper|repack|extended|uncut|"
    r"unrated|internal|limited|multi|dual|sub|subs|eng|ukr|hi|sdh|cc|forced"
    r")\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
_EPISODE = re.compile(r"\bS\d{1,2}E\d{1,3}\b", re.IGNORECASE)


def guess_title_from_filename(path: str | Path) -> tuple[str, int | None]:
    """Витягує назву й рік з імені файлу релізу.

    `The.Substance.2024.1080p.WEB-DL.x264-GROUP.en.srt` → ("The Substance", 2024)
    Результат — підказка, не істина: модель підтверджує його за репліками.
    """
    stem = Path(path).stem
    stem = re.sub(r"[._]+", " ", stem)

    year_match = _YEAR.search(stem)
    year = int(year_match.group(1)) if year_match else None
    if year_match:
        stem = stem[: year_match.start()]  # усе після року — технічні теги

    stem = _EPISODE.sub(" ", stem)
    # Технічні теги знімаємо ДО назви релізної групи: інакше «WEB-DL» втрачає
    # «-DL» як групу, а «WEB» лишається в назві фільму.
    stem = _RELEASE_NOISE.sub(" ", stem)
    stem = re.sub(r"-[A-Za-z0-9]{2,15}\s*$", " ", stem)  # -GROUP у кінці
    stem = re.sub(r"[\[\](){}]", " ", stem)
    title = re.sub(r"\s{2,}", " ", stem).strip(" -–—")
    return title, year


def _sample_dialogue(cues: list[Cue], limit: int = 60) -> str:
    """Вибірка реплик для підтвердження фільму: початок, середина, кінець."""
    speaking = [cue for cue in cues if cue.is_translatable and cue.kind.value == "dialogue"]
    if len(speaking) <= limit:
        chosen = speaking
    else:
        third = limit // 3
        middle = len(speaking) // 2
        chosen = (
            speaking[:third]
            + speaking[middle : middle + third]
            + speaking[-(limit - 2 * third) :]
        )
    return "\n".join(cue.flat_text for cue in chosen)


def research(
    runner: Runner,
    cues: list[Cue],
    *,
    source_path: str | Path,
    title: str | None = None,
    year: int | None = None,
    extra_context: str | None = None,
    effort: str = "high",
) -> Dossier:
    """Знаходить контекст фільму й повертає досьє."""
    guessed_title, guessed_year = guess_title_from_filename(source_path)
    title = title or guessed_title
    year = year or guessed_year

    # --- крок 1: вільний пошук ---
    search_user = "\n\n".join(
        part
        for part in [
            f"Ім'я файлу субтитрів: {Path(source_path).name}",
            f"Припущення з імені файлу: назва «{title}»" + (f", рік {year}" if year else ""),
            "Вибірка реплик для підтвердження, що це саме той фільм:\n" + _sample_dialogue(cues),
            (
                "Додатковий контекст від користувача — це джерело найвищого пріоритету, "
                f"воно перекриває будь-які знайдені дані:\n{extra_context}"
                if extra_context
                else ""
            ),
            "Знайди й викладі: ідентифікацію фільму, каст із персонажами та їхньою "
            "статтю, офіційні українські написання імен і назв, синопсис, сюжетні "
            "точки, епоху й місце дії, віковий рейтинг, внутрішньосвітові терміни. "
            "Для кожного твердження вкажи джерело.",
        ]
        if part
    )
    findings = runner.run(
        "research-search",
        Prompt(system=prompts.load("system_research"), user=search_user),
        effort=effort,
        tools=Tools.SEARCH,
    )

    # --- крок 2: структурування ---
    extract_user = "\n\n".join(
        [
            "Ось результати пошуку. Перенеси їх у схему без домислів: чого в тексті "
            "немає — лишай порожнім, не заповнюй здогадами.",
            findings.text,
            f"Назва з імені файлу (використай, якщо пошук не дав кращої): {title}",
        ]
    )
    extracted = runner.run_structured(
        "research-extract",
        Prompt(
            system=(
                "Ти переносиш результати дослідження у структуру. Не додавай нічого, "
                "чого немає у вхідному тексті. Порожнє поле краще за вигадане."
            ),
            user=extract_user,
        ),
        Dossier,
        effort="medium",
    )

    dossier = Dossier.model_validate(extracted.data or {})
    if not dossier.title_en:
        dossier.title_en = title
    if dossier.year is None:
        dossier.year = year
    return dossier


def load_dossier(path: str | Path) -> Dossier:
    """Читає готове досьє з диска — щоб можна було правити руками й перезапускати."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Dossier.model_validate(data)
