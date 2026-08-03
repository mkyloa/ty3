"""Прохід ⑤: біблія перекладу.

Єдиний виклик, який бачить весь фільм одразу — і єдина можливість ухвалити
рішення глобально: матрицю ти/Ви, написання імен, драбинку інвективи. Далі ці
рішення лише виконуються.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import prompts
from ..model.backends.base import Prompt
from ..model.runner import Runner
from ..model.schemas import Bible, Dossier, SpeakerMap
from ..srtio.parse import Cue
from .context import bible_context, dossier_context, transcript

__all__ = ["build_bible", "load_bible", "bible_to_markdown"]


def build_bible(
    runner: Runner,
    cues: list[Cue],
    dossier: Dossier,
    speakers: SpeakerMap,
    *,
    effort: str = "high",
) -> Bible:
    pairs = _observed_pairs(speakers)
    user = "\n\n".join(
        part
        for part in [
            "Повний англійський скрипт фільму:",
            transcript(cues),
            (
                "Пари персонажів, які реально розмовляють між собою (для матриці ти/Ви):\n"
                + "\n".join(f"- {a} → {b}" for a, b in pairs)
                if pairs
                else ""
            ),
            "Ухвали рішення за всіма розділами: матриця ти/Ви з переходами, картки "
            "голосів, глосарій, звертання, повторювані репліки, драбинка інвективи, "
            "чорний список, відкриті питання.",
        ]
        if part
    )
    completion = runner.run_structured(
        "bible",
        Prompt(
            system=prompts.load("system_bible"),
            context=dossier_context(dossier),
            user=user,
        ),
        Bible,
        effort=effort,
    )
    bible = Bible.model_validate(completion.data or {})
    return _merge_official_names(bible, dossier)


def _observed_pairs(speakers: SpeakerMap) -> list[tuple[str, str]]:
    """Хто до кого справді звертається — щоб модель не вигадувала пар."""
    pairs: set[tuple[str, str]] = set()
    for item in speakers.cues:
        if item.speaker in ("", "UNKNOWN", "NARRATOR"):
            continue
        for addressee in item.addressees:
            if addressee and addressee != item.speaker:
                pairs.add((item.speaker, addressee))
    return sorted(pairs)


def _merge_official_names(bible: Bible, dossier: Dossier) -> Bible:
    """Офіційні написання з досьє мусять бути в глосарії — це не на розсуд моделі."""
    from ..model.schemas import GlossaryEntry

    already_listed = {entry.en.casefold() for entry in bible.glossary}
    bible.glossary = list(bible.glossary) + [
        GlossaryEntry(
            en=character.name_en,
            uk=character.name_uk,
            kind="person",
            note="офіційне написання з досьє",
        )
        for character in dossier.characters
        if character.name_uk and character.name_en.casefold() not in already_listed
    ]
    return bible


def load_bible(path: str | Path) -> Bible:
    return Bible.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def bible_to_markdown(bible: Bible) -> str:
    """Читабельний вигляд для людини — той самий текст, що йде в промпт."""
    text = bible_context(bible)
    if bible.open_questions:
        text += "\n\n## Відкриті питання для людини\n" + "\n".join(
            f"- {question}" for question in bible.open_questions
        )
    return text + "\n"
