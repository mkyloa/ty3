"""Прохід ④: хто говорить і до кого.

Найважливіший прохід для української мови. Англійський субтитр не містить ні статі
мовця, ні форми звертання, а українська репліка без них не будується: `I said`
може бути і «сказав», і «сказала», і вибір навмання помиляється в половині
випадків.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import prompts
from ..model.backends.base import Prompt
from ..model.runner import Runner
from ..model.schemas import CueSpeaker, Dossier, SpeakerMap
from ..srtio.parse import Cue
from ..srtio.segment import build_windows
from .context import dossier_context, transcript

__all__ = ["attribute_speakers", "load_speakers"]


def attribute_speakers(
    runner: Runner,
    cues: list[Cue],
    dossier: Dossier,
    *,
    window: int = 80,
    overlap: int = 12,
    effort: str = "high",
) -> SpeakerMap:
    """Проходить файл вікнами з перекриттям і збирає карту мовців."""
    translatable = [cue for cue in cues if cue.is_translatable]
    if not translatable:
        return SpeakerMap()

    windows = build_windows(translatable, size=window, overlap=overlap)
    system = prompts.load("system_speakers")
    context = dossier_context(dossier)

    collected: dict[int, CueSpeaker] = {}
    for number, (before, chunk) in enumerate(windows, start=1):
        body_parts = []
        if before:
            body_parts.append(
                "КОНТЕКСТ (не описувати, лише для розуміння):\n" + transcript(before)
            )
        body_parts.append(
            "РЕПЛІКИ ДЛЯ АТРИБУЦІЇ — дай запис на кожен індекс:\n" + transcript(chunk)
        )
        body_parts.append(
            f"Індекси, які мусять бути у відповіді: {[cue.index for cue in chunk]}"
        )

        completion = runner.run_structured(
            "speakers",
            Prompt(system=system, context=context, user="\n\n".join(body_parts)),
            SpeakerMap,
            effort=effort,
        )
        window_map = SpeakerMap.model_validate(completion.data or {})
        for item in window_map.cues:
            # Перекриття вікон дає повтори: лишаємо впевненіший запис
            existing = collected.get(item.cue)
            if existing is None or item.confidence > existing.confidence:
                collected[item.cue] = item

    # Мітка мовця з самого субтитра надійніша за будь-яке виведення
    for cue in translatable:
        if not cue.speaker_hint:
            continue
        item = collected.get(cue.index)
        if item is None:
            collected[cue.index] = CueSpeaker(
                cue=cue.index,
                speaker=cue.speaker_hint,
                speaker_gender=dossier.gender_of(cue.speaker_hint),
                confidence=0.9,
                evidence="явна мітка мовця в субтитрі",
            )
        elif item.speaker.casefold() != cue.speaker_hint.casefold():
            collected[cue.index] = item.model_copy(
                update={
                    "speaker": cue.speaker_hint,
                    "speaker_gender": dossier.gender_of(cue.speaker_hint)
                    or item.speaker_gender,
                    "confidence": max(item.confidence, 0.9),
                    "evidence": "явна мітка мовця в субтитрі",
                }
            )

    # Стать із досьє перекриває вгадану моделлю
    for index, item in collected.items():
        from_dossier = dossier.gender_of(item.speaker)
        if from_dossier != "unknown" and item.speaker_gender != from_dossier:
            collected[index] = item.model_copy(update={"speaker_gender": from_dossier})

    return SpeakerMap(cues=[collected[key] for key in sorted(collected)])


def load_speakers(path: str | Path) -> SpeakerMap:
    """Читає готову карту мовців з диска.

    Атрибуція — звичайні дані (хто говорить, кому, якою статтю), не привʼязані
    до моделі, що їх згенерувала. Тому вигідно рахувати цей прохід один раз і
    переносити результат між прогонами з різними бекендами/моделями для решти
    пайплайну — так само, як уже можна підкласти `dossier.json` чи `bible.json`.
    """
    return SpeakerMap.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
