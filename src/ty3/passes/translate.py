"""Прохід ⑤: переклад і редактура за один виклик на чанк.

Раніше це були два окремі проходи (переклад → окремий редакторський виклик, що
читав чернетку і виправляв). Об'єднано в один: модель проходить чекліст
редактора як самоперевірку перед відповіддю, а не в другому виклику над готовим
текстом. Причина суто економічна — заміряно на реальному прогоні: `translate`+
`edit` виконуються по разу на кожен чанк, і на повнометражному фільмі (~28
чанків) це найдорожча стаття витрат у всьому пайплайні. Об'єднання вдвічі
скорочує кількість викликів на цьому кроці, залишаючи ту саму перевірку.

Стабільний префікс промпта (правила + досьє + біблія) однаковий для всіх чанків —
саме тому він винесений у `Prompt.context` і кешується. Волатильна частина — самі
репліки чанка.
"""

from __future__ import annotations

from .. import prompts
from ..config import Config
from ..model.backends.base import Prompt
from ..model.runner import Runner
from ..model.schemas import Bible, ChunkTranslation, CueSpeaker, CueTranslation, Dossier
from ..srtio.segment import Chunk
from .context import bible_context, chunk_body, dossier_context

__all__ = ["translate_chunks", "TranslationResult"]


class TranslationResult:
    """Переклад + усе, що треба знати про його якість."""

    def __init__(self) -> None:
        self.text: dict[int, str] = {}
        self.flags: dict[int, list[str]] = {}
        self.notes: dict[int, str] = {}
        self.confidence: dict[int, float] = {}
        self.edits: dict[int, list[str]] = {}
        self.missing: list[int] = []

    def apply(self, item: CueTranslation) -> None:
        self.text[item.cue] = item.uk.strip()
        if item.flags:
            self.flags[item.cue] = list(item.flags)
        if item.note:
            self.notes[item.cue] = item.note
        self.confidence[item.cue] = item.confidence


def _stable_context(dossier: Dossier, bible: Bible) -> str:
    return f"{dossier_context(dossier)}\n\n{bible_context(bible)}"


def _translator_system() -> str:
    return prompts.load_many(
        "system_translate_edit", "uk_rules", "en_uk_traps", "subtitle_norms"
    )


def translate_chunks(
    runner: Runner,
    chunks: list[Chunk],
    dossier: Dossier,
    bible: Bible,
    speakers: dict[int, CueSpeaker],
    cfg: Config,
) -> TranslationResult:
    """Перекладає чанки. Послідовно — з готовим українським контекстом ліворуч."""
    result = TranslationResult()
    system = _translator_system()
    context = _stable_context(dossier, bible)

    if cfg.run.jobs > 1:
        # Паралель по чанках: контекст ліворуч подається англійською, бо
        # українського ще немає. Дрейф між чанками вирівняє прохід консистентності.
        bodies = [
            chunk_body(
                chunk.cues, cfg.subtitle, speakers,
                before=chunk.context_before, after=chunk.context_after,
            )
            for chunk in chunks
        ]
        completions = runner.map_structured(
            "translate",
            [Prompt(system=system, context=context, user=body) for body in bodies],
            ChunkTranslation,
            effort=cfg.model.effort,
            max_tokens=cfg.model.max_tokens,
        )
        for chunk, completion in zip(chunks, completions):
            _absorb(result, chunk, completion.data)
        return result

    for chunk in chunks:
        body = chunk_body(
            chunk.cues,
            cfg.subtitle,
            speakers,
            before=chunk.context_before,
            after=chunk.context_after,
            translated=result.text,  # уже перекладене — найкращий контекст
        )
        completion = runner.run_structured(
            "translate",
            Prompt(system=system, context=context, user=body),
            ChunkTranslation,
            effort=cfg.model.effort,
            max_tokens=cfg.model.max_tokens,
        )
        _absorb(result, chunk, completion.data)
    return result


def _absorb(result: TranslationResult, chunk: Chunk, data: dict | None) -> None:
    payload = ChunkTranslation.model_validate(data or {})
    expected = {cue.index for cue in chunk.cues if cue.is_translatable}
    for item in payload.cues:
        if item.cue in expected:
            result.apply(item)
    result.missing.extend(sorted(expected - set(result.text)))
