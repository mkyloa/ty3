"""Серіалізація контексту для промптів.

Формат подачі реплик — не косметика. Модель мусить бачити рід мовця, форму
звертання й бюджет символів **поряд із текстом**, інакше вона їх не використає,
навіть якщо все це є десь вище в промпті.
"""

from __future__ import annotations

from ..config import SubtitleConfig
from ..model.schemas import Bible, CueSpeaker, Dossier
from ..srtio.parse import Cue

__all__ = [
    "transcript",
    "cue_block",
    "chunk_body",
    "dossier_context",
    "bible_context",
]


def transcript(cues: list[Cue], *, with_hints: bool = True) -> str:
    """Компактний скрипт для проходів, які читають фільм цілком."""
    lines: list[str] = []
    for cue in cues:
        if not cue.is_translatable:
            continue
        label = ""
        if with_hints and cue.speaker_hint:
            note = f" ({cue.speaker_note})" if cue.speaker_note else ""
            label = f"{cue.speaker_hint}{note}: "
        text = cue.flat_text
        if cue.kind.value != "dialogue":
            text = f"[{cue.kind.value}] {text}"
        lines.append(f"{cue.index} {label}{text}")
    return "\n".join(lines)


def cue_block(
    cue: Cue,
    cfg: SubtitleConfig,
    speaker: CueSpeaker | None = None,
    *,
    context_only: bool = False,
    uk_text: str | None = None,
) -> str:
    """Одна репліка з усім, що потрібно для правильного українського рядка."""
    if context_only:
        head = f"#{cue.index} КОНТЕКСТ  EN: {cue.flat_text}"
        return head + (f"\n           UA: {uk_text}" if uk_text else "")

    budget = cue.char_budget(cfg.max_cps, cfg.max_lines, cfg.max_chars_per_line)
    parts = [
        f"#{cue.index} [{cue.kind.value}] {cue.duration_ms / 1000:.1f}с budget={budget}",
        f"EN: {cue.text}",
    ]
    if speaker is not None:
        who = f"{speaker.speaker} ({speaker.speaker_gender})"
        to = ", ".join(speaker.addressees) or "—"
        parts.append(
            f"WHO: {who} → {to} ({speaker.addressee_number}), звертання: {speaker.formality}"
        )
    if cue.speaker_hint:
        parts.append(f"МІТКА: {cue.speaker_hint}")
    if len(cue.lines) > 1 and all(line.dialogue_dash for line in cue.lines):
        parts.append("УВАГА: два мовці в одній репліці — збережи розрив рядка")
    return "\n".join(parts)


def chunk_body(
    cues: list[Cue],
    cfg: SubtitleConfig,
    speakers: dict[int, CueSpeaker],
    *,
    before: list[Cue] | None = None,
    after: list[Cue] | None = None,
    translated: dict[int, str] | None = None,
) -> str:
    """Тіло запиту на переклад чанка: контекст ліворуч, робота, контекст праворуч."""
    translated = translated or {}
    blocks: list[str] = []

    for cue in before or []:
        blocks.append(
            cue_block(cue, cfg, context_only=True, uk_text=translated.get(cue.index))
        )
    if before:
        blocks.append("--- ПЕРЕКЛАДАТИ ЗВІДСИ ---")

    for cue in cues:
        blocks.append(cue_block(cue, cfg, speakers.get(cue.index)))

    if after:
        blocks.append("--- ДАЛІ ЛИШЕ КОНТЕКСТ ---")
    for cue in after or []:
        blocks.append(cue_block(cue, cfg, context_only=True))

    return "\n\n".join(blocks)


def dossier_context(dossier: Dossier) -> str:
    """Досьє у стабільному текстовому вигляді — частина кешованого префікса."""
    lines = ["# Досьє фільму", ""]
    lines.append(f"Назва: {dossier.title_en}" + (f" / {dossier.title_uk}" if dossier.title_uk else ""))
    if dossier.year:
        lines.append(f"Рік: {dossier.year}")
    if dossier.director:
        lines.append(f"Режисер: {dossier.director}")
    if dossier.genre:
        lines.append(f"Жанр: {', '.join(dossier.genre)}")
    if dossier.setting:
        lines.append(f"Місце й час дії: {dossier.setting}")
    if dossier.rating:
        lines.append(f"Рейтинг: {dossier.rating} (визначає межі інвективи)")
    if dossier.synopsis:
        lines += ["", "## Синопсис", dossier.synopsis]
    if dossier.plot_beats:
        lines += ["", "## Сюжетні точки"] + [f"- {beat}" for beat in dossier.plot_beats]

    if dossier.characters:
        lines += ["", "## Персонажі", "", "| EN | UA | Стать | Роль | Мовлення |", "|---|---|---|---|---|"]
        for character in dossier.characters:
            lines.append(
                f"| {character.name_en} | {character.name_uk or '—'} | {character.gender} "
                f"| {character.role or '—'} | {character.speech_notes or '—'} |"
            )
    if dossier.notes:
        lines += ["", f"Примітки: {dossier.notes}"]
    return "\n".join(lines)


def bible_context(bible: Bible) -> str:
    """Біблія у стабільному текстовому вигляді — частина кешованого префікса."""
    lines = ["# Біблія перекладу", ""]
    if bible.register_overall:
        lines.append(f"Загальний реєстр: {bible.register_overall}")
    if bible.era_notes:
        lines.append(f"Епоха й лексика: {bible.era_notes}")

    if bible.formality:
        lines += ["", "## Матриця ти/Ви", "", "| Мовець | Адресат | Форма | Перехід |", "|---|---|---|---|"]
        for pair in bible.formality:
            switch = (
                f"з #{pair.switches_at_cue} → {pair.switches_to}"
                if pair.switches_at_cue and pair.switches_to
                else "—"
            )
            lines.append(f"| {pair.speaker} | {pair.addressee} | {pair.form} | {switch} |")

    if bible.voices:
        lines += ["", "## Голоси персонажів"]
        for voice in bible.voices:
            bits = [f"реєстр: {voice.speech_register}", f"лайка: {voice.swearing}"]
            if voice.quirks:
                bits.append("звички: " + ", ".join(voice.quirks))
            if voice.catchphrases:
                bits.append("фрази: " + ", ".join(voice.catchphrases))
            lines.append(f"- **{voice.character}** — " + "; ".join(bits))

    for title, entries in (
        ("Глосарій", bible.glossary),
        ("Звертання", bible.honorifics),
    ):
        if entries:
            lines += ["", f"## {title}"]
            for entry in entries:
                note = f" — {entry.note}" if entry.note else ""
                lines.append(f"- {entry.en} → **{entry.uk}**{note}")

    if bible.recurring:
        lines += ["", "## Повторювані репліки (перекладати однаково)"]
        for item in bible.recurring:
            lines.append(f"- «{item.en}» → «{item.uk}»")

    if bible.profanity_ladder:
        lines += ["", "## Драбинка інвективи"]
        for rung in bible.profanity_ladder:
            lines.append(f"- {rung.en} → {rung.uk} ({rung.strength})")

    if bible.avoid:
        lines += ["", "## Не вживати"]
        for item in bible.avoid:
            why = f" — {item.why}" if item.why else ""
            lines.append(f"- ~~{item.avoid}~~ → {item.use}{why}")

    return "\n".join(lines)
