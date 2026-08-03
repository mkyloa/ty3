"""Проходи ⑧ і ⑨: точкові виправлення за лінтерами та підгонка під норми.

Ідея одна: спершу детерміновані перевірки називають конкретні репліки, потім
модель викликається **лише для них**. Так виправлення дешеві й адресні, а не
«перекладімо все ще раз і сподіваймося».
"""

from __future__ import annotations

from .. import prompts
from ..checks.base import Issue
from ..checks.subtitle_lint import check_all
from ..checks.uk_lint import check_uk_all, load_russianisms
from ..config import Config
from ..model.backends.base import Prompt
from ..model.runner import Runner
from ..model.schemas import (
    Bible,
    ChunkFit,
    ChunkTranslation,
    CueSpeaker,
    Dossier,
)
from ..srtio.parse import Cue
from ..srtio.render import layout_cue
from .context import bible_context, cue_block, dossier_context
from .translate import TranslationResult

__all__ = ["lint", "fix_flagged", "fit_overlong", "rendered_lines", "allowed_latin_from"]

_MAX_FIX_BATCH = 12


def rendered_lines(
    cues: list[Cue], translations: dict[int, str], cfg: Config
) -> dict[int, list[str]]:
    """Готові рядки для кожної реплики — те, що реально побачить глядач."""
    output: dict[int, list[str]] = {}
    for cue in cues:
        uk = translations.get(cue.index)
        if uk:
            output[cue.index] = layout_cue(cue, uk, cfg.subtitle)
    return output


def allowed_latin_from(bible: Bible) -> set[str]:
    """Латиниця, яка законно лишається в перекладі: терміни з глосарію."""
    allowed: set[str] = set()
    for entry in list(bible.glossary) + list(bible.honorifics):
        for word in entry.uk.split():
            if any(char.isascii() and char.isalpha() for char in word):
                allowed.add(word.strip(".,!?;:«»\"'"))
    return allowed


def lint(
    cues: list[Cue],
    result: TranslationResult,
    bible: Bible,
    speakers: dict[int, CueSpeaker],
    cfg: Config,
) -> list[Issue]:
    """Усі детерміновані перевірки разом."""
    issues = check_all(
        cues,
        rendered_lines(cues, result.text, cfg),
        cfg.subtitle,
        allowed_latin=allowed_latin_from(bible),
    )
    issues += check_uk_all(
        cues,
        result.text,
        bible=bible,
        speakers=speakers,
        rules=load_russianisms(),
    )
    return issues


def fix_flagged(
    runner: Runner,
    cues: list[Cue],
    result: TranslationResult,
    issues: list[Issue],
    dossier: Dossier,
    bible: Bible,
    speakers: dict[int, CueSpeaker],
    cfg: Config,
) -> TranslationResult:
    """Виправляє репліки, які лінтери позначили як помилки."""
    by_cue: dict[int, list[Issue]] = {}
    for issue in issues:
        if issue.needs_fix and issue.cue in result.text:
            by_cue.setdefault(issue.cue, []).append(issue)
    if not by_cue:
        return result

    by_index = {cue.index: cue for cue in cues}
    targets = sorted(by_cue)
    batches = [
        targets[start : start + _MAX_FIX_BATCH]
        for start in range(0, len(targets), _MAX_FIX_BATCH)
    ]

    system = prompts.load_many("system_edit", "uk_rules", "en_uk_traps")
    context = f"{dossier_context(dossier)}\n\n{bible_context(bible)}"

    prompt_list = []
    for batch in batches:
        blocks = []
        for index in batch:
            cue = by_index[index]
            problems = "\n".join(f"  - {issue}" for issue in by_cue[index])
            blocks.append(
                cue_block(cue, cfg.subtitle, speakers.get(index))
                + f"\nПОТОЧНИЙ ПЕРЕКЛАД: {result.text[index]}"
                + f"\nЗНАЙДЕНІ ПОРУШЕННЯ:\n{problems}"
            )
        prompt_list.append(
            Prompt(
                system=system,
                context=context,
                user=(
                    "Виправ саме названі порушення. Не переписуй репліку заново, якщо "
                    "порушення можна прибрати меншою правкою.\n\n" + "\n\n".join(blocks)
                ),
            )
        )

    completions = runner.map_structured(
        "fix",
        prompt_list,
        ChunkTranslation,
        effort=cfg.model.effort,
        max_tokens=cfg.model.max_tokens,
    )
    for completion in completions:
        payload = ChunkTranslation.model_validate(completion.data or {})
        for item in payload.cues:
            if item.cue in result.text and item.uk.strip():
                result.text[item.cue] = item.uk.strip()
                # Коди лінтера, а не загальна фраза — саме вони підказують,
                # що додати в prompts/uk_rules.md перед наступним фільмом.
                codes = sorted({issue.code for issue in by_cue.get(item.cue, [])})
                result.edits.setdefault(item.cue, []).extend(codes or ["лінтер"])
                if item.flags:
                    result.flags[item.cue] = sorted(
                        set(result.flags.get(item.cue, [])) | set(item.flags)
                    )
    return result


def fit_overlong(
    runner: Runner,
    cues: list[Cue],
    result: TranslationResult,
    dossier: Dossier,
    bible: Bible,
    speakers: dict[int, CueSpeaker],
    cfg: Config,
) -> TranslationResult:
    """Стискає репліки, що не влазять у бюджет символів."""
    rendered = rendered_lines(cues, result.text, cfg)
    by_index = {cue.index: cue for cue in cues}

    overlong = [
        issue.cue
        for issue in check_all(cues, rendered, cfg.subtitle)
        if issue.code in ("CPS_HIGH", "LINE_TOO_LONG", "TOO_MANY_LINES")
    ]
    targets = sorted(set(overlong))
    if not targets:
        return result

    system = prompts.load_many("system_fit", "subtitle_norms")
    context = f"{dossier_context(dossier)}\n\n{bible_context(bible)}"

    prompt_list = []
    batches = [
        targets[start : start + _MAX_FIX_BATCH]
        for start in range(0, len(targets), _MAX_FIX_BATCH)
    ]
    for batch in batches:
        blocks = []
        for index in batch:
            cue = by_index[index]
            budget = cue.char_budget(
                cfg.subtitle.max_cps, cfg.subtitle.max_lines, cfg.subtitle.max_chars_per_line
            )
            current = result.text[index]
            blocks.append(
                cue_block(cue, cfg.subtitle, speakers.get(index))
                + f"\nПОТОЧНИЙ ПЕРЕКЛАД ({len(current)} симв.): {current}"
                + f"\nМАКСИМУМ: {budget} симв."
            )
        prompt_list.append(
            Prompt(system=system, context=context, user="\n\n".join(blocks))
        )

    completions = runner.map_structured(
        "fit",
        prompt_list,
        ChunkFit,
        effort=cfg.model.effort_light,
        max_tokens=cfg.model.max_tokens,
    )
    for completion in completions:
        payload = ChunkFit.model_validate(completion.data or {})
        for item in payload.cues:
            if item.cue in result.text and item.uk.strip():
                result.text[item.cue] = item.uk.strip()
                if item.dropped:
                    result.notes[item.cue] = (
                        result.notes.get(item.cue, "") + f" Скорочено: {item.dropped}"
                    ).strip()
    return result
