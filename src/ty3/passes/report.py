"""Прохід ⑩: QA-звіт.

Звіт існує, щоб людина не читала 1500 реплик підряд. Він показує рівно ті місця,
де переклад міг зламатися: низька впевненість, втрачена гра слів, порушення норм,
які не вдалося виправити. Усе решта — статистика, щоб бачити тренд між прогонами.
"""

from __future__ import annotations

from collections import Counter

from ..checks.base import Issue, Severity
from ..config import Config
from ..model.schemas import Bible, CueSpeaker, Dossier
from ..srtio.parse import Cue
from .translate import TranslationResult

__all__ = ["build_report"]

_LOW_CONFIDENCE = 0.7


def build_report(
    cues: list[Cue],
    result: TranslationResult,
    issues: list[Issue],
    dossier: Dossier,
    bible: Bible,
    speakers: dict[int, CueSpeaker],
    cfg: Config,
    *,
    stats: dict | None = None,
) -> str:
    by_index = {cue.index: cue for cue in cues}
    translatable = [cue for cue in cues if cue.is_translatable]
    errors = [issue for issue in issues if issue.severity is Severity.ERROR]
    warnings = [issue for issue in issues if issue.severity is Severity.WARNING]

    lines: list[str] = [
        f"# QA-звіт: {dossier.title_uk or dossier.title_en}",
        "",
        f"Джерело: {dossier.title_en}"
        + (f" ({dossier.year})" if dossier.year else "")
        + (f", реж. {dossier.director}" if dossier.director else ""),
        f"Впевненість ідентифікації фільму: {dossier.identification_confidence:.0%}",
        "",
        "## Статистика",
        "",
        f"- Реплик усього: {len(cues)}, з текстом: {len(translatable)}",
        f"- Перекладено: {len(result.text)}",
    ]
    if result.missing:
        lines.append(f"- **Пропущено: {len(result.missing)}** — {result.missing[:20]}")
    lines += [
        f"- Правок редактора: {len(result.edits)}",
        f"- Помилок лінтера після виправлень: {len(errors)}",
        f"- Попереджень: {len(warnings)}",
    ]
    if stats:
        cost = stats.get("cost_usd")
        lines += [
            f"- Викликів моделі: {stats.get('billed_calls')} "
            f"(з кешу: {stats.get('cached_calls')})",
            f"- Токенів: вхід {stats.get('input_tokens')}, вихід {stats.get('output_tokens')}, "
            f"кеш-читання {stats.get('cache_read_tokens')}",
        ]
        if cost is not None:
            lines.append(f"- Витрати за оцінкою бекенда: ${cost:.2f}")

    # --- те, що вимагає людини ---
    if bible.open_questions:
        lines += ["", "## Питання, які модель не змогла вирішити сама", ""]
        lines += [f"- {question}" for question in bible.open_questions]

    unknown_gender = sorted(
        {item.speaker for item in speakers.values() if item.speaker_gender == "unknown"}
    )
    if unknown_gender:
        lines += [
            "",
            "## Персонажі з невизначеною статтю",
            "",
            "Для них рід дієслів обходився конструкціями без роду — перевір, чи це доречно.",
            "",
        ]
        lines += [f"- {name}" for name in unknown_gender]

    lines += _flag_section(result, by_index)
    lines += _low_confidence_section(result, by_index)
    lines += _issue_section("Помилки, які лишилися", errors, by_index, result)
    lines += _issue_section("Попередження", warnings, by_index, result)
    lines += _edits_section(result)
    lines += _glossary_section(bible)

    return "\n".join(lines) + "\n"


def _flag_section(result: TranslationResult, by_index: dict[int, Cue]) -> list[str]:
    if not result.flags:
        return []
    counts = Counter(flag for flags in result.flags.values() for flag in flags)
    lines = ["", "## Прапорці перекладача", "", "Зведення: " + ", ".join(
        f"{flag} — {count}" for flag, count in counts.most_common()
    ), ""]

    priority = ("wordplay_lost", "too_long", "gender_guessed", "ambiguous_source")
    for flag in priority:
        flagged = [index for index, flags in result.flags.items() if flag in flags]
        if not flagged:
            continue
        lines += [f"### {flag} ({len(flagged)})", ""]
        for index in sorted(flagged)[:30]:
            lines.append(_cue_line(index, by_index, result))
        if len(flagged) > 30:
            lines.append(f"- …ще {len(flagged) - 30}")
        lines.append("")
    return lines


def _low_confidence_section(
    result: TranslationResult, by_index: dict[int, Cue]
) -> list[str]:
    low = sorted(
        index
        for index, value in result.confidence.items()
        if value and value < _LOW_CONFIDENCE
    )
    if not low:
        return []
    lines = ["", f"## Низька впевненість (< {_LOW_CONFIDENCE:.0%}): {len(low)}", ""]
    for index in low[:40]:
        note = result.notes.get(index)
        lines.append(
            _cue_line(index, by_index, result)
            + (f"\n  - {note}" if note else "")
            + f"\n  - впевненість: {result.confidence[index]:.0%}"
        )
    if len(low) > 40:
        lines.append(f"- …ще {len(low) - 40}")
    return lines


def _issue_section(
    title: str, issues: list[Issue], by_index: dict[int, Cue], result: TranslationResult
) -> list[str]:
    if not issues:
        return []
    counts = Counter(issue.code for issue in issues)
    lines = ["", f"## {title}: {len(issues)}", "", "Зведення: " + ", ".join(
        f"{code} — {count}" for code, count in counts.most_common()
    ), ""]
    for issue in issues[:60]:
        lines.append(f"- **#{issue.cue}** {issue.code}: {issue.message}")
        if issue.detail:
            lines.append(f"  - {issue.detail}")
        if issue.cue in result.text:
            lines.append(f"  - переклад: {result.text[issue.cue]!r}")
    if len(issues) > 60:
        lines.append(f"- …ще {len(issues) - 60}")
    return lines


def _edits_section(result: TranslationResult) -> list[str]:
    """Точкові виправлення за лінтером (переклад/редактура тепер один виклик —
    самоперевірка модель робить у ньому самому, тут лишається лише те, що
    впіймали детерміновані перевірки після нього)."""
    if not result.edits:
        return []
    reasons = Counter(
        reason for reasons_list in result.edits.values() for reason in reasons_list
    )
    lines = [
        "",
        f"## Точкові виправлення за лінтером: {len(result.edits)}",
        "",
        "Найчастіші коди порушень — підказка, що додати в `prompts/uk_rules.md` "
        "чи в чорний список `prompts/russianisms.tsv`:",
        "",
    ]
    lines += [f"- {reason} — {count}" for reason, count in reasons.most_common(20)]
    return lines


def _glossary_section(bible: Bible) -> list[str]:
    if not bible.glossary:
        return []
    lines = ["", "## Використаний глосарій", "", "| EN | UA |", "|---|---|"]
    lines += [f"| {entry.en} | {entry.uk} |" for entry in bible.glossary]
    if bible.formality:
        lines += ["", "## Матриця ти/Ви", "", "| Мовець | Адресат | Форма |", "|---|---|---|"]
        lines += [
            f"| {pair.speaker} | {pair.addressee} | {pair.form} |"
            for pair in bible.formality
        ]
    return lines


def _cue_line(index: int, by_index: dict[int, Cue], result: TranslationResult) -> str:
    cue = by_index.get(index)
    source = cue.flat_text if cue else "?"
    return f"- **#{index}** EN: {source!r}\n  - UA: {result.text.get(index, '')!r}"
