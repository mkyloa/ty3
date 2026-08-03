"""Українські перевірки: русизми, ти/Ви, глосарій, консистентність.

Головна з них — ти/Ви. Помилка тут не виглядає як помилка: речення граматично
правильне, тому редактор-людина її пропускає, а глядач відчуває, що персонажі
поводяться не так, як мають. Детермінований сканер проти матриці з біблії ловить
це надійно, бо це чиста робота з множинами.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..model.schemas import Bible, CueSpeaker, GlossaryEntry
from ..srtio.parse import Cue
from .base import Issue, Severity

__all__ = [
    "AvoidRule",
    "load_russianisms",
    "check_russianisms",
    "check_formality",
    "check_glossary",
    "check_repeated_lines",
    "check_uk_all",
    "normalize_word",
    "same_word",
    "stem",
]

_RUSSIANISMS_PATH = Path(__file__).resolve().parent.parent / "prompts" / "russianisms.tsv"

# Займенники та присвійні — надійні маркери форми звертання.
# Дієслівні закінчення свідомо не беремо: 2-га особа множини й ввічлива форма
# збігаються, тому вони дають хибні спрацювання.
_TY_FORMS = {
    "ти", "тебе", "тобі", "тобою", "тя",
    "твій", "твоя", "твоє", "твої", "твого", "твоєї", "твоєму", "твоїй",
    "твоїм", "твоїх", "твоїми", "твою", "твоє",
}
_VY_FORMS = {
    "ви", "вас", "вам", "вами",
    "ваш", "ваша", "ваше", "ваші", "вашого", "вашої", "вашому", "вашій",
    "вашим", "ваших", "вашими", "вашу",
}

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True)
class AvoidRule:
    avoid: str
    use: str
    category: str = "lexical"
    note: str | None = None

    @property
    def pattern(self) -> re.Pattern[str]:
        return _compile_phrase(self.avoid)


@lru_cache(maxsize=256)
def _compile_phrase(phrase: str) -> re.Pattern[str]:
    """Шукає фразу з межами слів; усередині фрази допускає будь-які пробіли."""
    parts = [re.escape(word) for word in phrase.split()]
    return re.compile(r"(?<![^\W\d_])" + r"\s+".join(parts) + r"(?![^\W\d_])", re.IGNORECASE)


def load_russianisms(path: Path | None = None) -> list[AvoidRule]:
    source = path or _RUSSIANISMS_PATH
    rules: list[AvoidRule] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        rules.append(
            AvoidRule(
                avoid=fields[0].strip(),
                use=fields[1].strip(),
                category=fields[2].strip() if len(fields) > 2 else "lexical",
                note=fields[3].strip() if len(fields) > 3 else None,
            )
        )
    return rules


# Чергування голосних у закритому/відкритому складі: обхід→обходу, ніч→ночі,
# дім→дому, кінь→коня. Префікс основи через це не збігається, тому голосні
# і/о/е/и зводимо до одного класу.
_VOWEL_FOLD = str.maketrans({"і": "о", "е": "о", "и": "о", "ї": "о"})


def normalize_word(word: str) -> str:
    """Зводить слово до форми, стійкої до словозміни й чергувань."""
    normalized = unicodedata.normalize("NFC", word).casefold().replace("’", "'")
    normalized = normalized.rstrip("ь")
    return normalized.translate(_VOWEL_FOLD)


def same_word(first: str, second: str, floor: int = 3) -> bool:
    """Чи це те саме слово в різних формах.

    Порівнюємо спільний префікс завдовжки на один символ менше за коротшу з
    форм: цього досить, щоб «обхід»/«обходу» й «Маркус»/«Маркусе» збіглися, і
    щоб «обхід»/«візит» — ні. Точна лематизація потребувала б словника.

    Свідомо поблажливе: хибне «термін є» лише пропускає перевірку, а хибне
    «терміну немає» — помилка рівня ERROR, яка тягне зайвий виклик моделі.
    """
    left, right = normalize_word(first), normalize_word(second)
    if not left or not right:
        return left == right
    cut = max(floor, min(len(left), len(right)) - 1)
    return left[:cut] == right[:cut]


def stem(word: str) -> str:
    """Нормалізована основа. Для порівняння двох форм бери `same_word`."""
    return normalize_word(word)


# ---------- русизми ----------


def check_russianisms(
    cue_index: int, text: str, rules: list[AvoidRule]
) -> list[Issue]:
    issues: list[Issue] = []
    for rule in rules:
        match = rule.pattern.search(text)
        if match:
            severity = (
                Severity.ERROR if rule.category == "profanity" else Severity.WARNING
            )
            issues.append(
                Issue(
                    cue_index,
                    "RUSSIANISM",
                    severity,
                    f"«{match.group(0)}» → «{rule.use}»",
                    rule.note,
                )
            )
    return issues


# ---------- ти / Ви ----------


def _forms_in(text: str) -> tuple[set[str], set[str]]:
    words = {word.casefold() for word in _WORD.findall(text)}
    return words & _TY_FORMS, words & _VY_FORMS


def check_formality(
    cue_index: int,
    text: str,
    speaker: CueSpeaker | None,
    bible: Bible | None,
) -> list[Issue]:
    """Порівнює форму звертання в тексті з матрицею ти/Ви."""
    ty_found, vy_found = _forms_in(text)
    if not ty_found and not vy_found:
        return []

    issues: list[Issue] = []

    # Змішування форм в одній репліці до однієї особи — завжди помилка
    if ty_found and vy_found and speaker and speaker.addressee_number == "sg":
        issues.append(
            Issue(
                cue_index,
                "FORMALITY_MIXED",
                Severity.ERROR,
                "в одній репліці і «ти», і «Ви» до однієї особи",
                f"{sorted(ty_found)} + {sorted(vy_found)}",
            )
        )

    expected = _expected_form(speaker, bible, cue_index)
    if expected == "unknown":
        return issues

    if expected == "ти" and vy_found and (speaker and speaker.addressee_number == "sg"):
        issues.append(
            Issue(
                cue_index,
                "FORMALITY_MISMATCH",
                Severity.ERROR,
                "матриця вимагає «ти», у тексті ввічлива форма",
                ", ".join(sorted(vy_found)),
            )
        )
    if expected == "Ви" and ty_found:
        issues.append(
            Issue(
                cue_index,
                "FORMALITY_MISMATCH",
                Severity.ERROR,
                "матриця вимагає «Ви», у тексті «ти»",
                ", ".join(sorted(ty_found)),
            )
        )
    return issues


def _expected_form(
    speaker: CueSpeaker | None, bible: Bible | None, cue_index: int
) -> str:
    if speaker is None:
        return "unknown"
    if speaker.formality in ("ти", "Ви"):
        return speaker.formality
    if bible is None or not speaker.addressees:
        return "unknown"
    return bible.formality_for(speaker.speaker, speaker.addressees[0], cue_index)


# ---------- глосарій ----------


def check_glossary(
    cue: Cue, uk_text: str, entries: list[GlossaryEntry]
) -> list[Issue]:
    """Якщо в англійському рядку є термін із глосарію — його відповідник мусить бути в перекладі."""
    source = cue.flat_text
    if not source or not uk_text:
        return []

    text_words = _WORD.findall(uk_text)
    issues: list[Issue] = []
    for entry in entries:
        if not entry.en or not entry.uk:
            continue
        if not _compile_phrase(entry.en).search(source):
            continue
        expected = _WORD.findall(entry.uk)
        present = any(
            same_word(wanted, found) for wanted in expected for found in text_words
        )
        if expected and not present:
            issues.append(
                Issue(
                    cue.index,
                    "GLOSSARY_MISS",
                    Severity.ERROR,
                    f"«{entry.en}» мусить перекладатися як «{entry.uk}»",
                    entry.note,
                )
            )
    return issues


# ---------- консистентність ----------


def check_repeated_lines(
    cues: list[Cue],
    translations: dict[int, str],
    speakers: dict[int, CueSpeaker] | None = None,
) -> list[Issue]:
    """Однакові англійські рядки того самого мовця мусять перекладатися однаково."""
    speakers = speakers or {}
    groups: dict[tuple[str, str], list[int]] = {}
    for cue in cues:
        key_text = cue.flat_text.casefold().strip()
        if len(key_text) < 4:
            continue  # «Ok.», «Так.» законно перекладаються по-різному
        speaker = speakers.get(cue.index)
        groups.setdefault((key_text, speaker.speaker if speaker else ""), []).append(
            cue.index
        )

    issues: list[Issue] = []
    for (key_text, _speaker), indices in groups.items():
        if len(indices) < 2:
            continue
        variants = {
            translations[index].strip() for index in indices if translations.get(index)
        }
        if len(variants) > 1:
            issues.append(
                Issue(
                    indices[0],
                    "INCONSISTENT_REPEAT",
                    Severity.WARNING,
                    f"однаковий рядок перекладено по-різному в репліках {indices}",
                    " | ".join(sorted(variants)),
                )
            )
    return issues


# ---------- усе разом ----------


def check_uk_all(
    cues: list[Cue],
    translations: dict[int, str],
    *,
    bible: Bible | None = None,
    speakers: dict[int, CueSpeaker] | None = None,
    rules: list[AvoidRule] | None = None,
) -> list[Issue]:
    rules = rules if rules is not None else load_russianisms()
    speakers = speakers or {}
    glossary = list(bible.glossary) + list(bible.honorifics) if bible else []

    issues: list[Issue] = []
    for cue in cues:
        uk_text = translations.get(cue.index)
        if not uk_text:
            continue
        issues.extend(check_russianisms(cue.index, uk_text, rules))
        issues.extend(
            check_formality(cue.index, uk_text, speakers.get(cue.index), bible)
        )
        if glossary:
            issues.extend(check_glossary(cue, uk_text, glossary))
    issues.extend(check_repeated_lines(cues, translations, speakers))
    return issues
