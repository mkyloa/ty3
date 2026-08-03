"""Схеми виводу для кожного проходу — контракт між пайплайном і моделлю.

Схеми тут не просто «щоб парсилося». Кожне поле — це рішення, яке інакше модель
ухвалювала б заново на кожному чанку і щоразу трохи інакше: стать персонажа,
форма звертання, написання імені. Витягнуті у схему, вони фіксуються один раз.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "Gender",
    "Formality",
    "Character",
    "GlossaryEntry",
    "Dossier",
    "CueSpeaker",
    "SpeakerMap",
    "VoiceCard",
    "FormalityPair",
    "RecurringLine",
    "ProfanityRung",
    "AvoidEntry",
    "Bible",
    "CueTranslation",
    "ChunkTranslation",
    "CueEdit",
    "ChunkEdit",
    "CueFit",
    "ChunkFit",
]

Gender = Literal["m", "f", "unknown"]
Formality = Literal["ти", "Ви", "unknown"]
TranslationFlag = Literal[
    "wordplay_lost",  # гру слів відтворити не вдалося
    "culture_specific",  # реалія, яка потребує адаптації
    "ambiguous_source",  # англійський рядок сам неоднозначний
    "gender_guessed",  # стать мовця вгадана, не підтверджена
    "too_long",  # переклад не вміщується в бюджет символів
    "name_unofficial",  # написання імені не підтверджене джерелом
]


# ---------- прохід research ----------


class Character(BaseModel):
    name_en: str = Field(description="Імʼя як у англійських субтитрах")
    name_uk: str | None = Field(
        default=None, description="Українське написання: офіційне, якщо існує"
    )
    gender: Gender = "unknown"
    actor: str | None = None
    role: str | None = Field(default=None, description="Роль у сюжеті одним рядком")
    age_range: str | None = None
    speech_notes: str | None = Field(
        default=None, description="Особливості мовлення: акцент, освіта, жаргон"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_url: str | None = None


class GlossaryEntry(BaseModel):
    en: str
    uk: str
    kind: Literal["person", "place", "org", "term", "title", "honorific", "other"] = "other"
    note: str | None = None


class Dossier(BaseModel):
    """Досьє фільму. Невідоме поле лишається порожнім — вигадувати заборонено."""

    title_en: str
    title_uk: str | None = None
    year: int | None = None
    director: str | None = None
    genre: list[str] = Field(default_factory=list)
    setting: str | None = Field(default=None, description="Епоха й місце дії")
    rating: str | None = Field(
        default=None, description="Віковий рейтинг — визначає політику інвективи"
    )
    synopsis: str = ""
    plot_beats: list[str] = Field(default_factory=list)
    characters: list[Character] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    identification_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str | None = None
    sources: list[str] = Field(default_factory=list)

    def gender_of(self, name: str | None) -> Gender:
        if not name:
            return "unknown"
        needle = name.strip().casefold()
        for character in self.characters:
            if character.name_en.strip().casefold() == needle:
                return character.gender
        return "unknown"


# ---------- прохід speakers ----------


class CueSpeaker(BaseModel):
    cue: int = Field(description="Індекс репліки з SRT")
    speaker: str = Field(description="Імʼя мовця, або NARRATOR / UNKNOWN")
    speaker_gender: Gender = "unknown"
    addressees: list[str] = Field(default_factory=list)
    addressee_number: Literal["sg", "pl", "none"] = "none"
    formality: Formality = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str | None = Field(
        default=None, description="Чому саме цей мовець — коротко"
    )


class SpeakerMap(BaseModel):
    cues: list[CueSpeaker] = Field(default_factory=list)

    def by_index(self) -> dict[int, CueSpeaker]:
        return {item.cue: item for item in self.cues}


# ---------- прохід bible ----------


class VoiceCard(BaseModel):
    character: str
    # не `register`: це імʼя затінює атрибут pydantic.BaseModel
    speech_register: str = Field(description="Висока / нейтральна / розмовна / арго")
    quirks: list[str] = Field(default_factory=list)
    catchphrases: list[str] = Field(default_factory=list)
    swearing: Literal["none", "mild", "strong", "crude"] = "none"
    notes: str | None = None


class FormalityPair(BaseModel):
    speaker: str
    addressee: str
    form: Literal["ти", "Ви"]
    switches_at_cue: int | None = Field(
        default=None, description="Індекс репліки, з якої форма змінюється"
    )
    switches_to: Literal["ти", "Ви"] | None = None
    reason: str | None = None


class RecurringLine(BaseModel):
    en: str
    uk: str
    note: str | None = None


class ProfanityRung(BaseModel):
    en: str = Field(description="Англійський вираз")
    uk: str = Field(description="Український відповідник відповідної сили")
    strength: Literal["mild", "medium", "strong", "crude"] = "mild"


class AvoidEntry(BaseModel):
    avoid: str = Field(description="Русизм, калька або канцелярит")
    use: str = Field(description="Що вживати натомість")
    why: str | None = None


class Bible(BaseModel):
    """Біблія перекладу — стабільний контекст для всіх наступних викликів."""

    register_overall: str = ""
    era_notes: str | None = None
    voices: list[VoiceCard] = Field(default_factory=list)
    formality: list[FormalityPair] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    honorifics: list[GlossaryEntry] = Field(default_factory=list)
    recurring: list[RecurringLine] = Field(default_factory=list)
    profanity_ladder: list[ProfanityRung] = Field(default_factory=list)
    avoid: list[AvoidEntry] = Field(default_factory=list)
    open_questions: list[str] = Field(
        default_factory=list, description="Що треба вирішити людині"
    )

    def formality_for(self, speaker: str, addressee: str, cue: int) -> Formality:
        """Форма звертання для пари на конкретній репліці, з урахуванням переходу."""
        for pair in self.formality:
            if (
                pair.speaker.casefold() == speaker.casefold()
                and pair.addressee.casefold() == addressee.casefold()
            ):
                if (
                    pair.switches_at_cue is not None
                    and pair.switches_to is not None
                    and cue >= pair.switches_at_cue
                ):
                    return pair.switches_to
                return pair.form
        return "unknown"


# ---------- прохід translate ----------


class CueTranslation(BaseModel):
    cue: int
    uk: str
    flags: list[TranslationFlag] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    note: str | None = Field(
        default=None, description="Коротко: що втрачено або чому саме так"
    )


class ChunkTranslation(BaseModel):
    cues: list[CueTranslation] = Field(default_factory=list)


# ---------- прохід edit ----------


class CueEdit(BaseModel):
    cue: int
    uk: str = Field(description="Виправлений текст; якщо правок немає — той самий")
    changed: bool = False
    reasons: list[str] = Field(
        default_factory=list, description="Пункти чекліста, за якими було виправлено"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ChunkEdit(BaseModel):
    cues: list[CueEdit] = Field(default_factory=list)


# ---------- прохід fit ----------


class CueFit(BaseModel):
    cue: int
    uk: str = Field(description="Стиснутий текст без втрати змісту")
    dropped: str | None = Field(
        default=None, description="Що саме прибрано, якщо це щось значуще"
    )


class ChunkFit(BaseModel):
    cues: list[CueFit] = Field(default_factory=list)
