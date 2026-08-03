"""Детерміновані частини проходів: розбір імені файлу, серіалізація, звіт."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ty3.checks.base import Issue, Severity
from ty3.config import Config, SubtitleConfig
from ty3.model.schemas import (
    Bible,
    Character,
    CueSpeaker,
    CueTranslation,
    Dossier,
    FormalityPair,
    GlossaryEntry,
    ProfanityRung,
    SpeakerMap,
    VoiceCard,
)
from ty3.passes.bible import _merge_official_names, _observed_pairs, bible_to_markdown
from ty3.passes.consistency import allowed_latin_from
from ty3.passes.context import bible_context, chunk_body, cue_block, dossier_context, transcript
from ty3.passes.report import build_report
from ty3.passes.research import guess_title_from_filename
from ty3.passes.speakers import load_speakers
from ty3.passes.translate import TranslationResult
from ty3.pipeline import slugify
from ty3.srtio.parse import parse_srt

FIXTURE = Path(__file__).parent / "fixtures" / "sample.en.srt"
CFG = Config()


@pytest.fixture(scope="module")
def cues():
    return parse_srt(FIXTURE.read_text(encoding="utf-8"))


# ================= розбір імені файлу релізу =================


@pytest.mark.parametrize(
    "filename,expected_title,expected_year",
    [
        ("The.Substance.2024.1080p.WEB-DL.x264-GROUP.en.srt", "The Substance", 2024),
        ("Sinners (2025) [1080p] BluRay x265.srt", "Sinners", 2025),
        ("Nosferatu.2024.2160p.UHD.BluRay.REMUX.DV.HDR.srt", "Nosferatu", 2024),
        ("The_Brutalist_2024_WEBRip_AAC5.1.srt", "The Brutalist", 2024),
        ("Severance.S02E05.1080p.ATVP.WEB-DL.srt", "Severance", None),
        ("Anora.srt", "Anora", None),
    ],
)
def test_title_and_year_from_release_name(filename, expected_title, expected_year):
    title, year = guess_title_from_filename(filename)
    assert title == expected_title
    assert year == expected_year


def test_release_group_suffix_is_stripped():
    title, _ = guess_title_from_filename("Conclave.2024.1080p.WEB.H264-SLOT.srt")
    assert title == "Conclave"


def test_sdh_and_language_tags_are_stripped():
    title, year = guess_title_from_filename("Dune.Part.Two.2024.1080p.eng.sdh.forced.srt")
    assert title == "Dune Part Two" and year == 2024


def test_full_path_works_not_just_basename():
    title, year = guess_title_from_filename("/movies/Heretic.2024.WEB-DL.srt")
    assert (title, year) == ("Heretic", 2024)


# ================= slug для теки =================


@pytest.mark.parametrize(
    "value,expected",
    [
        ("The Substance", "the-substance"),
        ("Dune: Part Two", "dune-part-two"),
        ("Мій фільм", "мій-фільм"),
        ("!!!", "film"),
    ],
)
def test_slugify(value, expected):
    assert slugify(value) == expected


# ================= серіалізація контексту =================


def test_transcript_includes_indices_and_labels(cues):
    text = transcript(cues)
    assert "3 MARCUS: You're late again, Elena." in text
    assert "20 NURSE (V.O.): Room four is asking" in text


def test_transcript_marks_non_dialogue_kinds(cues):
    text = transcript(cues)
    assert "[sdh] [RAIN DRUMMING ON GLASS]" in text
    assert "[forced] ST. AUGUSTINE GENERAL" in text


def test_transcript_skips_empty_cues():
    [cue] = parse_srt("1\n00:00:01,000 --> 00:00:02,000\n \n")
    assert transcript([cue]) == ""


def test_cue_block_carries_gender_and_formality(cues):
    speaker = CueSpeaker(
        cue=3, speaker="Marcus", speaker_gender="m", addressees=["Elena"],
        addressee_number="sg", formality="ти",
    )
    block = cue_block(cues[2], CFG.subtitle, speaker)
    assert "WHO: Marcus (m) → Elena (sg), звертання: ти" in block
    assert "budget=" in block
    assert "EN: You're late again, Elena." in block


def test_cue_block_warns_about_two_speakers(cues):
    assert "два мовці" in cue_block(cues[3], CFG.subtitle)


def test_cue_block_context_form_is_compact(cues):
    block = cue_block(cues[0], CFG.subtitle, context_only=True, uk_text="Раніше…")
    assert "КОНТЕКСТ" in block and "UA: Раніше…" in block
    assert "budget=" not in block


def test_chunk_body_marks_translation_boundaries(cues):
    body = chunk_body(
        cues[3:6], CFG.subtitle, {}, before=cues[:3], after=cues[6:8],
        translated={1: "Раніше, у нічну зміну…"},
    )
    assert "--- ПЕРЕКЛАДАТИ ЗВІДСИ ---" in body
    assert "--- ДАЛІ ЛИШЕ КОНТЕКСТ ---" in body
    assert "UA: Раніше, у нічну зміну…" in body


def test_chunk_body_without_context_has_no_markers(cues):
    body = chunk_body(cues[:3], CFG.subtitle, {})
    assert "ПЕРЕКЛАДАТИ ЗВІДСИ" not in body


# ================= досьє та біблія у промпті =================


def _dossier() -> Dossier:
    return Dossier(
        title_en="Night Shift",
        title_uk="Нічна зміна",
        year=2024,
        director="A. Director",
        genre=["драма"],
        setting="сучасна лікарня",
        rating="R",
        synopsis="Двоє медиків у нічну зміну.",
        plot_beats=["Елена спізнюється", "Аварія"],
        characters=[
            Character(name_en="Elena", name_uk="Елена", gender="f", role="інтерн"),
            Character(name_en="Marcus", name_uk="Маркус", gender="m", role="ординатор"),
        ],
        identification_confidence=0.9,
    )


def test_dossier_context_is_readable_and_complete():
    text = dossier_context(_dossier())
    assert "Нічна зміна" in text
    assert "Рейтинг: R" in text
    assert "| Elena | Елена | f | інтерн |" in text


def test_dossier_gender_lookup_is_case_insensitive():
    assert _dossier().gender_of("elena") == "f"
    assert _dossier().gender_of("Marcus") == "m"
    assert _dossier().gender_of("Ніхто") == "unknown"


def _bible() -> Bible:
    return Bible(
        register_overall="розмовний, напружений",
        voices=[VoiceCard(character="Elena", speech_register="розмовний", swearing="mild")],
        formality=[
            FormalityPair(speaker="Elena", addressee="Marcus", form="Ви",
                          switches_at_cue=18, switches_to="ти", reason="сварка зблизила"),
        ],
        glossary=[GlossaryEntry(en="Hollis", uk="Голліс", kind="person")],
        honorifics=[GlossaryEntry(en="Doctor", uk="лікарю", kind="honorific")],
        profanity_ladder=[ProfanityRung(en="damn", uk="дідько", strength="mild")],
        open_questions=["Як писати назву лікарні?"],
    )


def test_bible_context_contains_matrix_with_switch():
    text = bible_context(_bible())
    assert "| Elena | Marcus | Ви | з #18 → ти |" in text
    assert "Hollis → **Голліс**" in text
    assert "damn → дідько" in text


def test_bible_markdown_adds_open_questions():
    text = bible_to_markdown(_bible())
    assert "Відкриті питання" in text
    assert "Як писати назву лікарні?" in text


def test_formality_lookup_honours_switch_point():
    bible = _bible()
    assert bible.formality_for("Elena", "Marcus", 5) == "Ви"
    assert bible.formality_for("Elena", "Marcus", 20) == "ти"
    assert bible.formality_for("Elena", "Nobody", 5) == "unknown"


# ================= допоміжне для біблії =================


def test_observed_pairs_skips_narrator_and_unknown():
    speaker_map = SpeakerMap(
        cues=[
            CueSpeaker(cue=1, speaker="NARRATOR", addressees=["Elena"]),
            CueSpeaker(cue=2, speaker="UNKNOWN", addressees=["Elena"]),
            CueSpeaker(cue=3, speaker="Elena", addressees=["Marcus"]),
            CueSpeaker(cue=4, speaker="Marcus", addressees=["Elena"]),
        ]
    )
    assert _observed_pairs(speaker_map) == [("Elena", "Marcus"), ("Marcus", "Elena")]


def test_observed_pairs_ignores_self_address():
    speaker_map = SpeakerMap(cues=[CueSpeaker(cue=1, speaker="Elena", addressees=["Elena"])])
    assert _observed_pairs(speaker_map) == []


def test_official_names_are_forced_into_glossary():
    bible = _merge_official_names(Bible(), _dossier())
    assert {entry.en for entry in bible.glossary} == {"Elena", "Marcus"}
    assert all("офіційне написання" in (entry.note or "") for entry in bible.glossary)


def test_official_names_do_not_duplicate_existing_entries():
    bible = Bible(glossary=[GlossaryEntry(en="Elena", uk="Олена", kind="person")])
    merged = _merge_official_names(bible, _dossier())
    elena = [entry for entry in merged.glossary if entry.en == "Elena"]
    assert len(elena) == 1
    assert elena[0].uk == "Олена"  # рішення моделі не перезаписується


def test_allowed_latin_collects_glossary_terms():
    bible = Bible(glossary=[GlossaryEntry(en="FBI", uk="FBI", kind="org")])
    assert "FBI" in allowed_latin_from(bible)


# ================= QA-звіт =================


def _result() -> TranslationResult:
    result = TranslationResult()
    result.apply(CueTranslation(cue=1, uk="Раніше, у нічну зміну…", confidence=0.95))
    result.apply(
        CueTranslation(
            cue=3, uk="Ти знову спізнилася, Елено.", confidence=0.5,
            flags=["gender_guessed"], note="стать виведена з діалогу",
        )
    )
    result.apply(
        CueTranslation(cue=7, uk="Задовгий рядок", confidence=0.9, flags=["too_long"])
    )
    result.edits[3] = ["кличний відмінок"]
    result.missing = [24]
    return result


def test_report_covers_statistics_and_problems(cues):
    issues = [
        Issue(7, "CPS_HIGH", Severity.ERROR, "22.1 симв./с перевищує норму 17"),
        Issue(3, "LATIN_LEFTOVER", Severity.WARNING, "латиниця в перекладі"),
    ]
    text = build_report(
        cues, _result(), issues, _dossier(), _bible(),
        {3: CueSpeaker(cue=3, speaker="Marcus", speaker_gender="unknown")},
        CFG, stats={"billed_calls": 8, "cached_calls": 2, "input_tokens": 100,
                    "output_tokens": 50, "cache_read_tokens": 900, "cost_usd": 1.25},
    )
    assert "# QA-звіт: Нічна зміна" in text
    assert "**Пропущено: 1**" in text
    assert "Помилок лінтера після виправлень: 1" in text
    assert "gender_guessed" in text
    assert "Низька впевненість" in text
    assert "CPS_HIGH" in text
    assert "$1.25" in text
    assert "Як писати назву лікарні?" in text  # відкрите питання з біблії
    assert "Персонажі з невизначеною статтю" in text
    assert "кличний відмінок" in text  # причини правок
    assert "| Hollis | Голліс |" in text


def test_report_is_quiet_when_nothing_is_wrong(cues):
    result = TranslationResult()
    result.apply(CueTranslation(cue=1, uk="Усе добре.", confidence=0.95))
    text = build_report(cues, result, [], _dossier(), Bible(), {}, CFG)
    assert "Помилок лінтера після виправлень: 0" in text
    assert "Прапорці перекладача" not in text
    assert "Низька впевненість" not in text


# ================= перенесення атрибуції мовців між прогонами =================


def test_speakers_round_trip_through_disk(tmp_path):
    """Атрибуція — звичайні дані, не привʼязані до моделі: можна порахувати раз
    (напр. на Opus) і підкласти в прогін з іншим --model/--backend, точнісінько
    як уже можна підкласти dossier.json чи bible.json."""
    original = SpeakerMap(
        cues=[
            CueSpeaker(
                cue=3, speaker="Marcus", speaker_gender="m", addressees=["Elena"],
                addressee_number="sg", formality="ти", confidence=0.97,
                evidence="явна мітка мовця",
            )
        ]
    )
    path = tmp_path / "speakers.json"
    path.write_text(
        json.dumps(original.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
    )

    loaded = load_speakers(path)
    assert loaded == original
