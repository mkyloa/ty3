"""Лінтери: технічні норми та українські перевірки."""

from __future__ import annotations

from pathlib import Path

import pytest

from ty3.checks.base import Severity
from ty3.checks.subtitle_lint import (
    check_all,
    check_cue,
    check_timings_preserved,
    count_tags,
)
from ty3.checks.uk_lint import (
    AvoidRule,
    check_formality,
    check_glossary,
    check_repeated_lines,
    check_russianisms,
    check_uk_all,
    load_russianisms,
    same_word,
)
from ty3.config import SubtitleConfig
from ty3.model.schemas import Bible, CueSpeaker, FormalityPair, GlossaryEntry
from ty3.srtio.parse import parse_srt
from ty3.srtio.render import render_srt

CFG = SubtitleConfig()
FIXTURE = Path(__file__).parent / "fixtures" / "sample.en.srt"


def make_cue(text: str, start_ms: int = 0, end_ms: int = 4000, index: int = 1):
    [cue] = parse_srt(f"{index}\n{_ts(start_ms)} --> {_ts(end_ms)}\n{text}\n")
    return cue


def _ts(ms: int) -> str:
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def codes(issues) -> set[str]:
    return {issue.code for issue in issues}


# ================= технічні норми =================


def test_clean_cue_has_no_issues():
    cue = make_cue("Fine.", 0, 3000)
    assert check_cue(cue, ["Гаразд."], CFG) == []


def test_cps_violation_detected():
    cue = make_cue("Short source.", 0, 1000)
    issues = check_cue(cue, ["Дуже довгий український рядок на одну секунду."], CFG)
    assert "CPS_HIGH" in codes(issues)
    assert all(issue.severity is Severity.ERROR for issue in issues if issue.code == "CPS_HIGH")


def test_line_too_long_detected():
    cue = make_cue("Source.", 0, 20_000)
    long_line = "я" * 50
    issues = check_cue(cue, [long_line], CFG)
    assert "LINE_TOO_LONG" in codes(issues)


def test_too_many_lines_detected():
    cue = make_cue("Source.", 0, 20_000)
    issues = check_cue(cue, ["Раз", "Два", "Три"], CFG)
    assert "TOO_MANY_LINES" in codes(issues)


def test_empty_translation_is_an_error():
    issues = check_cue(make_cue("Something."), [""], CFG)
    assert "EMPTY_TRANSLATION" in codes(issues)


def test_short_duration_is_a_warning_not_an_error():
    cue = make_cue("Hi.", 0, 400)
    issues = [i for i in check_cue(cue, ["Ой."], CFG) if i.code == "DURATION_SHORT"]
    assert issues and issues[0].severity is Severity.WARNING


def test_long_duration_is_only_informational():
    cue = make_cue("Hello there my friend.", 0, 9000)
    issues = [i for i in check_cue(cue, ["Привіт, друже."], CFG) if i.code == "DURATION_LONG"]
    assert issues and issues[0].severity is Severity.INFO


def test_tag_mismatch_detected():
    cue = make_cue("<i>Italic source.</i>", 0, 5000)
    assert "TAG_MISMATCH" in codes(check_cue(cue, ["Курсив зник."], CFG))


def test_matching_tags_produce_no_mismatch():
    cue = make_cue("<i>Italic source.</i>", 0, 5000)
    assert "TAG_MISMATCH" not in codes(check_cue(cue, ["<i>Курсив на місці.</i>"], CFG))


def test_font_attribute_change_is_tolerated():
    cue = make_cue('<font color="#FFFFFF">Text.</font>', 0, 5000)
    assert "TAG_MISMATCH" not in codes(
        check_cue(cue, ['<font color="#EEEEEE">Текст.</font>'], CFG)
    )


def test_latin_leftover_flagged():
    cue = make_cue("The hospital.", 0, 5000)
    assert "LATIN_LEFTOVER" in codes(check_cue(cue, ["Лікарня hospital."], CFG))


def test_latin_allowed_when_whitelisted():
    cue = make_cue("The FBI.", 0, 5000)
    issues = check_cue(cue, ["Агенти FBI."], CFG, allowed_latin={"FBI"})
    assert "LATIN_LEFTOVER" not in codes(issues)


def test_forced_signs_may_keep_latin():
    cue = make_cue(r"{\an8}ST. AUGUSTINE GENERAL", 0, 5000)
    assert "LATIN_LEFTOVER" not in codes(check_cue(cue, [r"{\an8}ST. AUGUSTINE"], CFG))


def test_count_tags_finds_both_syntaxes():
    assert count_tags(r"{\an8}<i>text</i>") == [r"{\an8}", "<i>", "</i>"]


def test_check_all_reports_untranslated_cue():
    cues = parse_srt(FIXTURE.read_text(encoding="utf-8"))
    issues = check_all(cues, {}, CFG)
    assert "NOT_TRANSLATED" in codes(issues)


# ================= тайминги =================


def test_timings_preserved_on_real_render():
    source = FIXTURE.read_text(encoding="utf-8")
    cues = parse_srt(source)
    output = render_srt(cues, {cue.index: f"Переклад {cue.index}." for cue in cues}, CFG)
    assert check_timings_preserved(source, output) == []


def test_changed_timing_is_caught():
    source = "1\n00:00:01,000 --> 00:00:02,000\nA.\n"
    broken = "1\n00:00:01,000 --> 00:00:02,500\nА.\n"
    issues = check_timings_preserved(source, broken)
    assert codes(issues) == {"TIMING_CHANGED"}
    assert issues[0].severity is Severity.ERROR


def test_dropped_cue_is_caught():
    source = "1\n00:00:01,000 --> 00:00:02,000\nA.\n\n2\n00:00:03,000 --> 00:00:04,000\nB.\n"
    truncated = "1\n00:00:01,000 --> 00:00:02,000\nА.\n"
    assert codes(check_timings_preserved(source, truncated)) == {"CUE_COUNT_CHANGED"}


def test_crlf_difference_does_not_count_as_a_change():
    source = "1\r\n00:00:01,000 --> 00:00:02,000\r\nA.\r\n"
    output = "1\n00:00:01,000 --> 00:00:02,000\nА.\n"
    assert check_timings_preserved(source, output) == []


# ================= русизми =================


def test_russianisms_file_loads():
    rules = load_russianisms()
    assert len(rules) > 30
    assert any(rule.avoid == "співпадати" for rule in rules)
    assert all(rule.use for rule in rules)


def test_russianism_detected_with_suggestion():
    rules = [AvoidRule(avoid="співпадати", use="збігатися")]
    issues = check_russianisms(1, "Наші плани співпадати не мусять.", rules)
    assert len(issues) == 1
    assert "збігатися" in issues[0].message


def test_multiword_russianism_detected():
    rules = load_russianisms()
    issues = check_russianisms(1, "На протязі дня нічого не сталося.", rules)
    assert any("протягом" in issue.message for issue in issues)


def test_russianism_respects_word_boundaries():
    rules = [AvoidRule(avoid="ждати", use="чекати")]
    assert check_russianisms(1, "Він нічого не очікував.", rules) == []


def test_russian_profanity_is_an_error_not_a_warning():
    rules = load_russianisms()
    issues = check_russianisms(1, "Це якийсь пиздець.", rules)
    assert issues and issues[0].severity is Severity.ERROR


def test_clean_ukrainian_passes():
    assert check_russianisms(1, "Наші плани збігаються протягом тижня.", load_russianisms()) == []


# ================= ти / Ви =================


def _speaker(**kwargs) -> CueSpeaker:
    defaults = dict(
        cue=1, speaker="Elena", speaker_gender="f", addressees=["Marcus"],
        addressee_number="sg", formality="ти", confidence=0.9,
    )
    return CueSpeaker(**{**defaults, "cue": kwargs.pop("cue", 1), **kwargs})


def test_formality_matches_matrix():
    assert check_formality(1, "Ти знову спізнився.", _speaker(formality="ти"), None) == []


def test_polite_form_where_matrix_says_informal():
    issues = check_formality(1, "Ви знову спізнилися.", _speaker(formality="ти"), None)
    assert codes(issues) == {"FORMALITY_MISMATCH"}
    assert issues[0].severity is Severity.ERROR


def test_informal_form_where_matrix_says_polite():
    issues = check_formality(1, "Ти знову спізнився.", _speaker(formality="Ви"), None)
    assert codes(issues) == {"FORMALITY_MISMATCH"}


def test_mixed_forms_in_one_cue_is_an_error():
    issues = check_formality(1, "Ти ж казали Вам.", _speaker(formality="unknown"), None)
    assert "FORMALITY_MIXED" in codes(issues)


def test_plural_addressee_allows_vy():
    speaker = _speaker(formality="ти", addressee_number="pl", addressees=["Marcus", "Elena"])
    assert check_formality(1, "Ви всі спізнилися.", speaker, None) == []


def test_no_pronouns_means_nothing_to_check():
    assert check_formality(1, "Гаразд.", _speaker(formality="Ви"), None) == []


def test_formality_falls_back_to_bible_matrix():
    bible = Bible(
        formality=[FormalityPair(speaker="Elena", addressee="Marcus", form="Ви")]
    )
    speaker = _speaker(formality="unknown")
    assert codes(check_formality(1, "Ти спізнився.", speaker, bible)) == {
        "FORMALITY_MISMATCH"
    }


def test_bible_matrix_switch_point_is_respected():
    bible = Bible(
        formality=[
            FormalityPair(
                speaker="Elena", addressee="Marcus", form="Ви",
                switches_at_cue=100, switches_to="ти",
            )
        ]
    )
    speaker_before = _speaker(cue=50, formality="unknown")
    speaker_after = _speaker(cue=150, formality="unknown")
    assert codes(check_formality(50, "Ти спізнився.", speaker_before, bible)) == {
        "FORMALITY_MISMATCH"
    }
    assert check_formality(150, "Ти спізнився.", speaker_after, bible) == []


def test_unknown_speaker_is_not_second_guessed():
    assert check_formality(1, "Ти спізнився.", None, None) == []


# ================= глосарій =================


def test_glossary_term_present_passes():
    cue = make_cue("Dr. Hollis wants the forms.")
    entries = [GlossaryEntry(en="Hollis", uk="Голліс", kind="person")]
    assert check_glossary(cue, "Доктор Голліс хоче формуляри.", entries) == []


def test_glossary_term_missing_is_an_error():
    cue = make_cue("Dr. Hollis wants the forms.")
    entries = [GlossaryEntry(en="Hollis", uk="Голліс", kind="person")]
    issues = check_glossary(cue, "Лікарка хоче формуляри.", entries)
    assert codes(issues) == {"GLOSSARY_MISS"}


def test_glossary_tolerates_ukrainian_inflection():
    cue = make_cue("Hollis signed it.")
    entries = [GlossaryEntry(en="Hollis", uk="Голліс", kind="person")]
    # «Голліс» у родовому відмінку — «Голліса»; це той самий термін
    assert check_glossary(cue, "Це підписала Голліса.", entries) == []


def test_glossary_ignores_terms_absent_from_source():
    cue = make_cue("Nothing relevant here.")
    entries = [GlossaryEntry(en="Hollis", uk="Голліс", kind="person")]
    assert check_glossary(cue, "Нічого важливого.", entries) == []


@pytest.mark.parametrize(
    "nominative,inflected",
    [
        ("Маркус", "Маркуса"),
        ("Маркус", "Маркусе"),
        ("Голліс", "Голліса"),
        ("обхід", "обходу"),  # спостережено на живому прогоні: і↔о
        ("ніч", "ночі"),
        ("дім", "дому"),
        ("кінь", "коня"),
        ("папір", "паперу"),
        ("Елена", "Елено"),
    ],
)
def test_same_word_across_inflection_and_alternation(nominative, inflected):
    """Чергування і↔о/е в закритому складі ламало префіксне порівняння основ."""
    assert same_word(nominative, inflected)


@pytest.mark.parametrize(
    "first,second",
    [
        ("обхід", "візит"),
        ("Маркус", "Маргарита"),
        ("Голліс", "Гоголь"),
        ("форми", "фарба"),
    ],
)
def test_same_word_does_not_conflate_different_words(first, second):
    assert not same_word(first, second)


def test_known_limitation_words_sharing_a_long_prefix_collide():
    """Задокументована межа підходу, а не помилка.

    Префіксна евристика без словника не розділить «палата» і «палац». Обрано
    свідомо: наслідок колізії — пропущена перевірка глосарію, тоді як зворотна
    помилка дала б ERROR і зайвий виклик моделі на кожній такій реплиці.
    """
    assert same_word("палата", "палац")


def test_glossary_tolerates_vowel_alternation():
    """Регресія з живого прогону: «до обходу» вважалося відсутністю «обхід»."""
    cue = make_cue("Dr. Hollis wants the forms before rounds.")
    entries = [GlossaryEntry(en="rounds", uk="обхід", kind="term")]
    assert check_glossary(cue, "Докторка Голліс хоче форми до обходу.", entries) == []


def test_glossary_still_catches_a_genuinely_missing_term():
    """Поблажливість до словозміни не має ховати справжній пропуск."""
    cue = make_cue("Dr. Hollis wants the forms before rounds.")
    entries = [GlossaryEntry(en="rounds", uk="обхід", kind="term")]
    issues = check_glossary(cue, "Докторка Голліс хоче форми до візиту.", entries)
    assert codes(issues) == {"GLOSSARY_MISS"}


# ================= консистентність =================


def test_same_line_translated_differently_is_flagged():
    cues = [make_cue("Are you serious?", index=1), make_cue("Are you serious?", index=2)]
    translations = {1: "Ти серйозно?", 2: "Ви що, жартуєте?"}
    issues = check_repeated_lines(cues, translations)
    assert codes(issues) == {"INCONSISTENT_REPEAT"}


def test_same_line_translated_the_same_passes():
    cues = [make_cue("Are you serious?", index=1), make_cue("Are you serious?", index=2)]
    assert check_repeated_lines(cues, {1: "Ти серйозно?", 2: "Ти серйозно?"}) == []


def test_different_speakers_may_differ():
    cues = [make_cue("Fine by me.", index=1), make_cue("Fine by me.", index=2)]
    speakers = {
        1: CueSpeaker(cue=1, speaker="Elena"),
        2: CueSpeaker(cue=2, speaker="Marcus"),
    }
    assert check_repeated_lines(cues, {1: "Мене влаштовує.", 2: "Я не проти."}, speakers) == []


def test_very_short_lines_are_not_compared():
    cues = [make_cue("Ok.", index=1), make_cue("Ok.", index=2)]
    assert check_repeated_lines(cues, {1: "Гаразд.", 2: "Добре."}) == []


# ================= усе разом =================


def test_check_uk_all_combines_checks():
    cues = [make_cue("You are late.", index=1, end_ms=5000)]
    bible = Bible(
        glossary=[GlossaryEntry(en="late", uk="спізнився")],
        formality=[FormalityPair(speaker="Marcus", addressee="Elena", form="Ви")],
    )
    speakers = {
        1: CueSpeaker(
            cue=1, speaker="Marcus", addressees=["Elena"],
            addressee_number="sg", formality="Ви",
        )
    }
    issues = check_uk_all(
        cues, {1: "Ти співпадати спізнився."}, bible=bible, speakers=speakers
    )
    assert {"RUSSIANISM", "FORMALITY_MISMATCH"} <= codes(issues)


def test_check_uk_all_on_clean_text_is_quiet():
    cues = [make_cue("You are late.", index=1, end_ms=5000)]
    speakers = {
        1: CueSpeaker(
            cue=1, speaker="Marcus", addressees=["Elena"],
            addressee_number="sg", formality="ти",
        )
    }
    assert check_uk_all(cues, {1: "Ти спізнилася."}, speakers=speakers) == []
