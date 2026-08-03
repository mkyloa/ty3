"""Парсер і рендер: головна гарантія — тайминги не рухаються ніколи."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ty3.config import SubtitleConfig
from ty3.srtio.parse import CueKind, SrtParseError, parse_srt, parse_timestamp
from ty3.srtio.render import (
    normalize_typography,
    render_bilingual,
    render_srt,
    wrap_text,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample.en.srt"
CFG = SubtitleConfig()


@pytest.fixture(scope="module")
def source() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cues(source: str):
    return parse_srt(source)


# ---------- таймкоди ----------


def test_parse_timestamp():
    assert parse_timestamp("00:00:02,000") == 2000
    assert parse_timestamp("01:02:03,456") == 3_723_456
    assert parse_timestamp("00:00:00.500") == 500  # точка теж трапляється
    assert parse_timestamp("00:00:01,5") == 1500  # неповні мілісекунди


def test_parse_timestamp_rejects_garbage():
    with pytest.raises(SrtParseError):
        parse_timestamp("зовсім не таймкод")


def test_cue_timings(cues):
    assert cues[0].start_ms == 2000
    assert cues[0].end_ms == 4120
    assert cues[0].duration_ms == 2120


# ---------- структура ----------


def test_all_cues_parsed(cues):
    assert len(cues) == 24
    assert [c.index for c in cues] == list(range(1, 25))


def test_speaker_label_extracted(cues):
    marcus = cues[2]
    assert marcus.speaker_hint == "MARCUS"
    assert marcus.text == "You're late again, Elena."


def test_speaker_label_with_note(cues):
    nurse = cues[19]
    assert nurse.speaker_hint == "NURSE"
    assert nurse.speaker_note == "V.O."
    assert nurse.text.startswith("Room four is asking")


def test_colon_in_normal_sentence_is_not_a_speaker_label():
    [cue] = parse_srt("1\n00:00:01,000 --> 00:00:02,000\nOne thing: it works.\n")
    assert cue.speaker_hint is None
    assert cue.text == "One thing: it works."


def test_cue_kinds(cues):
    kinds = {c.index: c.kind for c in cues}
    assert kinds[2] is CueKind.SDH  # [RAIN DRUMMING ON GLASS]
    assert kinds[13] is CueKind.SDH  # (SIGHS)
    assert kinds[24] is CueKind.SDH
    assert kinds[11] is CueKind.LYRICS  # ♪ ... ♪
    assert kinds[10] is CueKind.FORCED  # {\an8} — екранний напис
    assert kinds[3] is CueKind.DIALOGUE


def test_italics_are_separated_from_text(cues):
    first = cues[0]
    assert first.lines[0].wrap_tags == ["<i>"]
    assert first.lines[0].partial_markup is False
    assert first.text == "Previously, on the night shift..."


def test_dialogue_dashes(cues):
    two_speakers = cues[3]
    assert len(two_speakers.lines) == 2
    assert all(line.dialogue_dash for line in two_speakers.lines)
    assert two_speakers.lines[0].text == "I know."


def test_forced_cue_keeps_position_tag(cues):
    forced = cues[9]
    assert forced.lines[0].prefix == r"{\an8}"
    assert forced.text == "ST. AUGUSTINE GENERAL"
    assert forced.all_caps is True


def test_lyrics_music_marks_moved_to_prefix_suffix(cues):
    song = cues[10]
    assert song.lines[0].prefix.startswith("♪")
    assert song.lines[0].suffix.strip() == "♪"
    assert "♪" not in song.lines[0].text


def test_partial_markup_flagged():
    [cue] = parse_srt(
        "1\n00:00:01,000 --> 00:00:02,000\nHe said <i>maybe</i> tomorrow.\n"
    )
    assert cue.lines[0].partial_markup is True
    assert cue.lines[0].text == "He said maybe tomorrow."
    assert cue.lines[0].wrap_tags == ["<i>"]


# ---------- стійкість парсера ----------


def test_bom_and_crlf():
    raw = "﻿1\r\n00:00:01,000 --> 00:00:02,000\r\nHello.\r\n\r\n"
    [cue] = parse_srt(raw)
    assert cue.text == "Hello."


def test_missing_index_lines():
    raw = "00:00:01,000 --> 00:00:02,000\nFirst.\n\n00:00:03,000 --> 00:00:04,000\nSecond.\n"
    parsed = parse_srt(raw)
    assert [c.text for c in parsed] == ["First.", "Second."]


def test_non_sequential_indices_are_preserved():
    raw = "7\n00:00:01,000 --> 00:00:02,000\nSeven.\n\n99\n00:00:03,000 --> 00:00:04,000\nNinety-nine.\n"
    parsed = parse_srt(raw)
    assert [c.index for c in parsed] == [7, 99]


def test_timing_line_with_coordinates_is_preserved_verbatim():
    line = "00:00:01,000 --> 00:00:02,000  X1:100 X2:600 Y1:400 Y2:460"
    [cue] = parse_srt(f"1\n{line}\nPositioned.\n")
    assert cue.timing_line == line
    assert line in render_srt([cue], {1: "Розміщено."}, CFG)


def test_extra_blank_lines_between_blocks():
    raw = "1\n00:00:01,000 --> 00:00:02,000\nA.\n\n\n\n2\n00:00:03,000 --> 00:00:04,000\nB.\n"
    assert len(parse_srt(raw)) == 2


def test_empty_input_is_an_error():
    with pytest.raises(SrtParseError):
        parse_srt("   \n\n")


# ---------- бюджет символів ----------


def test_char_budget_limited_by_duration(cues):
    short = cues[5]  # "Fine." — 0,7 с
    assert short.duration_ms == 700
    assert short.char_budget(17.0, 2, 42) == 11  # int(0.7 * 17)


def test_char_budget_limited_by_screen_space(cues):
    long_cue = cues[6]  # 4.4 с — за часом влізло б 74 символи, але екран дає 84
    assert long_cue.char_budget(17.0, 2, 42) == 74
    assert long_cue.char_budget(30.0, 2, 42) == 84  # тут уже обмежує екран


def test_cps_computed(cues):
    assert cues[5].cps == pytest.approx(5 / 0.7, rel=1e-3)


# ---------- рендер ----------


def _timing_lines(text: str) -> list[str]:
    return re.findall(r"^.*-->.*$", text, re.MULTILINE)


def test_render_preserves_timings_and_indices(source, cues):
    translations = {cue.index: f"Переклад {cue.index}." for cue in cues}
    out = render_srt(cues, translations, CFG)
    assert _timing_lines(out) == _timing_lines(source)
    assert re.findall(r"^\d+$", out, re.MULTILINE) == re.findall(
        r"^\d+$", source, re.MULTILINE
    )


def test_render_without_translation_falls_back_to_source_text(cues):
    out = render_srt(cues, {}, CFG)
    assert "You're late again, Elena." in out  # мітка мовця знята, текст лишився


def test_render_restores_italics(cues):
    out = render_srt([cues[0]], {1: "Раніше, у нічну зміну…"}, CFG)
    assert "<i>Раніше, у нічну зміну…</i>" in out


def test_render_restores_position_tag(cues):
    out = render_srt([cues[9]], {10: "ЛІКАРНЯ СВЯТОГО АВГУСТИНА"}, CFG)
    assert r"{\an8}ЛІКАРНЯ СВЯТОГО АВГУСТИНА" in out


def test_render_restores_music_marks(cues):
    out = render_srt([cues[10]], {11: "а вуличні ліхтарі все горять"}, CFG)
    assert "♪ а вуличні ліхтарі все горять ♪" in out


def test_render_two_speaker_dialogue_keeps_both_dashes(cues):
    out = render_srt([cues[3]], {4: "Знаю.\nЦе не відповідь."}, CFG)
    body = out.split("\n", 2)[2]
    assert body.splitlines()[:2] == ["— Знаю.", "— Це не відповідь."]


def test_render_reinserts_translated_speaker_label(cues):
    out = render_srt([cues[2]], {3: "Ти знову спізнилася, Елено."}, CFG, labels={3: "МАРКУС"})
    assert "МАРКУС: Ти знову спізнилася, Елено." in out


def test_render_wraps_long_line_within_norm(cues):
    long_uk = "Автобус не приїхав, телефон сів, і я бігла під дощем."
    out = render_srt([cues[6]], {7: long_uk}, CFG)
    lines = out.split("\n", 2)[2].strip().splitlines()
    assert len(lines) == 2
    assert all(len(line) <= CFG.max_chars_per_line for line in lines)
    assert " ".join(lines) == long_uk


def test_wrap_overflow_is_kept_not_mangled():
    """Якщо текст фізично не влазить у 2×42, він лишається цілим —
    порушення піймає лінтер і воно піде у звіт, а не втратиться."""
    too_long = (
        "Слухай, автобус так і не приїхав, телефон сів, "
        "і я бігла одинадцять кварталів під дощем."
    )
    assert len(too_long) > CFG.max_lines * CFG.max_chars_per_line
    lines = wrap_text(too_long, CFG.max_chars_per_line, CFG.max_lines)
    assert len(lines) == 2
    assert " ".join(lines) == too_long  # нічого не викинуто
    assert max(len(line) for line in lines) > CFG.max_chars_per_line  # і це видно


def test_bilingual_has_english_underneath(cues):
    out = render_bilingual([cues[5]], {6: "Гаразд."}, CFG)
    assert "Гаразд." in out
    assert "<i>Fine.</i>" in out


# ---------- розрив рядків ----------


def test_wrap_short_text_stays_one_line():
    assert wrap_text("Коротко.", 42) == ["Коротко."]


def test_wrap_respects_max_chars():
    text = "Це досить довгий рядок, який просто не влізе в один рядок субтитрів."
    lines = wrap_text(text, 42)
    assert len(lines) == 2
    assert all(len(line) <= 42 for line in lines)
    assert " ".join(lines) == text


def test_wrap_does_not_orphan_preposition_at_line_end():
    text = "Вона поклала документи на стіл біля вікна і вийшла з кабінету."
    for line in wrap_text(text, 34):
        assert line.split()[-1].lower().strip(",.") not in {"на", "з", "біля", "і"}


def test_wrap_prefers_breaking_after_punctuation():
    text = "Я знаю, що ти запізнився, але це вже не має значення."
    lines = wrap_text(text, 30)
    assert lines[0].endswith(",")


def test_wrap_single_long_word_is_not_mangled():
    assert wrap_text("Нерозривнедовгеслово", 5) == ["Нерозривнедовгеслово"]


# ---------- типографіка ----------


def test_typography_quotes_and_apostrophe():
    assert normalize_typography('Він сказав "так" і пішов.', CFG) == (
        "Він сказав «так» і пішов."
    )
    assert normalize_typography("п'ять", CFG) == "п’ять"


def test_typography_ellipsis_and_dash():
    assert normalize_typography("Стоп... чекай", CFG) == "Стоп… чекай"
    assert normalize_typography("Це - точно", CFG) == "Це — точно"


def test_typography_removes_space_before_punctuation():
    assert normalize_typography("Справді ?  Так !", CFG) == "Справді? Так!"
