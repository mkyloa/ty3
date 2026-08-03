"""Витягання субтитрів із контейнерів.

Логіка вибору дорожки перевіряється юніт-тестами (вона чиста), а сам витяг —
на справжньому MKV, який генерується ffmpeg під час тесту. Мок тут нічого не
довів би: половина проблем саме в тому, що ffprobe віддає у теґах.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import io

from ty3.media import (
    BitmapSubtitleError,
    MediaError,
    SubtitleTrack,
    TrackSelectionCancelled,
    choose_track,
    describe_tracks,
    estimate_timeout_s,
    extract_subtitles,
    extract_track,
    is_container,
    media_duration_s,
    pick_track,
    probe_subtitle_tracks,
)
from ty3.srtio.parse import parse_file

FIXTURE = Path(__file__).parent / "fixtures" / "sample.en.srt"

has_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="потрібні ffmpeg і ffprobe",
)


def track(**kwargs) -> SubtitleTrack:
    defaults = dict(subtitle_index=0, stream_index=1, codec="subrip", language="eng")
    return SubtitleTrack(**{**defaults, **kwargs})


# ================= розпізнавання контейнера =================


@pytest.mark.parametrize(
    "name,expected",
    [
        ("movie.mkv", True),
        ("movie.MKV", True),
        ("movie.mp4", True),
        ("movie.webm", True),
        ("movie.srt", False),
        ("notes.md", False),
        ("movie.ass", False),
    ],
)
def test_is_container(name, expected):
    assert is_container(name) is expected


# ================= властивості дорожки =================


def test_bitmap_codecs_are_recognised():
    for codec in ("hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"):
        assert track(codec=codec).is_bitmap
        assert not track(codec=codec).is_text


def test_text_codecs_are_recognised():
    for codec in ("subrip", "ass", "mov_text", "webvtt"):
        assert track(codec=codec).is_text
        assert not track(codec=codec).is_bitmap


def test_forced_detected_from_disposition_and_from_title():
    assert track(forced=True).looks_forced
    assert track(title="English (Forced)").looks_forced
    assert track(title="Signs & Songs").looks_forced
    assert not track(title="English Full").looks_forced


def test_sdh_detected_from_disposition_and_from_title():
    assert track(hearing_impaired=True).looks_sdh
    assert track(title="English SDH").looks_sdh
    assert not track(title="English").looks_sdh


def test_english_recognised_across_code_variants():
    for code in ("eng", "en", "EN", "en-US", "English"):
        assert track(language=code).is_english
    assert not track(language="ukr").is_english


def test_label_is_informative():
    label = track(
        subtitle_index=2, codec="ass", language="eng", title="Full", default=True, frames=1500
    ).label()
    assert "s:2" in label and "ass" in label and "Full" in label
    assert "default" in label and "1500 реплик" in label


# ================= вибір дорожки =================


def test_full_english_beats_forced_english():
    """Найдорожча помилка: перекласти forced-дорожку (кілька десятків реплик)."""
    tracks = [
        track(subtitle_index=0, forced=True, title="Forced"),
        track(subtitle_index=1, title="Full"),
    ]
    assert pick_track(tracks).subtitle_index == 1


def test_english_beats_other_languages():
    tracks = [
        track(subtitle_index=0, language="ukr"),
        track(subtitle_index=1, language="fre"),
        track(subtitle_index=2, language="eng"),
    ]
    assert pick_track(tracks).subtitle_index == 2


def test_non_sdh_preferred_over_sdh():
    tracks = [
        track(subtitle_index=0, title="English SDH"),
        track(subtitle_index=1, title="English"),
    ]
    assert pick_track(tracks).subtitle_index == 1


def test_sdh_is_still_used_when_it_is_the_only_english_track():
    tracks = [
        track(subtitle_index=0, language="ukr"),
        track(subtitle_index=1, title="English SDH"),
    ]
    assert pick_track(tracks).subtitle_index == 1


def test_larger_track_preferred_when_frames_are_known():
    tracks = [
        track(subtitle_index=0, frames=60),
        track(subtitle_index=1, frames=1500),
    ]
    assert pick_track(tracks).subtitle_index == 1


def test_undefined_language_is_a_fallback_not_a_rejection():
    tracks = [track(subtitle_index=0, language="ukr"), track(subtitle_index=1, language=None)]
    assert pick_track(tracks).subtitle_index == 1


def test_text_track_wins_over_bitmap_even_if_bitmap_looks_better():
    tracks = [
        track(subtitle_index=0, codec="hdmv_pgs_subtitle", default=True, frames=1500),
        track(subtitle_index=1, codec="subrip"),
    ]
    assert pick_track(tracks).subtitle_index == 1


def test_bitmap_only_container_raises_with_an_explanation():
    tracks = [
        track(subtitle_index=0, codec="hdmv_pgs_subtitle"),
        track(subtitle_index=1, codec="dvd_subtitle", language="ukr"),
    ]
    with pytest.raises(BitmapSubtitleError) as caught:
        pick_track(tracks)
    message = str(caught.value)
    assert "OCR" in message
    assert "hdmv_pgs_subtitle" in message  # видно, що саме знайдено


def test_missing_requested_language_lists_what_is_available():
    tracks = [track(subtitle_index=0, language="ukr"), track(subtitle_index=1, language="fre")]
    with pytest.raises(MediaError) as caught:
        pick_track(tracks, language="eng")
    assert "--track" in str(caught.value)


def test_forced_can_be_requested_explicitly():
    tracks = [
        track(subtitle_index=0, forced=True, title="Forced"),
        track(subtitle_index=1, language="ukr"),
    ]
    assert pick_track(tracks, allow_forced=True).subtitle_index == 0


def test_no_subtitle_tracks_at_all():
    with pytest.raises(MediaError, match="немає субтитрових дорожок"):
        pick_track([])


def test_subrip_preferred_over_ass_when_otherwise_equal():
    tracks = [track(subtitle_index=0, codec="ass"), track(subtitle_index=1, codec="subrip")]
    assert pick_track(tracks).subtitle_index == 1


# ================= ліміт часу на витягання =================


def test_timeout_scales_with_file_size(tmp_path):
    """Регресія з реального прогону: 7.6 ГБ на диску 16 МБ/с не влізли в 600 с."""
    small = tmp_path / "small.mkv"
    small.write_bytes(b"x" * (10 * 1024 * 1024))  # 10 МБ
    big = tmp_path / "big.mkv"
    big.write_bytes(b"x" * (200 * 1024 * 1024))  # 200 МБ

    assert estimate_timeout_s(big) > estimate_timeout_s(small)


def sparse_file(path: Path, size_bytes: int) -> Path:
    """Файл заявленого розміру, який не займає місця на диску."""
    with path.open("wb") as handle:
        handle.truncate(size_bytes)
    return path


def test_timeout_for_a_bluray_sized_file_exceeds_the_old_constant(tmp_path):
    """Регресія: 7.6 ГБ на диску 16 МБ/с не влізли у фіксовані 600 с."""
    bluray = sparse_file(tmp_path / "bluray.mkv", int(7.6 * 1024**3))
    computed = estimate_timeout_s(bluray)
    assert computed > 600, "ліміт мусить перевищувати колишню константу"
    assert computed < 7200


def test_timeout_has_a_floor_for_tiny_files(tmp_path):
    tiny = tmp_path / "tiny.mkv"
    tiny.write_bytes(b"x" * 1024)
    assert estimate_timeout_s(tiny) >= 300  # дрібний файл усе одно має запас


def test_timeout_is_capped(tmp_path):
    """Навіть на дуже великому файлі ліміт не стає безмежним."""
    huge = sparse_file(tmp_path / "huge.mkv", 200 * 1024**3)  # 200 ГБ
    assert estimate_timeout_s(huge, cap_s=1800) == 1800


def test_timeout_survives_a_missing_file():
    """Відсутній файл не має валити розрахунок — його зловить probe."""
    assert estimate_timeout_s("/nonexistent/movie.mkv") > 0


def test_slow_media_assumption_makes_the_limit_larger(tmp_path):
    target = tmp_path / "m.mkv"
    target.write_bytes(b"x" * (100 * 1024 * 1024))
    assert estimate_timeout_s(target, floor_mb_s=2.0) > estimate_timeout_s(
        target, floor_mb_s=20.0
    )


@has_ffmpeg
def test_media_duration_is_read(mkv):
    duration = media_duration_s(mkv)
    assert duration is not None
    assert 85 < duration < 95  # фікстура на 90 с


def test_media_duration_of_a_non_media_file_is_none(tmp_path):
    fake = tmp_path / "fake.mkv"
    fake.write_text("не відео", encoding="utf-8")
    assert media_duration_s(fake) is None


# ================= вибір вручну =================


class FakeTty(io.StringIO):
    """Потік, який вдає термінал, — інакше choose_track відступить до автовибору."""

    def isatty(self) -> bool:
        return True


def three_tracks() -> list[SubtitleTrack]:
    return [
        track(subtitle_index=0, title="English Full", default=True),
        track(subtitle_index=1, title="English Forced", forced=True),
        track(subtitle_index=2, language="ukr", title="Українські"),
    ]


def collect():
    lines: list[str] = []
    return lines, lines.append


def test_empty_answer_takes_the_recommended_track():
    lines, say = collect()
    chosen = choose_track(three_tracks(), input_stream=FakeTty("\n"), say=say)
    assert chosen.subtitle_index == 0
    assert any("рекомендовано" in line for line in lines)


def test_explicit_number_is_honoured():
    _lines, say = collect()
    chosen = choose_track(three_tracks(), input_stream=FakeTty("2\n"), say=say)
    assert chosen.subtitle_index == 2


def test_prefixed_form_is_accepted():
    """Список показує «s:2», тож користувач цілком може ввести саме так."""
    _lines, say = collect()
    chosen = choose_track(three_tracks(), input_stream=FakeTty("s:1\n"), say=say)
    assert chosen.subtitle_index == 1


def test_forced_track_can_be_chosen_manually_even_though_auto_avoids_it():
    _lines, say = collect()
    chosen = choose_track(three_tracks(), input_stream=FakeTty("1\n"), say=say)
    assert chosen.looks_forced  # ручний вибір бʼє евристику


def test_invalid_number_reprompts_then_succeeds():
    lines, say = collect()
    chosen = choose_track(three_tracks(), input_stream=FakeTty("9\nabc\n2\n"), say=say)
    assert chosen.subtitle_index == 2
    assert sum("Немає дорожки" in line for line in lines) == 2


def test_q_cancels():
    _lines, say = collect()
    with pytest.raises(TrackSelectionCancelled):
        choose_track(three_tracks(), input_stream=FakeTty("q\n"), say=say)


def test_eof_falls_back_to_recommended_instead_of_hanging():
    lines, say = collect()
    chosen = choose_track(three_tracks(), input_stream=FakeTty(""), say=say)
    assert chosen.subtitle_index == 0
    assert any("Ввід закінчився" in line for line in lines)


def test_non_tty_stdin_does_not_prompt():
    """Прогін у фоні чи в CI не має ставати на паузу через вибір дорожки."""
    lines, say = collect()
    chosen = choose_track(three_tracks(), input_stream=io.StringIO("2\n"), say=say)
    assert chosen.subtitle_index == 2  # переданий потік усе одно читається


def test_bitmap_track_cannot_be_chosen():
    """Растрова дорожка є у списку, але вибрати її не можна — з поясненням."""
    tracks = [
        track(subtitle_index=0, codec="hdmv_pgs_subtitle"),
        track(subtitle_index=1, codec="subrip", title="English Full"),
        track(subtitle_index=2, codec="subrip", language="ukr"),
    ]
    lines, say = collect()
    chosen = choose_track(tracks, input_stream=FakeTty("0\n2\n"), say=say)
    assert chosen.subtitle_index == 2
    assert any("растрова" in line for line in lines)
    assert any("потрібен OCR" in line for line in lines)  # позначена і в списку


def test_single_text_track_is_taken_without_asking():
    tracks = [track(subtitle_index=0), track(subtitle_index=1, codec="hdmv_pgs_subtitle")]
    lines, say = collect()
    chosen = choose_track(tracks, input_stream=FakeTty(""), say=say)
    assert chosen.subtitle_index == 0
    assert any("Текстова дорожка одна" in line for line in lines)


def test_repeated_garbage_eventually_falls_back():
    lines, say = collect()
    chosen = choose_track(
        three_tracks(), input_stream=FakeTty("x\n" * 20), say=say, max_attempts=3
    )
    assert chosen.subtitle_index == 0
    assert any("Забагато невдалих спроб" in line for line in lines)


def test_bitmap_only_still_raises_before_prompting():
    tracks = [track(subtitle_index=0, codec="dvd_subtitle")]
    with pytest.raises(BitmapSubtitleError):
        choose_track(tracks, input_stream=FakeTty("0\n"), say=lambda _m: None)


def test_all_tracks_are_listed_with_selectable_numbers():
    lines, say = collect()
    choose_track(three_tracks(), input_stream=FakeTty("\n"), say=say)
    listing = "\n".join(lines)
    assert "[0]" in listing and "[1]" in listing and "[2]" in listing


# ================= справжній контейнер =================


@pytest.fixture(scope="module")
def mkv(tmp_path_factory) -> Path:
    """Збирає MKV із трьома дорожками: eng full, eng forced, ukr."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("немає ffmpeg")
    work = tmp_path_factory.mktemp("mkv")

    forced = work / "forced.srt"
    forced.write_text(
        "1\n00:00:36,500 --> 00:00:39,700\n[Sings in French]\n\n"
        "2\n00:00:40,000 --> 00:00:41,000\nSIGN: EXIT\n",
        encoding="utf-8",
    )
    ukrainian = work / "ukr.srt"
    ukrainian.write_text(
        "1\n00:00:15,100 --> 00:00:17,600\nТи знову спізнилася.\n", encoding="utf-8"
    )

    target = work / "movie.mkv"
    command = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=192x108:d=90:r=5",
        "-i", str(FIXTURE), "-i", str(forced), "-i", str(ukrainian),
        "-map", "0:v", "-map", "1", "-map", "2", "-map", "3",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "40", "-c:s", "srt",
        "-metadata:s:s:0", "language=eng", "-metadata:s:s:0", "title=English Full",
        "-disposition:s:0", "default",
        "-metadata:s:s:1", "language=eng", "-metadata:s:s:1", "title=English Forced",
        "-disposition:s:1", "forced",
        "-metadata:s:s:2", "language=ukr",
        str(target),
    ]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if done.returncode != 0:
        pytest.skip(f"ffmpeg не зібрав фікстуру: {done.stderr[:200]}")
    return target


@has_ffmpeg
def test_probe_finds_all_three_tracks(mkv):
    tracks = probe_subtitle_tracks(mkv)
    assert len(tracks) == 3
    assert [t.subtitle_index for t in tracks] == [0, 1, 2]
    assert [t.language for t in tracks] == ["eng", "eng", "ukr"]
    assert all(t.codec == "subrip" for t in tracks)


@has_ffmpeg
def test_probe_reads_dispositions_and_titles(mkv):
    tracks = probe_subtitle_tracks(mkv)
    assert tracks[0].default is True and tracks[0].looks_forced is False
    assert tracks[1].forced is True and tracks[1].looks_forced is True
    assert tracks[0].title == "English Full"


@has_ffmpeg
def test_extract_picks_the_full_english_track(mkv, tmp_path):
    out, chosen = extract_subtitles(mkv, tmp_path / "out.srt")
    assert chosen.subtitle_index == 0
    cues = parse_file(out)
    assert len(cues) == 24  # повна дорожка, не forced (там 2)


@has_ffmpeg
def test_extracted_text_matches_the_source_srt(mkv, tmp_path):
    out, _ = extract_subtitles(mkv, tmp_path / "out.srt")
    extracted = parse_file(out)
    original = parse_file(FIXTURE)
    assert [c.flat_text for c in extracted] == [c.flat_text for c in original]


@has_ffmpeg
def test_extracted_timings_match_the_source(mkv, tmp_path):
    """Тайминги — це те, що ми обіцяємо не змінювати; витяг не має їх зсунути."""
    out, _ = extract_subtitles(mkv, tmp_path / "out.srt")
    extracted = parse_file(out)
    original = parse_file(FIXTURE)
    assert [(c.start_ms, c.end_ms) for c in extracted] == [
        (c.start_ms, c.end_ms) for c in original
    ]


@has_ffmpeg
def test_explicit_track_index_overrides_the_choice(mkv, tmp_path):
    out, chosen = extract_subtitles(mkv, tmp_path / "ukr.srt", track_index=2)
    assert chosen.language == "ukr"
    assert len(parse_file(out)) == 1


@has_ffmpeg
def test_requesting_a_missing_track_fails_clearly(mkv, tmp_path):
    with pytest.raises(MediaError, match="немає дорожки s:9"):
        extract_subtitles(mkv, tmp_path / "x.srt", track_index=9)


@has_ffmpeg
def test_extract_by_language(mkv, tmp_path):
    out, chosen = extract_subtitles(mkv, tmp_path / "ukr.srt", language="ukr")
    assert chosen.language == "ukr"
    assert out.exists()


@has_ffmpeg
def test_describe_tracks_marks_the_one_that_will_be_used(mkv):
    text = describe_tracks(mkv)
    assert "буде вибрана" in text
    assert text.count("s:") >= 3


@has_ffmpeg
def test_extract_creates_missing_parent_directories(mkv, tmp_path):
    target = tmp_path / "deep" / "nested" / "out.srt"
    out, _ = extract_subtitles(mkv, target)
    assert out.exists()


# ================= помилки =================


def test_missing_file_is_reported_clearly():
    with pytest.raises(MediaError, match="файл не існує"):
        probe_subtitle_tracks("/nonexistent/movie.mkv")


def test_extracting_a_bitmap_track_is_refused(tmp_path):
    bitmap = track(codec="hdmv_pgs_subtitle")
    with pytest.raises(BitmapSubtitleError, match="OCR"):
        extract_track("/tmp/whatever.mkv", bitmap, tmp_path / "out.srt")


@has_ffmpeg
def test_non_media_file_is_reported_not_crashed(tmp_path):
    fake = tmp_path / "fake.mkv"
    fake.write_text("це не відеофайл", encoding="utf-8")
    with pytest.raises(MediaError):
        probe_subtitle_tracks(fake)
