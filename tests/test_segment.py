"""Сегментація: чанки не мусять розривати сцену без потреби."""

from __future__ import annotations

from pathlib import Path

import pytest

from ty3.config import SegmentConfig
from ty3.srtio.parse import parse_srt
from ty3.srtio.segment import build_chunks, build_scenes, build_windows

FIXTURE = Path(__file__).parent / "fixtures" / "sample.en.srt"


@pytest.fixture(scope="module")
def cues():
    return parse_srt(FIXTURE.read_text(encoding="utf-8"))


def _cue(index: int, start_ms: int, end_ms: int, text: str = "Line."):
    [parsed] = parse_srt(
        f"{index}\n{_ts(start_ms)} --> {_ts(end_ms)}\n{text}\n"
    )
    return parsed


def _ts(ms: int) -> str:
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


# ---------- сцени ----------


def test_scenes_split_on_long_pause(cues):
    scenes = build_scenes(cues, gap_ms=2500)
    assert len(scenes) > 1
    # Пауза 2,0 → 12,4 і 9,3 → 15,0 — очевидні межі сцен
    starts = [scene.cues[0].index for scene in scenes]
    assert 1 in starts and 2 in starts and 22 in starts


def test_scene_numbers_are_sequential(cues):
    scenes = build_scenes(cues, gap_ms=2500)
    assert [scene.number for scene in scenes] == list(range(1, len(scenes) + 1))


def test_every_cue_lands_in_exactly_one_scene(cues):
    scenes = build_scenes(cues, gap_ms=2500)
    collected = [cue.index for scene in scenes for cue in scene.cues]
    assert collected == [cue.index for cue in cues]


def test_no_scenes_for_empty_input():
    assert build_scenes([], gap_ms=2500) == []


def test_single_cue_is_one_scene():
    scenes = build_scenes([_cue(1, 0, 1000)], gap_ms=2500)
    assert len(scenes) == 1 and len(scenes[0].cues) == 1


def test_gap_exactly_at_threshold_starts_new_scene():
    cues = [_cue(1, 0, 1000), _cue(2, 3500, 4500)]
    assert len(build_scenes(cues, gap_ms=2500)) == 2


def test_gap_below_threshold_keeps_one_scene():
    cues = [_cue(1, 0, 1000), _cue(2, 3400, 4400)]
    assert len(build_scenes(cues, gap_ms=2500)) == 1


# ---------- чанки ----------


def test_all_cues_covered_exactly_once(cues):
    chunks = build_chunks(cues, SegmentConfig())
    collected = [cue.index for chunk in chunks for cue in chunk.cues]
    assert collected == [cue.index for cue in cues]


def test_small_file_is_one_chunk(cues):
    chunks = build_chunks(cues, SegmentConfig())
    assert len(chunks) == 1
    assert chunks[0].number == 1


def test_chunks_respect_cue_limit(cues):
    cfg = SegmentConfig(max_cues_per_chunk=6, max_chars_per_chunk=10_000)
    chunks = build_chunks(cues, cfg)
    assert len(chunks) > 1
    assert all(len(chunk.cues) <= 6 for chunk in chunks)


def test_chunks_respect_char_limit(cues):
    cfg = SegmentConfig(max_cues_per_chunk=999, max_chars_per_chunk=120)
    chunks = build_chunks(cues, cfg)
    assert len(chunks) > 1
    # Одна репліка може сама перевищувати ліміт — тоді вона їде окремим чанком
    assert all(chunk.char_count <= 120 or len(chunk.cues) == 1 for chunk in chunks)


def test_chunk_boundaries_prefer_scene_boundaries(cues):
    """Коли жодна сцена не перевищує ліміт, чанки ріжуться строго по сценах."""
    scenes = build_scenes(cues, SegmentConfig().scene_gap_ms)
    largest = max(len(scene.cues) for scene in scenes)
    cfg = SegmentConfig(max_cues_per_chunk=largest, max_chars_per_chunk=10_000)

    scene_starts = {scene.cues[0].index for scene in scenes}
    chunks = build_chunks(cues, cfg)
    assert len(chunks) == len(scenes)
    for chunk in chunks:
        assert chunk.cues[0].index in scene_starts


def test_oversized_scene_is_split_even_though_that_breaks_a_scene(cues):
    """Якщо сцена сама не влазить у чанк — розриву не уникнути, це очікувано."""
    cfg = SegmentConfig(max_cues_per_chunk=8, max_chars_per_chunk=10_000)
    scenes = build_scenes(cues, cfg.scene_gap_ms)
    assert max(len(scene.cues) for scene in scenes) > cfg.max_cues_per_chunk
    chunks = build_chunks(cues, cfg)
    assert all(len(chunk.cues) <= 8 for chunk in chunks)
    # Порядок і повнота зберігаються навіть при розриві сцени
    assert [c.index for ch in chunks for c in ch.cues] == [c.index for c in cues]


def test_oversized_scene_is_split():
    # Одна щільна сцена з 12 реплик без пауз
    cues = [_cue(n, n * 1000, n * 1000 + 900) for n in range(1, 13)]
    cfg = SegmentConfig(max_cues_per_chunk=5, max_chars_per_chunk=10_000)
    assert len(build_scenes(cues, cfg.scene_gap_ms)) == 1
    chunks = build_chunks(cues, cfg)
    assert len(chunks) == 3
    assert [len(chunk.cues) for chunk in chunks] == [5, 5, 2]


def test_context_windows_attached(cues):
    cfg = SegmentConfig(max_cues_per_chunk=6, max_chars_per_chunk=10_000, context_before=3, context_after=3)
    chunks = build_chunks(cues, cfg)
    assert chunks[0].context_before == []  # перед першим чанком контексту немає
    assert chunks[-1].context_after == []

    # Другий чанк починається з репліки 2 — ліворуч від нього є лише одна
    assert [c.index for c in chunks[1].context_before] == [1]

    # А в глибині файлу контекст повний з обох боків
    middle = chunks[2]
    assert len(middle.context_before) == 3
    assert middle.context_before[-1].index == middle.cues[0].index - 1
    assert middle.context_after[0].index == middle.cues[-1].index + 1


def test_context_does_not_overlap_own_cues(cues):
    cfg = SegmentConfig(max_cues_per_chunk=6, max_chars_per_chunk=10_000)
    for chunk in build_chunks(cues, cfg):
        own = set(chunk.indices)
        assert own.isdisjoint({c.index for c in chunk.context_before})
        assert own.isdisjoint({c.index for c in chunk.context_after})


def test_scene_numbers_recorded_on_chunks(cues):
    chunks = build_chunks(cues, SegmentConfig())
    assert chunks[0].scene_numbers
    assert chunks[0].scene_numbers == sorted(chunks[0].scene_numbers)


def test_no_chunks_for_empty_input():
    assert build_chunks([], SegmentConfig()) == []


# ---------- вікна для атрибуції мовців ----------


def test_windows_cover_everything(cues):
    windows = build_windows(cues, size=10, overlap=4)
    covered = [cue.index for _, window in windows for cue in window]
    assert covered == [cue.index for cue in cues]


def test_window_overlap_is_context_only(cues):
    windows = build_windows(cues, size=10, overlap=4)
    first_context, first_window = windows[0]
    assert first_context == []
    second_context, second_window = windows[1]
    assert len(second_context) == 4
    assert second_context[-1].index == second_window[0].index - 1


def test_window_size_must_be_positive(cues):
    with pytest.raises(ValueError):
        build_windows(cues, size=0, overlap=2)
