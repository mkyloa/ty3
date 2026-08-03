"""Витягання субтитрів із контейнерів (MKV, MP4, WebM) через ffmpeg.

Два нетривіальні місця, через які тут окремий модуль, а не три рядки в CLI:

1. **Вибір дорожки.** У релізі їх зазвичай кілька: повна англійська, forced
   (лише чужомовні репліки — кілька десятків реплик), SDH з описами звуків,
   плюс інші мови. Узяти першу-ліпшу — це з великою ймовірністю перекласти
   forced-дорожку й отримати 40 реплик замість 1500.
2. **Растрові субтитри.** PGS (Blu-ray), VobSub (DVD) і DVB — це зображення, а не
   текст. Витягти їх як SRT неможливо без OCR. Тихо віддати порожній файл або
   зіпсований текст — найгірше, що тут можна зробити, тому це явна помилка.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

__all__ = [
    "SubtitleTrack",
    "MediaError",
    "BitmapSubtitleError",
    "TrackSelectionCancelled",
    "is_container",
    "estimate_timeout_s",
    "media_duration_s",
    "probe_subtitle_tracks",
    "pick_track",
    "choose_track",
    "extract_track",
    "extract_subtitles",
    "describe_tracks",
]

# Контейнери, у яких взагалі варто шукати субтитри
CONTAINER_SUFFIXES = {".mkv", ".mp4", ".m4v", ".webm", ".mov", ".ts", ".avi", ".ogm"}

# Кодеки-зображення: текст із них дістати не можна, потрібен OCR
BITMAP_CODECS = {
    "hdmv_pgs_subtitle",  # Blu-ray
    "dvd_subtitle",  # DVD VobSub
    "dvb_subtitle",
    "xsub",
    "dvbsub",
}
# Текстові кодеки, які ffmpeg конвертує в SRT
TEXT_CODECS = {
    "subrip",
    "srt",
    "ass",
    "ssa",
    "mov_text",
    "text",
    "webvtt",
    "ttml",
    "microdvd",
    "subviewer",
    "subviewer1",
    "sami",
    "realtext",
    "stl",
}

_ENGLISH = {"eng", "en", "en-us", "en-gb", "english"}
_MARKERS_FORCED = ("forced", "foreign", "signs", "songs")
_MARKERS_SDH = ("sdh", "hearing", "cc", "closed caption")


class MediaError(RuntimeError):
    """Контейнер не читається або в ньому немає придатних субтитрів."""


class BitmapSubtitleError(MediaError):
    """Знайдено лише растрові субтитри — потрібен OCR, якого тут немає."""


class TrackSelectionCancelled(MediaError):
    """Користувач скасував вибір дорожки."""


@dataclass(frozen=True)
class SubtitleTrack:
    subtitle_index: int  # порядковий номер серед субтитрових потоків → -map 0:s:N
    stream_index: int  # абсолютний індекс потоку в контейнері
    codec: str
    language: str | None = None
    title: str | None = None
    forced: bool = False
    default: bool = False
    hearing_impaired: bool = False
    frames: int | None = None  # к-ть реплик, якщо контейнер її записав

    @property
    def is_bitmap(self) -> bool:
        return self.codec in BITMAP_CODECS

    @property
    def is_text(self) -> bool:
        return self.codec in TEXT_CODECS

    @property
    def is_english(self) -> bool:
        return (self.language or "").lower() in _ENGLISH

    @property
    def looks_forced(self) -> bool:
        """Forced-дорожка містить лише чужомовні репліки — не те, що нам треба."""
        haystack = (self.title or "").lower()
        return self.forced or any(marker in haystack for marker in _MARKERS_FORCED)

    @property
    def looks_sdh(self) -> bool:
        haystack = (self.title or "").lower()
        return self.hearing_impaired or any(marker in haystack for marker in _MARKERS_SDH)

    def label(self) -> str:
        bits = [f"s:{self.subtitle_index}", self.codec, self.language or "мова?"]
        if self.title:
            bits.append(f"«{self.title}»")
        for flag, name in (
            (self.default, "default"),
            (self.looks_forced, "forced"),
            (self.looks_sdh, "SDH"),
            (self.is_bitmap, "РАСТР"),
        ):
            if flag:
                bits.append(name)
        if self.frames:
            bits.append(f"{self.frames} реплик")
        return " | ".join(bits)


def is_container(path: str | Path) -> bool:
    return Path(path).suffix.lower() in CONTAINER_SUFFIXES


def _require(tool: str) -> str:
    found = shutil.which(tool)
    if found is None:
        raise MediaError(
            f"не знайдено {tool!r} у PATH. Встанови ffmpeg: brew install ffmpeg"
        )
    return found


def probe_subtitle_tracks(path: str | Path) -> list[SubtitleTrack]:
    """Перелічує субтитрові дорожки контейнера через ffprobe."""
    path = Path(path)
    if not path.exists():
        raise MediaError(f"файл не існує: {path}")

    command = [
        _require("ffprobe"),
        "-v", "error",
        "-select_streams", "s",
        "-show_entries", "stream=index,codec_name:stream_tags:stream_disposition",
        "-of", "json",
        str(path),
    ]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"ffprobe не завершився за 120 с на {path.name}") from exc
    if done.returncode != 0:
        raise MediaError(f"ffprobe не зміг прочитати {path.name}: {done.stderr.strip()[:400]}")

    try:
        payload = json.loads(done.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe віддав не JSON для {path.name}") from exc

    tracks: list[SubtitleTrack] = []
    for position, stream in enumerate(payload.get("streams") or []):
        tags = {key.lower(): value for key, value in (stream.get("tags") or {}).items()}
        disposition = stream.get("disposition") or {}
        frames = tags.get("number_of_frames") or tags.get("number_of_frames-eng")
        tracks.append(
            SubtitleTrack(
                subtitle_index=position,
                stream_index=int(stream.get("index", position)),
                codec=(stream.get("codec_name") or "?").lower(),
                language=tags.get("language"),
                title=tags.get("title"),
                forced=bool(disposition.get("forced")),
                default=bool(disposition.get("default")),
                hearing_impaired=bool(disposition.get("hearing_impaired")),
                frames=int(frames) if frames and str(frames).isdigit() else None,
            )
        )
    return tracks


def _score(track: SubtitleTrack, language: str, allow_forced: bool) -> int:
    """Чим вище, тим краще. Порядок вагів відображає ціну помилки."""
    score = 0
    wanted = language.lower()

    if (track.language or "").lower() == wanted or (
        wanted in _ENGLISH and track.is_english
    ):
        score += 1000
    elif track.language in (None, "und"):
        score += 200  # мова не вказана — можливо, та сама
    else:
        score -= 800  # інша мова: гірше за будь-що інше

    # Forced-дорожка коротка й неповна: перекласти її замість повної — груба помилка
    if track.looks_forced and not allow_forced:
        score -= 500
    if track.looks_sdh:
        score -= 40  # придатна, але з описами звуків; повна краща
    if track.default:
        score += 30
    if track.codec in ("subrip", "srt"):
        score += 20  # найпростіший шлях, без конвертації з ASS
    if track.frames:
        score += min(track.frames // 50, 60)  # повна дорожка має більше реплик
    return score


def pick_track(
    tracks: list[SubtitleTrack],
    *,
    language: str = "eng",
    allow_forced: bool = False,
) -> SubtitleTrack:
    """Вибирає найдоречнішу текстову дорожку.

    Растрові дорожки не є придатними: якщо текстових немає взагалі, кидає
    BitmapSubtitleError із переліком того, що знайдено, — щоб було видно, що файл
    не «без субтитрів», а потребує OCR.
    """
    if not tracks:
        raise MediaError("у контейнері немає субтитрових дорожок")

    text_tracks = [track for track in tracks if track.is_text]
    if not text_tracks:
        found = "; ".join(track.label() for track in tracks)
        raise BitmapSubtitleError(
            "усі субтитри в контейнері растрові (зображення), текст із них без OCR "
            f"дістати неможливо. Знайдено: {found}. "
            "Візьми окремий .srt або пропусти файл через OCR (напр. Subtitle Edit)."
        )

    best = max(text_tracks, key=lambda track: _score(track, language, allow_forced))
    if _score(best, language, allow_forced) < 0:
        available = "; ".join(track.label() for track in text_tracks)
        raise MediaError(
            f"немає текстової дорожки мовою {language!r}. Доступні: {available}. "
            "Вкажи потрібну через --track N або іншу мову через --language."
        )
    return best


def choose_track(
    tracks: list[SubtitleTrack],
    *,
    language: str = "eng",
    allow_forced: bool = False,
    input_stream=None,
    say=None,
    max_attempts: int = 5,
) -> SubtitleTrack:
    """Показує дорожки й дає вибрати вручну; Enter бере рекомендовану.

    Свідомо стійка до неінтерактивного оточення: якщо stdin не термінал або
    вхідний потік вичерпано, повертає рекомендовану дорожку з повідомленням, а не
    зависає. Прогін у фоні чи в CI не має ставати на паузу через це.
    """
    import sys

    stream = input_stream if input_stream is not None else sys.stdin
    report = say if say is not None else print

    if not tracks:
        raise MediaError("у контейнері немає субтитрових дорожок")

    # Растр без жодної текстової дорожки — питати нічого, тут потрібен OCR
    recommended = pick_track(tracks, language=language, allow_forced=allow_forced)

    selectable = [track for track in tracks if track.is_text]
    if len(selectable) == 1:
        report(f"Текстова дорожка одна, беру її: {selectable[0].label()}")
        return selectable[0]

    report("Субтитрові дорожки:")
    for track in tracks:
        mark = "  ← рекомендовано" if track is recommended else ""
        note = "  (растрова, потрібен OCR)" if track.is_bitmap else ""
        report(f"  [{track.subtitle_index}] {track.label()}{note}{mark}")

    if not (hasattr(stream, "isatty") and stream.isatty()) and input_stream is None:
        report(f"Вхід неінтерактивний — беру рекомендовану: s:{recommended.subtitle_index}")
        return recommended

    by_index = {track.subtitle_index: track for track in tracks}
    for _attempt in range(max_attempts):
        report(
            f"Яку взяти? [Enter — рекомендовану s:{recommended.subtitle_index}, "
            f"номер, q — скасувати]: "
        )
        line = stream.readline()
        if not line:  # EOF
            report(f"Ввід закінчився — беру рекомендовану: s:{recommended.subtitle_index}")
            return recommended

        answer = line.strip()
        if not answer:
            return recommended
        if answer.lower() in ("q", "quit", "скасувати"):
            raise TrackSelectionCancelled("вибір дорожки скасовано")

        digits = answer.removeprefix("s:").strip()
        if not digits.isdigit() or int(digits) not in by_index:
            report(f"  Немає дорожки {answer!r}. Доступні: {sorted(by_index)}")
            continue

        chosen = by_index[int(digits)]
        if chosen.is_bitmap:
            report(
                f"  Дорожка s:{chosen.subtitle_index} растрова — текст із неї без OCR "
                "не дістати. Вибери текстову."
            )
            continue
        return chosen

    report(f"Забагато невдалих спроб — беру рекомендовану: s:{recommended.subtitle_index}")
    return recommended


def media_duration_s(path: str | Path) -> float | None:
    """Тривалість контейнера — потрібна, щоб показувати відсоток витягання."""
    command = [
        _require("ffprobe"),
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    try:
        return float((done.stdout or "").strip())
    except ValueError:
        return None


def estimate_timeout_s(
    path: str | Path,
    *,
    floor_mb_s: float = 6.0,
    base_s: int = 300,
    cap_s: int = 7200,
) -> int:
    """Скільки чекати на ffmpeg, виходячи з розміру файлу.

    ffmpeg мусить прочитати контейнер **цілком**: субтитрові пакети розкидані по
    всій довжині фільму, тож пропустити відеодані не можна. Виміряно на реальному
    зовнішньому диску: 16 МБ/с, тобто 7.6 ГБ — це ~475 с лише на читання. Фіксована
    константа тут завжди буде або замалою для великого файлу на повільному носії,
    або безглуздо великою для маленького.

    `floor_mb_s` — консервативна нижня оцінка швидкості носія (USB 2.0, HDD).
    """
    try:
        size_mb = Path(path).stat().st_size / (1024 * 1024)
    except OSError:
        return base_s + 600
    return int(min(cap_s, base_s + size_mb / max(0.5, floor_mb_s)))


def extract_track(
    path: str | Path,
    track: SubtitleTrack,
    out_path: str | Path,
    *,
    overwrite: bool = True,
    timeout_s: int | None = None,
    report=None,
) -> Path:
    """Витягає одну дорожку в SRT. ASS/VTT конвертуються, розмітка спрощується.

    Показує прогрес: на великому файлі з повільного носія це триває хвилини, і без
    прогресу процес неможливо відрізнити від зависання.
    """
    path, out_path = Path(path), Path(out_path)
    say = report if report is not None else (lambda _message: None)
    if track.is_bitmap:
        raise BitmapSubtitleError(
            f"дорожка {track.label()} растрова — потрібен OCR, конвертація в SRT неможлива"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    deadline_s = timeout_s if timeout_s is not None else estimate_timeout_s(path)
    duration_s = media_duration_s(path)
    size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0
    if size_mb > 500:
        say(
            f"   файл {size_mb / 1024:.1f} ГБ — ffmpeg читає його цілком, "
            f"ліміт {deadline_s // 60} хв"
        )

    command = [
        _require("ffmpeg"),
        "-y" if overwrite else "-n",
        "-v", "error",
        "-nostats",
        "-progress", "pipe:1",
        "-i", str(path),
        "-map", f"0:s:{track.subtitle_index}",
        # Явно відрізаємо відео/аудіо/дані: демультиплексувати контейнер усе одно
        # доведеться, але жодного зайвого декодування не буде
        "-vn", "-an", "-dn",
        "-c:s", "srt",
        str(out_path),
    ]

    stderr_file = out_path.with_suffix(".ffmpeg.log")
    try:
        with stderr_file.open("w", encoding="utf-8") as errors:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
                bufsize=1,
            )
            _pump_progress(process, duration_s, deadline_s, say)
        returncode = process.returncode
        problems = stderr_file.read_text(encoding="utf-8", errors="replace").strip()
    finally:
        stderr_file.unlink(missing_ok=True)

    if returncode != 0:
        raise MediaError(
            f"ffmpeg не витягнув дорожку s:{track.subtitle_index}: {problems[:400]}"
        )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise MediaError(
            f"дорожка s:{track.subtitle_index} витягнулася порожньою — "
            "можливо, вона насправді не містить тексту"
        )
    return out_path


def _pump_progress(process, duration_s: float | None, deadline_s: int, say) -> None:
    """Читає `-progress` ffmpeg, доповідає відсоток і стежить за дедлайном."""
    import time

    started = time.monotonic()
    last_report = 0.0

    for line in process.stdout or []:
        elapsed = time.monotonic() - started
        if elapsed > deadline_s:
            process.kill()
            process.wait(timeout=30)
            raise MediaError(
                f"ffmpeg не завершився за {deadline_s // 60} хв. Файл великий або "
                f"носій повільний. Підніми ліміт: extract_timeout_s у config.toml"
            )

        key, _, value = line.strip().partition("=")
        if key == "out_time_us" and duration_s and elapsed - last_report >= 5:
            last_report = elapsed
            try:
                done_s = int(value) / 1_000_000
            except ValueError:
                continue
            percent = min(100.0, done_s / duration_s * 100)
            say(f"   витягнуто {percent:.0f}%  ({elapsed:.0f} с)")

    process.wait(timeout=60)


def extract_subtitles(
    path: str | Path,
    out_path: str | Path,
    *,
    language: str = "eng",
    track_index: int | None = None,
    allow_forced: bool = False,
    chooser: Callable[[list[SubtitleTrack]], SubtitleTrack] | None = None,
    timeout_s: int | None = None,
    report=None,
) -> tuple[Path, SubtitleTrack]:
    """Витягає задану, вибрану вручну або найдоречнішу дорожку.

    Пріоритет: явний `track_index` → `chooser` (інтерактивний вибір) → автовибір.
    Явно вказаний номер бʼє інтерактив: якщо людина вже сказала, що їй треба,
    питати вдруге — знущання.
    """
    tracks = probe_subtitle_tracks(path)
    if track_index is not None:
        matching = [track for track in tracks if track.subtitle_index == track_index]
        if not matching:
            available = "; ".join(track.label() for track in tracks) or "жодної"
            raise MediaError(f"немає дорожки s:{track_index}. Доступні: {available}")
        chosen = matching[0]
    elif chooser is not None:
        chosen = chooser(tracks)
    else:
        chosen = pick_track(tracks, language=language, allow_forced=allow_forced)
    extracted = extract_track(
        path, chosen, out_path, timeout_s=timeout_s, report=report
    )
    return extracted, chosen


def describe_tracks(path: str | Path) -> str:
    """Людиночитний перелік дорожок — для `ty3 tracks`."""
    tracks = probe_subtitle_tracks(path)
    if not tracks:
        return "Субтитрових дорожок не знайдено."
    lines = [f"Субтитрові дорожки у {Path(path).name}:"]
    try:
        best = pick_track(tracks)
    except MediaError:
        best = None
    for track in tracks:
        mark = " ← буде вибрана" if best is not None and track is best else ""
        lines.append(f"  {track.label()}{mark}")
    return "\n".join(lines)
