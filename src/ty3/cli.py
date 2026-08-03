"""CLI: translate | research | check | estimate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .checks.base import Severity
from .checks.subtitle_lint import check_all, check_timings_preserved
from .checks.uk_lint import check_uk_all
from .config import Config, load_config
from .media import (
    MediaError,
    TrackSelectionCancelled,
    choose_track,
    describe_tracks,
    extract_subtitles,
)
from .model.backends import available_backends, build_backend
from .model.backends.base import BackendError, PermanentBackendError, Prompt
from .model.runner import Runner
from .model.schemas import Bible, SpeakerMap
from .passes import bible as bible_pass
from .passes import consistency, research
from .passes.context import chunk_body
from .pipeline import Pipeline, prepare_source, slugify
from .srtio.parse import SrtParseError, parse_file
from .srtio.segment import build_chunks

__all__ = ["main"]


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Мінімальний .env, щоб не тягнути залежність для трьох рядків."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _say(message: str) -> None:
    print(message, flush=True)


def _make_chooser(args):
    """Інтерактивний вибір дорожки — лише коли попросили прапорцем.

    Не робимо це поведінкою за замовчуванням: прогін у фоні або в CI не має
    ставати на паузу й чекати на ввід, якого ніхто не дасть.
    """
    if not getattr(args, "pick_track", False):
        return None

    def chooser(tracks):
        return choose_track(
            tracks,
            language=args.sub_language,
            allow_forced=args.allow_forced,
            say=_say,
        )

    return chooser


def _add_track_flags(parser: argparse.ArgumentParser) -> None:
    """Прапорці вибору дорожки — потрібні лише коли вхід є контейнером."""
    parser.add_argument(
        "--track", type=int, default=None,
        help="номер субтитрової дорожки (див. `ty3 tracks`); типово вибирається сама",
    )
    parser.add_argument(
        "--sub-language", default="eng",
        help="мова дорожки для автовибору (типово eng)",
    )
    parser.add_argument(
        "--allow-forced", action="store_true",
        help="дозволити forced-дорожку (лише чужомовні репліки — зазвичай не те, що треба)",
    )
    parser.add_argument(
        "--pick-track", action="store_true",
        help="показати дорожки й вибрати вручну (Enter — рекомендовану)",
    )


def _add_common(parser: argparse.ArgumentParser, cfg: Config) -> None:
    parser.add_argument(
        "srt", type=Path, help="англійський .srt або контейнер (.mkv/.mp4/.webm)"
    )
    parser.add_argument(
        "--backend",
        default=cfg.model.backend,
        choices=list(available_backends),
        help=f"провайдер моделі (з config.toml: {cfg.model.backend})",
    )
    parser.add_argument("--model", default=None, help="перевизначити модель бекенда")
    parser.add_argument("--out", type=Path, default=Path("out"), help="коренева тека виводу")


def build_parser(cfg: Config) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ty3",
        description="Багатопрохідний переклад субтитрів EN→UK з контекстом персонажів.",
    )
    parser.add_argument("--version", action="version", version=f"ty3 {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- translate ---
    translate_cmd = sub.add_parser("translate", help="повний прогін: SRT → український SRT")
    _add_common(translate_cmd, cfg)
    translate_cmd.add_argument("--title", default=None, help="назва фільму, якщо ім'я файлу неточне")
    translate_cmd.add_argument("--year", type=int, default=None)
    translate_cmd.add_argument("--dossier", type=Path, default=None, help="готове dossier.json")
    translate_cmd.add_argument(
        "--speakers", type=Path, default=None,
        help="готове speakers.json — атрибуція не залежить від моделі, можна "
        "порахувати раз і переносити між прогонами з різним --model/--backend",
    )
    translate_cmd.add_argument("--bible", type=Path, default=None, help="готове bible.json")
    translate_cmd.add_argument(
        "--context", type=Path, default=None,
        help="файл із додатковим контекстом: синопсис, імена, каст",
    )
    translate_cmd.add_argument("--bilingual", action="store_true", help="ще й двомовний файл")
    translate_cmd.add_argument(
        "--review", action="store_true",
        help="пауза після досьє та біблії, щоб виправити руками",
    )
    _add_track_flags(translate_cmd)
    translate_cmd.add_argument("--jobs", type=int, default=None, help="паралельних викликів")
    translate_cmd.add_argument(
        "--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"]
    )
    translate_cmd.add_argument("--no-cache", action="store_true", help="ігнорувати дисковий кеш")
    translate_cmd.add_argument(
        "--max-chunks", type=int, default=None,
        help="перекласти лише перші N чанків — для оцінки вартості перед повним "
        "прогоном; підклади готові --dossier/--speakers/--bible, щоб порахувати "
        "чисту вартість самого перекладу без research/speakers/bible",
    )

    # --- research ---
    research_cmd = sub.add_parser("research", help="лише знайти контекст → dossier.json")
    _add_common(research_cmd, cfg)
    research_cmd.add_argument("--title", default=None)
    research_cmd.add_argument("--year", type=int, default=None)
    research_cmd.add_argument("--context", type=Path, default=None)
    _add_track_flags(research_cmd)

    # --- tracks ---
    tracks_cmd = sub.add_parser(
        "tracks", help="перелічити субтитрові дорожки контейнера"
    )
    tracks_cmd.add_argument("media", type=Path, help="файл .mkv / .mp4 / .webm")

    # --- extract ---
    extract_cmd = sub.add_parser(
        "extract", help="витягти субтитри з контейнера в .srt"
    )
    extract_cmd.add_argument("media", type=Path, help="файл .mkv / .mp4 / .webm")
    extract_cmd.add_argument(
        "-o", "--out-file", type=Path, default=None,
        help="куди писати (типово поряд із контейнером, <назва>.en.srt)",
    )
    _add_track_flags(extract_cmd)

    # --- check ---
    check_cmd = sub.add_parser("check", help="прогнати лінтери на готовому перекладі")
    check_cmd.add_argument("srt", type=Path, help="англійський .srt (джерело)")
    check_cmd.add_argument("translated", type=Path, help="український .srt")
    check_cmd.add_argument("--bible", type=Path, default=None)
    check_cmd.add_argument("--speakers", type=Path, default=None)

    # --- estimate ---
    estimate_cmd = sub.add_parser("estimate", help="оцінити обсяг і витрати без запуску")
    _add_common(estimate_cmd, cfg)

    return parser


# ---------- команди ----------


def cmd_translate(args, cfg: Config) -> int:
    if args.jobs:
        cfg = _with(cfg, "run", jobs=args.jobs)
    if args.effort:
        cfg = _with(cfg, "model", effort=args.effort)

    pipeline = Pipeline(
        cfg,
        backend_name=args.backend,
        model=args.model,
        out_root=args.out,
        use_cache=not args.no_cache,
        report_progress=_say,
    )
    result = pipeline.run(
        args.srt,
        title=args.title,
        year=args.year,
        dossier_path=args.dossier,
        speakers_path=args.speakers,
        bible_path=args.bible,
        extra_context_path=args.context,
        bilingual=args.bilingual,
        review=args.review,
        track_index=args.track,
        sub_language=args.sub_language,
        allow_forced=args.allow_forced,
        chooser=_make_chooser(args),
        max_chunks=args.max_chunks,
    )
    _say("")
    _say(f"Переклад:  {result.final_srt}")
    _say(f"Звіт:      {result.report_md}")
    if result.bilingual_srt:
        _say(f"Двомовний: {result.bilingual_srt}")
    if result.error_count:
        _say(f"\nУвага: {result.error_count} помилок лишилося — дивись звіт.")
    return 0


def cmd_tracks(args, cfg: Config) -> int:
    _say(describe_tracks(args.media))
    return 0


def cmd_extract(args, cfg: Config) -> int:
    target = args.out_file or args.media.with_suffix("").with_suffix(".en.srt")
    path, track = extract_subtitles(
        args.media,
        target,
        language=args.sub_language,
        track_index=args.track,
        allow_forced=args.allow_forced,
        chooser=_make_chooser(args),
        timeout_s=cfg.media.extract_timeout_s or None,
        report=_say,  # на файлі в кілька ГБ без прогресу це схоже на зависання
    )
    cues = parse_file(path)
    _say(f"Дорожка: {track.label()}")
    _say(f"Реплик:  {len(cues)}")
    _say(f"Файл:    {path}")
    return 0


def cmd_research(args, cfg: Config) -> int:
    source, out_dir, _track = prepare_source(
        args.srt,
        args.out,
        title=args.title,
        track_index=args.track,
        sub_language=args.sub_language,
        allow_forced=args.allow_forced,
        chooser=_make_chooser(args),
        report=_say,
    )
    cues = parse_file(source)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = Runner(build_backend(args.backend, args.model), out_dir, on_event=_say)
    dossier = research.research(
        runner,
        cues,
        source_path=source,
        title=args.title,
        year=args.year,
        extra_context=(
            args.context.read_text(encoding="utf-8") if args.context else None
        ),
        effort=cfg.model.effort,
    )
    path = out_dir / "dossier.json"
    path.write_text(
        json.dumps(dossier.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _say(f"\n{dossier.title_en} ({dossier.year or '?'}) — {path}")
    for character in dossier.characters:
        _say(
            f"  {character.name_en:22} {character.name_uk or '—':22} "
            f"{character.gender:8} {character.confidence:.0%}"
        )
    return 0


def cmd_check(args, cfg: Config) -> int:
    source_text = args.srt.read_text(encoding="utf-8", errors="replace")
    cues = parse_file(args.srt)
    translated_cues = parse_file(args.translated)
    translations = {cue.index: cue.flat_text for cue in translated_cues}

    bible = bible_pass.load_bible(args.bible) if args.bible else Bible()
    speakers = {}
    if args.speakers:
        speakers = SpeakerMap.model_validate(
            json.loads(args.speakers.read_text(encoding="utf-8"))
        ).by_index()

    rendered = {cue.index: [line.render() for line in cue.lines] for cue in translated_cues}
    issues = check_all(
        cues, rendered, cfg.subtitle,
        allowed_latin=consistency.allowed_latin_from(bible),
    )
    issues += check_uk_all(cues, translations, bible=bible, speakers=speakers)
    issues += check_timings_preserved(
        source_text, args.translated.read_text(encoding="utf-8", errors="replace")
    )

    for severity in (Severity.ERROR, Severity.WARNING, Severity.INFO):
        selected = [issue for issue in issues if issue.severity is severity]
        if not selected:
            continue
        _say(f"\n{severity.value.upper()}: {len(selected)}")
        for issue in selected[:80]:
            _say(f"  {issue}")
        if len(selected) > 80:
            _say(f"  …ще {len(selected) - 80}")

    errors = sum(1 for issue in issues if issue.severity is Severity.ERROR)
    _say(f"\nПомилок: {errors}, попереджень: "
         f"{sum(1 for i in issues if i.severity is Severity.WARNING)}")
    return 1 if errors else 0


def cmd_estimate(args, cfg: Config) -> int:
    source, _out_dir, track = prepare_source(args.srt, args.out, report=_say)
    if track is not None:
        _say(f"Дорожка:         {track.label()}")
    cues = parse_file(source)
    chunks = build_chunks(cues, cfg.segment)
    translatable = [cue for cue in cues if cue.is_translatable]

    backend = build_backend(args.backend, args.model)
    prompts_for_translation = [
        Prompt(
            system="",
            context="",
            user=chunk_body(chunk.cues, cfg.subtitle, {}, before=chunk.context_before),
        )
        for chunk in chunks
    ]
    chunk_tokens = sum(backend.count_tokens(prompt) for prompt in prompts_for_translation)

    # Кожен чанк проходить переклад і редактуру; плюс research, speakers, bible
    windows = max(1, len(translatable) // 80 + 1)
    calls = 2 + windows + 1 + 2 * len(chunks)

    _say(f"Файл:            {source.name}")
    _say(f"Реплик:          {len(cues)} (з текстом {len(translatable)})")
    _say(f"Чанків:          {len(chunks)}")
    _say(f"Вікон мовців:    {windows}")
    _say(f"Викликів моделі: ≈{calls}")
    _say(f"Бекенд:          {backend.name} / {backend.model}")
    _say(f"Токенів чанків:  ≈{chunk_tokens} (без стабільного префікса)")
    _say("")
    _say("Точну вартість покаже run.log після прогону — там реальні usage за виклик.")
    return 0


def _with(cfg: Config, section: str, **changes) -> Config:
    from dataclasses import replace

    return replace(cfg, **{section: replace(getattr(cfg, section), **changes)})


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    cfg = load_config()
    parser = build_parser(cfg)
    args = parser.parse_args(argv)

    handlers = {
        "translate": cmd_translate,
        "research": cmd_research,
        "tracks": cmd_tracks,
        "extract": cmd_extract,
        "check": cmd_check,
        "estimate": cmd_estimate,
    }
    try:
        return handlers[args.command](args, cfg)
    except SrtParseError as exc:
        print(f"Помилка розбору SRT: {exc}", file=sys.stderr)
        return 2
    except TrackSelectionCancelled:
        print("Скасовано.", file=sys.stderr)
        return 130
    except MediaError as exc:
        print(f"Помилка витягання субтитрів: {exc}", file=sys.stderr)
        return 4
    except PermanentBackendError as exc:
        # Кеш уже зберіг усе, що вдалося перекласти — повторний запуск (після
        # того, як ліміт знято) продовжить з того самого місця, а не спочатку.
        print(f"Помилка бекенда (постійна, повтор не допоможе): {exc}", file=sys.stderr)
        print(
            "Кеш збережено — після зняття ліміту повторний запуск продовжить "
            "з того місця. Варіанти: почекати наступний розрахунковий період, "
            "попросити адміна підняти ліміт, або тимчасово --backend gemini "
            "(окрема безкоштовна квота, не витрачає цей ліміт).",
            file=sys.stderr,
        )
        return 5
    except BackendError as exc:
        print(f"Помилка бекенда: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nПерервано. Кеш збережено — повторний запуск продовжить з того місця.",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
