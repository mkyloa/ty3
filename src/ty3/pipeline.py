"""Склеювання проходів у один прогін.

Порядок проходів не довільний: кожен наступний спирається на рішення попереднього.
Досьє дає стать → атрибуція дає мовців → біблія фіксує форми звертання й імена →
переклад їх виконує → редактор і лінтери перевіряють виконання.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .checks.base import Issue, Severity
from .checks.subtitle_lint import check_timings_preserved
from .config import Config
from .media import SubtitleTrack, extract_subtitles, is_container
from .model.backends import build_backend
from .model.runner import Runner
from .model.schemas import Bible, Dossier
from .passes import bible as bible_pass
from .passes import consistency, report, research, speakers as speakers_pass, translate
from .srtio.parse import Cue, parse_file
from .srtio.render import render_bilingual, render_srt
from .srtio.segment import build_chunks

__all__ = ["Pipeline", "PipelineResult", "slugify", "prepare_source"]

Reporter = Callable[[str], None]


def slugify(value: str) -> str:
    keep = [char if char.isalnum() else "-" for char in value.casefold()]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "film"


def prepare_source(
    source: str | Path,
    out_root: Path,
    *,
    title: str | None = None,
    track_index: int | None = None,
    sub_language: str = "eng",
    allow_forced: bool = False,
    chooser: Callable[[list[SubtitleTrack]], SubtitleTrack] | None = None,
    timeout_s: int | None = None,
    report: Reporter | None = None,
) -> tuple[Path, Path, SubtitleTrack | None]:
    """Приводить вхід до .srt, витягаючи його з контейнера за потреби.

    Повертає (шлях до srt, тека виводу, вибрана дорожка або None).
    Витягнутий файл лягає в теку виводу як `source.en.srt` — щоб було видно, що
    саме перекладалося, і щоб повторний прогін не витягав його знову.
    """
    source = Path(source)
    say = report or (lambda _message: None)

    from .passes.research import guess_title_from_filename

    guessed, _ = guess_title_from_filename(source)
    out_dir = Path(out_root) / slugify(title or guessed)

    if not is_container(source):
        return source, out_dir, None

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "source.en.srt"

    # Витягання дороге на великому файлі — не робимо його вдруге
    if target.exists() and target.stat().st_size > 0:
        say(f"Субтитри вже витягнуті: {target} (видали файл, щоб витягти заново)")
        return target, out_dir, None

    say(f"Витягаю субтитри з {source.name}…")
    extracted, track = extract_subtitles(
        source,
        target,
        language=sub_language,
        track_index=track_index,
        allow_forced=allow_forced,
        chooser=chooser,
        timeout_s=timeout_s,
        report=say,
    )
    say(f"   дорожка: {track.label()}")
    say(f"   → {extracted}")
    return extracted, out_dir, track


@dataclass
class PipelineResult:
    out_dir: Path
    final_srt: Path
    report_md: Path
    bilingual_srt: Path | None
    issues: list[Issue]
    dossier: Dossier
    bible: Bible

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity is Severity.ERROR)


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        *,
        backend_name: str,
        model: str | None = None,
        out_root: Path = Path("out"),
        use_cache: bool = True,
        report_progress: Reporter | None = None,
    ) -> None:
        self.cfg = cfg
        self.backend_name = backend_name
        self.model = model
        self.out_root = Path(out_root)
        self.use_cache = use_cache
        self.say: Reporter = report_progress or (lambda message: None)

    def run(
        self,
        source: str | Path,
        *,
        title: str | None = None,
        year: int | None = None,
        dossier_path: str | Path | None = None,
        speakers_path: str | Path | None = None,
        bible_path: str | Path | None = None,
        extra_context_path: str | Path | None = None,
        bilingual: bool = False,
        review: bool = False,
        track_index: int | None = None,
        sub_language: str = "eng",
        allow_forced: bool = False,
        chooser: Callable[[list[SubtitleTrack]], SubtitleTrack] | None = None,
        max_chunks: int | None = None,
    ) -> PipelineResult:
        # Вхід може бути контейнером (.mkv/.mp4) — тоді спершу витягаємо субтитри
        source, out_dir, _track = prepare_source(
            source,
            self.out_root,
            title=title,
            track_index=track_index,
            sub_language=sub_language,
            allow_forced=allow_forced,
            chooser=chooser,
            timeout_s=self.cfg.media.extract_timeout_s or None,
            report=self.say,
        )
        source_text = source.read_text(encoding="utf-8", errors="replace")
        cues = parse_file(source)
        self.say(f"Розібрано {len(cues)} реплик із {source.name}")

        out_dir.mkdir(parents=True, exist_ok=True)

        extra = (
            {"search_budget_usd": self.cfg.model.search_budget_usd}
            if self.backend_name == "claude-cli"
            else {}
        )
        backend = build_backend(self.backend_name, self.model, **extra)
        runner = Runner(
            backend,
            out_dir,
            use_cache=self.use_cache,
            jobs=self.cfg.run.jobs,
            on_event=self.say,
        )
        self.say(f"Бекенд: {backend.name} / {backend.model}, тека: {out_dir}")

        extra_context = (
            Path(extra_context_path).read_text(encoding="utf-8")
            if extra_context_path
            else None
        )
        if dossier_path:
            dossier = research.load_dossier(dossier_path)
            self.say(f"Досьє взято з файлу: {dossier_path}")
        else:
            self.say("① Шукаю контекст фільму…")
            dossier = research.research(
                runner,
                cues,
                source_path=source,
                title=title,
                year=year,
                extra_context=extra_context,
                effort=self.cfg.model.effort,
            )
        _dump(out_dir / "dossier.json", dossier.model_dump(mode="json"))
        self.say(
            f"   {dossier.title_en} ({dossier.year or '?'}), персонажів: "
            f"{len(dossier.characters)}, впевненість: "
            f"{dossier.identification_confidence:.0%}"
        )
        if review:
            self._pause(out_dir / "dossier.json", "досьє")
            dossier = research.load_dossier(out_dir / "dossier.json")

        # --- ④ мовці ---
        if speakers_path:
            speaker_map = speakers_pass.load_speakers(speakers_path)
            self.say(f"Мовців взято з файлу: {speakers_path}")
        else:
            self.say("② Визначаю мовців і форми звертання…")
            speaker_map = speakers_pass.attribute_speakers(
                runner, cues, dossier, effort=self.cfg.model.effort
            )
        _dump(out_dir / "speakers.json", speaker_map.model_dump(mode="json"))
        speakers = speaker_map.by_index()
        unknown = sum(1 for item in speakers.values() if item.speaker_gender == "unknown")
        self.say(f"   атрибутовано {len(speakers)}, без визначеної статі: {unknown}")

        # --- ⑤ біблія ---
        if bible_path:
            bible = bible_pass.load_bible(bible_path)
            self.say(f"Біблія взята з файлу: {bible_path}")
        else:
            self.say("③ Укладаю біблію перекладу…")
            bible = bible_pass.build_bible(
                runner, cues, dossier, speaker_map, effort=self.cfg.model.effort
            )
        _dump(out_dir / "bible.json", bible.model_dump(mode="json"))
        (out_dir / "bible.md").write_text(
            bible_pass.bible_to_markdown(bible), encoding="utf-8"
        )
        self.say(
            f"   пар ти/Ви: {len(bible.formality)}, глосарій: {len(bible.glossary)}, "
            f"голоси: {len(bible.voices)}"
        )
        if review:
            self._pause(out_dir / "bible.json", "біблію")
            bible = bible_pass.load_bible(out_dir / "bible.json")

        # --- ⑤ переклад + редактура за один виклик на чанк ---
        chunks = build_chunks(cues, self.cfg.segment)
        total_chunks = len(chunks)
        if max_chunks is not None and max_chunks < total_chunks:
            chunks = chunks[:max_chunks]
            self.say(
                f"⚠️  Тестовий прогін: {len(chunks)} з {total_chunks} чанків — "
                f"це НЕ повний переклад, реплики поза чанком лишаться неперекладеними"
            )
        self.say(f"④ Перекладаю та редагую {len(chunks)} чанк(ів)…")
        result = translate.translate_chunks(
            runner, chunks, dossier, bible, speakers, self.cfg
        )
        _write_srt(out_dir / "draft.srt", cues, result.text, self.cfg)
        self.say(f"   перекладено {len(result.text)} реплик")

        # Репліки поза тестованими чанками — не помилка, а межа `--max-chunks`.
        # Лінтери й підгонка мають бачити лише те, що дійсно намагалися
        # перекласти, інакше кожен нетестований чанк виглядає як 1000+ "помилок".
        lint_cues = [cue for chunk in chunks for cue in chunk.cues]

        # --- ⑥ лінтери + точкові виправлення ---
        self.say("⑤ Перевірки й точкові виправлення…")
        issues = consistency.lint(lint_cues, result, bible, speakers, self.cfg)
        errors_before = sum(1 for issue in issues if issue.severity is Severity.ERROR)
        if errors_before:
            result = consistency.fix_flagged(
                runner, lint_cues, result, issues, dossier, bible, speakers, self.cfg
            )
        self.say(f"   помилок до виправлення: {errors_before}")

        # --- ⑦ підгонка ---
        self.say("⑥ Підгонка під норми читання…")
        result = consistency.fit_overlong(
            runner, lint_cues, result, dossier, bible, speakers, self.cfg
        )

        # --- ⑩ вивід ---
        issues = consistency.lint(lint_cues, result, bible, speakers, self.cfg)
        final_path = out_dir / "final.uk.srt"
        final_text = _write_srt(final_path, cues, result.text, self.cfg)

        timing_issues = check_timings_preserved(source_text, final_text)
        issues.extend(timing_issues)
        if timing_issues:
            self.say(f"   !!! тайминги розійшлися з джерелом: {len(timing_issues)}")

        bilingual_path = None
        if bilingual:
            bilingual_path = out_dir / "bilingual.srt"
            bilingual_path.write_text(
                render_bilingual(cues, result.text, self.cfg.subtitle), encoding="utf-8"
            )

        report_path = out_dir / "qa-report.md"
        report_path.write_text(
            report.build_report(
                cues, result, issues, dossier, bible, speakers, self.cfg,
                stats=runner.stats.summary(),
            ),
            encoding="utf-8",
        )

        summary = runner.stats.summary()
        self.say(
            f"Готово. Викликів: {summary['billed_calls']} "
            f"(з кешу {summary['cached_calls']}), "
            f"токенів виходу: {summary['output_tokens']}"
            + (f", ≈${summary['cost_usd']:.2f}" if summary["cost_usd"] else "")
        )

        return PipelineResult(
            out_dir=out_dir,
            final_srt=final_path,
            report_md=report_path,
            bilingual_srt=bilingual_path,
            issues=issues,
            dossier=dossier,
            bible=bible,
        )

    def _pause(self, path: Path, what: str) -> None:
        self.say(f"\n--- Пауза: перевір {what} у {path} і натисни Enter ---")
        try:
            input()
        except EOFError:
            self.say("(вхід недоступний, продовжую без паузи)")


def _dump(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_srt(path: Path, cues: list[Cue], translations: dict[int, str], cfg: Config) -> str:
    text = render_srt(cues, translations, cfg.subtitle)
    path.write_text(text, encoding="utf-8")
    return text
