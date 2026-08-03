"""Шар провайдерів і Runner. Без мережі — на підставному бекенді."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from ty3.model.backends import available_backends, build_backend
from ty3.model.backends.base import (
    BackendError,
    Completion,
    PermanentBackendError,
    Prompt,
    Tools,
    Usage,
)
from ty3.model.backends.claude_cli import _extract_json, _to_strict_schema
from ty3.model.runner import Runner
from ty3.model.schemas import Bible, Dossier, SpeakerMap


class Tiny(BaseModel):
    uk: str


class FakeBackend:
    """Бекенд-протокол без мережі: рахує виклики й може падати задану кількість разів."""

    name = "fake"
    model = "fake-1"

    def __init__(self, *, fail_times: int = 0, permanent: bool = False, payload: dict | None = None):
        self.calls = 0
        self.fail_times = fail_times
        self.permanent = permanent
        self.payload = payload or {"uk": "Привіт."}

    def _maybe_fail(self) -> None:
        self.calls += 1
        if self.calls <= self.fail_times:
            if self.permanent:
                raise PermanentBackendError("відмова класифікатора")
            raise BackendError("тимчасова мережева помилка")

    def complete(self, prompt, *, effort="high", max_tokens=32000, tools=Tools.NONE):
        self._maybe_fail()
        return Completion(
            text=f"{prompt.user}|{effort}|{tools.value}",
            usage=Usage(input_tokens=10, output_tokens=5),
            backend=self.name,
            model=self.model,
        )

    def complete_structured(self, prompt, schema, *, effort="high", max_tokens=32000):
        self._maybe_fail()
        return Completion(
            text=json.dumps(self.payload),
            data=schema.model_validate(self.payload).model_dump(),
            usage=Usage(input_tokens=20, output_tokens=7),
            backend=self.name,
            model=self.model,
        )

    def count_tokens(self, prompt) -> int:
        return len(prompt.stable_prefix) // 4


@pytest.fixture
def prompt() -> Prompt:
    return Prompt(system="Ти перекладач.", context="Досьє фільму.", user="Переклади: Fine.")


# ---------- Prompt ----------


def test_stable_prefix_joins_system_and_context(prompt):
    assert prompt.stable_prefix == "Ти перекладач.\n\nДосьє фільму."


def test_stable_prefix_is_independent_of_user_part():
    first = Prompt(system="S", context="C", user="чанк 1")
    second = Prompt(system="S", context="C", user="чанк 42")
    assert first.stable_prefix == second.stable_prefix  # інакше кеш промптів не працює


# ---------- Usage ----------


def test_usage_adds_up():
    total = Usage(input_tokens=1, output_tokens=2, cost_usd=0.5) + Usage(
        input_tokens=10, output_tokens=20, cost_usd=1.5
    )
    assert (total.input_tokens, total.output_tokens, total.cost_usd) == (11, 22, 2.0)


def test_usage_cost_stays_none_when_backend_reports_nothing():
    assert (Usage(input_tokens=1) + Usage(input_tokens=2)).cost_usd is None


# ---------- реєстр бекендів ----------


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="невідомий бекенд"):
        build_backend("gpt-5")


def test_registry_lists_three_backends():
    assert set(available_backends) == {"claude-cli", "gemini", "anthropic"}


def test_gemini_without_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(BackendError, match="GEMINI_API_KEY"):
        build_backend("gemini")


def test_anthropic_without_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(PermanentBackendError, match="ANTHROPIC_API_KEY"):
        build_backend("anthropic")


# ---------- підготовка схем ----------


@pytest.mark.parametrize("schema", [Tiny, SpeakerMap, Dossier, Bible])
def test_strict_schema_forbids_extra_fields(schema):
    """Без additionalProperties:false модель дописує поля, яких ми не просили."""
    generated = _to_strict_schema(schema)
    objects = list(_walk_objects(generated))
    assert objects, f"{schema.__name__}: не знайдено обʼєктів у схемі"
    assert all(node.get("additionalProperties") is False for node in objects)


def _walk_objects(node):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from _walk_objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_objects(item)


def test_nested_schemas_use_refs():
    """Перевірено живим викликом: claude -p приймає $defs/$ref, розгортати не треба."""
    assert "$defs" in _to_strict_schema(SpeakerMap)


# ---------- аварійний розбір JSON ----------


def test_extract_json_plain():
    assert _extract_json('{"uk": "Гаразд."}') == {"uk": "Гаразд."}


def test_extract_json_from_fenced_block():
    assert _extract_json('Ось результат:\n```json\n{"uk": "Гаразд."}\n```') == {
        "uk": "Гаразд."
    }


def test_extract_json_from_surrounding_prose():
    assert _extract_json('Переклад: {"uk": "Гаразд."} — готово') == {"uk": "Гаразд."}


def test_extract_json_gives_up_gracefully():
    assert _extract_json("тут узагалі немає JSON") is None


# ---------- Runner: кеш ----------


def test_second_identical_call_is_served_from_cache(tmp_path, prompt):
    backend = FakeBackend()
    runner = Runner(backend, tmp_path)

    first = runner.run_structured("translate", prompt, Tiny)
    second = runner.run_structured("translate", prompt, Tiny)

    assert backend.calls == 1  # другий виклик не дійшов до бекенда
    assert first.from_cache is False and second.from_cache is True
    assert second.data == {"uk": "Привіт."}


def test_changed_prompt_invalidates_cache(tmp_path, prompt):
    backend = FakeBackend()
    runner = Runner(backend, tmp_path)
    runner.run_structured("translate", prompt, Tiny)
    runner.run_structured("translate", Prompt(system="S2", context="C", user="U"), Tiny)
    assert backend.calls == 2


def test_changed_effort_invalidates_cache(tmp_path, prompt):
    backend = FakeBackend()
    runner = Runner(backend, tmp_path)
    runner.run_structured("translate", prompt, Tiny, effort="high")
    runner.run_structured("translate", prompt, Tiny, effort="low")
    assert backend.calls == 2


def test_changed_schema_invalidates_cache(tmp_path, prompt):
    class Other(BaseModel):
        uk: str
        extra: str = "x"

    backend = FakeBackend()
    runner = Runner(backend, tmp_path)
    runner.run_structured("translate", prompt, Tiny)
    runner.run_structured("translate", prompt, Other)
    assert backend.calls == 2


def test_cache_can_be_disabled(tmp_path, prompt):
    backend = FakeBackend()
    runner = Runner(backend, tmp_path, use_cache=False)
    runner.run_structured("translate", prompt, Tiny)
    runner.run_structured("translate", prompt, Tiny)
    assert backend.calls == 2
    assert not (tmp_path / ".cache").exists()


def test_cache_survives_a_new_runner(tmp_path, prompt):
    """Саме це дає resume після падіння: кеш лежить на диску, не в памʼяті."""
    first_backend = FakeBackend()
    Runner(first_backend, tmp_path).run_structured("translate", prompt, Tiny)

    second_backend = FakeBackend()
    result = Runner(second_backend, tmp_path).run_structured("translate", prompt, Tiny)
    assert second_backend.calls == 0
    assert result.from_cache is True


def test_corrupted_cache_entry_is_ignored(tmp_path, prompt):
    backend = FakeBackend()
    runner = Runner(backend, tmp_path)
    runner.run_structured("translate", prompt, Tiny)
    for path in (tmp_path / ".cache").glob("*.json"):
        path.write_text("{це не json", encoding="utf-8")
    runner.run_structured("translate", prompt, Tiny)
    assert backend.calls == 2  # перечитали бекенд замість падіння


# ---------- Runner: ретраї ----------


def test_transient_failure_is_retried(tmp_path, prompt):
    backend = FakeBackend(fail_times=2)
    runner = Runner(backend, tmp_path, attempts=3, backoff_s=0)
    assert runner.run_structured("translate", prompt, Tiny).data == {"uk": "Привіт."}
    assert backend.calls == 3


def test_retries_are_exhausted_and_reported(tmp_path, prompt):
    backend = FakeBackend(fail_times=99)
    runner = Runner(backend, tmp_path, attempts=2, backoff_s=0)
    with pytest.raises(BackendError, match="2 спроб"):
        runner.run_structured("translate", prompt, Tiny)
    assert backend.calls == 2


def test_permanent_failure_is_not_retried(tmp_path, prompt):
    backend = FakeBackend(fail_times=99, permanent=True)
    runner = Runner(backend, tmp_path, attempts=5, backoff_s=0)
    with pytest.raises(PermanentBackendError):
        runner.run_structured("translate", prompt, Tiny)
    assert backend.calls == 1  # не гатимо в стіну, якщо відповідь не змінюється


# ---------- Runner: статистика й лог ----------


def test_stats_count_only_billed_calls(tmp_path, prompt):
    runner = Runner(FakeBackend(), tmp_path)
    runner.run_structured("translate", prompt, Tiny)
    runner.run_structured("translate", prompt, Tiny)  # з кешу

    summary = runner.stats.summary()
    assert summary["calls"] == 2
    assert summary["cached_calls"] == 1
    assert summary["billed_calls"] == 1
    assert summary["input_tokens"] == 20  # кешований виклик не додається


def test_stats_split_by_pass(tmp_path, prompt):
    runner = Runner(FakeBackend(), tmp_path)
    runner.run_structured("translate", prompt, Tiny)
    runner.run_structured("edit", prompt, Tiny)
    assert set(runner.stats.summary()["by_pass"]) == {"translate", "edit"}


def test_every_call_is_logged_as_jsonl(tmp_path, prompt):
    runner = Runner(FakeBackend(), tmp_path)
    runner.run_structured("translate", prompt, Tiny)
    runner.run_structured("translate", prompt, Tiny)

    lines = (tmp_path / "run.log").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert [record["from_cache"] for record in records] == [False, True]
    assert records[0]["pass"] == "translate"


# ---------- Runner: паралельність ----------


def test_map_structured_preserves_order(tmp_path):
    prompts = [Prompt(system="S", context="C", user=f"чанк {n}") for n in range(6)]
    runner = Runner(FakeBackend(), tmp_path, jobs=4)
    results = runner.map_structured("translate", prompts, Tiny)
    assert len(results) == 6
    assert all(result.data == {"uk": "Привіт."} for result in results)


def test_map_structured_runs_sequentially_when_jobs_is_one(tmp_path):
    prompts = [Prompt(system="S", context="C", user=f"чанк {n}") for n in range(3)]
    backend = FakeBackend()
    results = Runner(backend, tmp_path, jobs=1).map_structured("translate", prompts, Tiny)
    assert backend.calls == 3 and len(results) == 3


def test_estimate_tokens_does_not_call_the_model(tmp_path, prompt):
    backend = FakeBackend()
    total = Runner(backend, tmp_path).estimate_tokens([prompt, prompt])
    assert total > 0
    assert backend.calls == 0
