"""Побудова команди для `claude -p`.

Тести тут — не формальність. Помилка в одному прапорці не валить прогін: він
успішно завершується, віддає правдоподібний результат і мовчки працює на памʼяті
моделі замість джерел. Саме так і сталося з `--allowed-tools`.
"""

from __future__ import annotations

import json

import pytest

from ty3.model.backends.base import Prompt, Tools
from ty3.model.backends.claude_cli import ClaudeCliBackend
from ty3.model.runner import Runner
from ty3.model.schemas import SpeakerMap


@pytest.fixture
def backend() -> ClaudeCliBackend:
    return ClaudeCliBackend(model="claude-opus-5")


@pytest.fixture
def prompt() -> Prompt:
    return Prompt(system="Роль.", context="Досьє.", user="Переклади це.")


def _pairs(command: list[str]) -> dict[str, str]:
    return {
        command[i]: command[i + 1]
        for i in range(len(command) - 1)
        if command[i].startswith("--")
    }


def test_search_requires_allowed_tools(backend, prompt):
    """Регресія: `--tools` лише вмикає наявність інструментів.

    Без `--allowed-tools` CLI відхиляє WebSearch (`permission_denials`), і прохід
    research тихо працює без вебу — перевірено живими викликами.
    """
    command = backend._build_command(
        prompt, effort="high", tools=Tools.SEARCH, json_schema=None
    )
    pairs = _pairs(command)
    assert pairs["--tools"] == "WebSearch,WebFetch"
    assert pairs["--allowed-tools"] == "WebSearch,WebFetch"


def test_search_is_capped_by_budget(prompt):
    """Регресія: для вигаданого фільму модель шукає без кінця.

    Без ліміту це впирається в 15-хвилинний таймаут і три марні спроби —
    спостережено на живому прогоні.
    """
    backend = ClaudeCliBackend(model="claude-opus-5", search_budget_usd=0.75)
    command = backend._build_command(
        prompt, effort="high", tools=Tools.SEARCH, json_schema=None
    )
    assert _pairs(command)["--max-budget-usd"] == "0.75"


def test_budget_cap_only_applies_to_search_calls(backend, prompt):
    """Переклад не має ліміту: обрізати його на середині чанка — гірше за витрати."""
    command = backend._build_command(
        prompt, effort="high", tools=Tools.NONE, json_schema=None
    )
    assert "--max-budget-usd" not in command


def test_without_search_tools_are_disabled_entirely(backend, prompt):
    """Порожній `--tools` знімає ~20 тис. токенів опису інструментів із промпта."""
    command = backend._build_command(
        prompt, effort="high", tools=Tools.NONE, json_schema=None
    )
    assert _pairs(command)["--tools"] == ""
    assert "--allowed-tools" not in command


def test_system_prompt_replaces_the_default_one(backend, prompt):
    """Заміряно: дефолтний системний промпт Claude Code — це ~26 тис. токенів
    накладних витрат на кожен виклик, або $0.169 за тривіальний запит."""
    command = backend._build_command(
        prompt, effort="high", tools=Tools.NONE, json_schema=None
    )
    assert _pairs(command)["--system-prompt"] == "Роль.\n\nДосьє."
    assert "--exclude-dynamic-system-prompt-sections" in command
    assert "--append-system-prompt" not in command  # саме заміна, не додавання


def test_effort_and_model_are_passed_through(backend, prompt):
    command = backend._build_command(
        prompt, effort="low", tools=Tools.NONE, json_schema=None
    )
    pairs = _pairs(command)
    assert pairs["--effort"] == "low"
    assert pairs["--model"] == "claude-opus-5"
    assert pairs["--output-format"] == "json"


def test_user_part_goes_to_print_flag(backend, prompt):
    command = backend._build_command(
        prompt, effort="high", tools=Tools.NONE, json_schema=None
    )
    assert command[1] == "-p"
    assert command[2] == "Переклади це."  # без стабільного префікса


def test_schema_is_serialised_deterministically(backend, prompt):
    """Ключ кешу залежить від байтів промпта, тому порядок ключів мусить бути стабільний."""
    schema = {"type": "object", "properties": {"b": {"type": "string"}, "a": {"type": "string"}}}
    first = backend._build_command(prompt, effort="high", tools=Tools.NONE, json_schema=schema)
    second = backend._build_command(prompt, effort="high", tools=Tools.NONE, json_schema=schema)
    assert first == second
    assert _pairs(first)["--json-schema"] == json.dumps(schema, sort_keys=True)


def test_revision_participates_in_cache_key(tmp_path, prompt):
    """Виправлення в адаптері мусить інвалідувати кеш, а не лишати старі відповіді."""

    class Fake:
        name = "fake"
        model = "m"
        revision = "1"

        def __init__(self):
            self.calls = 0

        def complete_structured(self, prompt, schema, **kwargs):
            from ty3.model.backends.base import Completion

            self.calls += 1
            return Completion(text="{}", data={"cues": []}, backend=self.name, model=self.model)

    backend = Fake()
    runner = Runner(backend, tmp_path)
    runner.run_structured("speakers", prompt, SpeakerMap)

    backend.revision = "2"  # виправили поведінку адаптера
    runner.run_structured("speakers", prompt, SpeakerMap)
    assert backend.calls == 2, "після бампу revision кеш мусить стати недійсним"
