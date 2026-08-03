"""Класифікація помилок claude-cli: постійні проти тимчасових.

Регресія з реального прогону: `429 monthly spend limit` (місячний ліміт витрат
орг-акаунта) трактувався як тимчасовий збій і повторювався тричі — марно, бо ця
відмова не минає ні за секунди, ні за 5 годин, а лише за наступний розрахунковий
період або після того, як адмін підніме ліміт.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from ty3.model.backends.base import BackendError, PermanentBackendError, Prompt
from ty3.model.backends.claude_cli import ClaudeCliBackend
from ty3.model.runner import Runner

# Скорочена, але автентична форма відповіді CLI, яку побачив користувач.
SPEND_LIMIT_PAYLOAD = {
    "type": "result",
    "subtype": "success",
    "is_error": True,
    "api_error_status": 429,
    "result": (
        "You've hit your org's monthly spend limit · run /usage-credits to ask "
        "your admin for a higher limit"
    ),
    "usage": {"input_tokens": 0, "output_tokens": 0},
    "total_cost_usd": 0,
}

ORDINARY_RATE_LIMIT_PAYLOAD = {
    "type": "result",
    "subtype": "error",
    "is_error": True,
    "api_error_status": 429,
    "result": "Rate limited, please retry shortly",
}


class Tiny(BaseModel):
    uk: str


class FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str, stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def backend() -> ClaudeCliBackend:
    return ClaudeCliBackend(model="claude-opus-5")


@pytest.fixture
def prompt() -> Prompt:
    return Prompt(system="S", context="C", user="U")


def _stub_run(monkeypatch, returncode: int, payload: dict, stderr: str = "") -> dict:
    calls = {"n": 0}

    def fake_run(*_args, **_kwargs):
        calls["n"] += 1
        return FakeCompletedProcess(returncode, json.dumps(payload), stderr)

    monkeypatch.setattr("ty3.model.backends.claude_cli.subprocess.run", fake_run)
    return calls


# ================= класифікація =================


def test_monthly_spend_limit_is_a_permanent_error(monkeypatch, backend, prompt):
    _stub_run(monkeypatch, 1, SPEND_LIMIT_PAYLOAD)
    with pytest.raises(PermanentBackendError, match="spend limit"):
        backend.complete_structured(prompt, Tiny)


def test_permanent_error_message_is_human_readable(monkeypatch, backend, prompt):
    """Повідомлення має пояснювати причину, а не тільки код і обрізаний JSON."""
    _stub_run(monkeypatch, 1, SPEND_LIMIT_PAYLOAD)
    with pytest.raises(PermanentBackendError) as caught:
        backend.complete_structured(prompt, Tiny)
    assert "usage-credits" in str(caught.value)


def test_ordinary_429_without_spend_limit_marker_stays_retryable(monkeypatch, backend, prompt):
    """Звичайний rate limit минає сам — його варто повторювати, на відміну від місячного ліміту."""
    _stub_run(monkeypatch, 1, ORDINARY_RATE_LIMIT_PAYLOAD)
    with pytest.raises(BackendError) as caught:
        backend.complete_structured(prompt, Tiny)
    assert not isinstance(caught.value, PermanentBackendError)


def test_marker_check_is_case_insensitive(monkeypatch, backend, prompt):
    payload = {**SPEND_LIMIT_PAYLOAD, "result": "MONTHLY SPEND LIMIT reached"}
    _stub_run(monkeypatch, 1, payload)
    with pytest.raises(PermanentBackendError):
        backend.complete_structured(prompt, Tiny)


def test_non_429_error_is_never_classified_as_permanent(monkeypatch, backend, prompt):
    """Маркер сам собою не рішення — 500-ка зі схожим текстом усе одно тимчасова."""
    payload = {**SPEND_LIMIT_PAYLOAD, "api_error_status": 500}
    _stub_run(monkeypatch, 1, payload)
    with pytest.raises(BackendError) as caught:
        backend.complete_structured(prompt, Tiny)
    assert not isinstance(caught.value, PermanentBackendError)


def test_unparseable_stdout_on_error_falls_back_to_generic_message(monkeypatch, backend, prompt):
    """Захисний шлях: не-JSON вивід не має ламати класифікацію, лише пропускати її."""
    def fake_run(*_args, **_kwargs):
        return FakeCompletedProcess(1, "не json взагалі", "boom")

    monkeypatch.setattr("ty3.model.backends.claude_cli.subprocess.run", fake_run)
    with pytest.raises(BackendError) as caught:
        backend.complete_structured(prompt, Tiny)
    assert not isinstance(caught.value, PermanentBackendError)


def test_permanent_error_detected_even_when_exit_code_is_zero(monkeypatch, backend, prompt):
    """CLI іноді повертає 0, але з `is_error: true` в тілі — перевіряємо і цей шлях."""
    _stub_run(monkeypatch, 0, SPEND_LIMIT_PAYLOAD)
    with pytest.raises(PermanentBackendError):
        backend.complete_structured(prompt, Tiny)


# ================= інтеграція з Runner: більше не марнує спроби =================


def test_runner_does_not_retry_a_permanent_error(monkeypatch, tmp_path, backend, prompt):
    calls = _stub_run(monkeypatch, 1, SPEND_LIMIT_PAYLOAD)
    runner = Runner(backend, tmp_path, attempts=3, backoff_s=0)

    with pytest.raises(PermanentBackendError):
        runner.run_structured("bible", prompt, Tiny)

    assert calls["n"] == 1, "постійна помилка не мала повторюватись — це марна трата часу"


def test_runner_still_retries_an_ordinary_transient_error(monkeypatch, tmp_path, backend, prompt):
    """Регресія навпаки: фікс не має зробити тимчасові помилки нерепрайзними."""
    calls = _stub_run(monkeypatch, 1, ORDINARY_RATE_LIMIT_PAYLOAD)
    runner = Runner(backend, tmp_path, attempts=3, backoff_s=0)

    with pytest.raises(BackendError):
        runner.run_structured("bible", prompt, Tiny)

    assert calls["n"] == 3, "тимчасова помилка мусить повторюватися за attempts"
