"""pause_turn: серверний вебпошук зупиняється посеред ходу, і бекенд мусить
продовжити його, а не віддати недороблену відповідь."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")

from ty3.model.backends.anthropic_api import _MAX_CONTINUATIONS, AnthropicBackend
from ty3.model.backends.base import BackendError, Prompt, Tools


def _message(stop_reason: str, text: str = "", tokens: int = 10):
    content = [SimpleNamespace(type="server_tool_use", id="srvtoolu_1")]
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=content,
        usage=SimpleNamespace(
            input_tokens=tokens,
            output_tokens=tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


class _Stream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


class _FakeMessages:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        # копія списку: бекенд не має мутувати попередні запити
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return _Stream(self._replies.pop(0))


def _backend(replies) -> tuple[AnthropicBackend, _FakeMessages]:
    backend = AnthropicBackend(api_key="test-key")
    fake = _FakeMessages(replies)
    backend._client = SimpleNamespace(messages=fake)
    return backend, fake


PROMPT = Prompt(system="sys", user="знайди каст")


def test_pause_turn_is_resumed_until_end_turn():
    paused_1 = _message("pause_turn", tokens=10)
    paused_2 = _message("pause_turn", tokens=20)
    done = _message("end_turn", text="готово", tokens=30)
    backend, fake = _backend([paused_1, paused_2, done])

    result = backend.complete(PROMPT, tools=Tools.SEARCH)

    assert result.text == "готово"
    assert len(fake.calls) == 3
    # Перший запит — лише user; далі той самий user + усі паузи, без «Continue»
    assert fake.calls[0]["messages"] == [{"role": "user", "content": "знайди каст"}]
    assert fake.calls[2]["messages"] == [
        {"role": "user", "content": "знайди каст"},
        {"role": "assistant", "content": paused_1.content},
        {"role": "assistant", "content": paused_2.content},
    ]
    # Інструменти й система не змінюються між продовженнями (інакше зламається кеш)
    assert fake.calls[0]["tools"] == fake.calls[2]["tools"]
    assert fake.calls[0]["system"] == fake.calls[2]["system"]


def test_usage_is_summed_across_continuations():
    backend, _ = _backend(
        [_message("pause_turn", tokens=10), _message("end_turn", text="ok", tokens=30)]
    )

    result = backend.complete(PROMPT, tools=Tools.SEARCH)

    assert result.usage.input_tokens == 40
    assert result.usage.output_tokens == 40


def test_endless_pause_turn_fails_instead_of_looping():
    replies = [_message("pause_turn") for _ in range(_MAX_CONTINUATIONS + 1)]
    backend, fake = _backend(replies)

    with pytest.raises(BackendError, match="pause_turn"):
        backend.complete(PROMPT, tools=Tools.SEARCH)
    assert len(fake.calls) == _MAX_CONTINUATIONS + 1


def test_no_pause_means_single_call():
    backend, fake = _backend([_message("end_turn", text="одразу")])

    assert backend.complete(PROMPT).text == "одразу"
    assert len(fake.calls) == 1
