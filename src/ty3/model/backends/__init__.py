"""Адаптери провайдерів. Реєстр за іменем із config.toml."""

from __future__ import annotations

from .base import Backend, BackendError, Completion, Prompt, Tools, Usage

__all__ = [
    "Backend",
    "BackendError",
    "Completion",
    "Prompt",
    "Tools",
    "Usage",
    "build_backend",
    "available_backends",
]

available_backends = ("claude-cli", "gemini", "anthropic")


def build_backend(name: str, model: str | None = None, **kwargs) -> Backend:
    """Створює бекенд за іменем. Імпорт лінивий — щоб `gemini` не тягнув SDK,
    коли користувач працює через `claude-cli`, і навпаки."""
    if name == "claude-cli":
        from .claude_cli import ClaudeCliBackend

        return ClaudeCliBackend(model=model or "claude-opus-5-5", **kwargs)
    if name == "gemini":
        from .gemini import GeminiBackend

        return GeminiBackend(model=model or "gemini-2.5-pro", **kwargs)
    if name == "anthropic":
        from .anthropic_api import AnthropicBackend

        return AnthropicBackend(model=model or "claude-opus-5-5", **kwargs)
    raise ValueError(
        f"невідомий бекенд {name!r}; доступні: {', '.join(available_backends)}"
    )
