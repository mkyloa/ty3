"""Бекенд Anthropic API. Потрібен платний ключ ANTHROPIC_API_KEY.

Єдиний бекенд із повним контролем кешу промптів (`ttl: 1h`) і Batch API (−50%),
тому найдешевший на повнометражних фільмах. **Не перевірений живими викликами** —
у проєкті немає ключа; перед першим серйозним прогоном варто зробити `estimate`
і один короткий файл.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ValidationError

from .base import BackendError, Completion, PermanentBackendError, Prompt, Tools, Usage

__all__ = ["AnthropicBackend"]

_WEB_SEARCH = {"type": "web_search_20260209", "name": "web_search"}
_WEB_FETCH = {"type": "web_fetch_20260209", "name": "web_fetch"}


class AnthropicBackend:
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        cache_ttl: str = "1h",
        timeout_s: int = 900,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise BackendError("не встановлено anthropic: pip install anthropic") from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise PermanentBackendError(
                "не задано ANTHROPIC_API_KEY. Або візьми інший бекенд: "
                "backend = \"claude-cli\" у config.toml"
            )
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=key, timeout=timeout_s)
        self.model = model
        self.cache_ttl = cache_ttl

    # ---------- публічний інтерфейс ----------

    def complete(
        self,
        prompt: Prompt,
        *,
        effort: str = "high",
        max_tokens: int = 32000,
        tools: Tools = Tools.NONE,
    ) -> Completion:
        message = self._stream(
            prompt,
            effort=effort,
            max_tokens=max_tokens,
            tools=[_WEB_SEARCH, _WEB_FETCH] if tools is Tools.SEARCH else None,
        )
        return self._to_completion(message, data=None)

    def complete_structured(
        self,
        prompt: Prompt,
        schema: type[BaseModel],
        *,
        effort: str = "high",
        max_tokens: int = 32000,
    ) -> Completion:
        message = self._stream(
            prompt,
            effort=effort,
            max_tokens=max_tokens,
            output_format={"type": "json_schema", "schema": schema.model_json_schema()},
        )
        text = self._text_of(message)
        try:
            validated = schema.model_validate_json(text)
        except ValidationError as exc:
            raise BackendError(
                f"{self.name}: вивід не пройшов схему {schema.__name__}: {exc}"
            ) from exc
        return self._to_completion(message, data=validated.model_dump())

    def count_tokens(self, prompt: Prompt) -> int:
        try:
            result = self._client.messages.count_tokens(
                model=self.model,
                system=self._system_blocks(prompt),
                messages=[{"role": "user", "content": prompt.user or "."}],
            )
        except Exception:
            return int(prompt.approx_chars() / 3.3)
        return int(result.input_tokens)

    # ---------- внутрішнє ----------

    def _system_blocks(self, prompt: Prompt) -> list[dict[str, Any]]:
        """Стабільний префікс одним блоком із точкою кешування на кінці.

        Кеш — це збіг **префікса байт у байт**, тому тут не має бути нічого
        змінного: ні часу, ні номера чанка, ні несортованого JSON.
        """
        return [
            {
                "type": "text",
                "text": prompt.stable_prefix,
                "cache_control": {"type": "ephemeral", "ttl": self.cache_ttl},
            }
        ]

    def _stream(
        self,
        prompt: Prompt,
        *,
        effort: str,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        output_format: dict[str, Any] | None = None,
    ):
        output_config: dict[str, Any] = {"effort": effort}
        if output_format is not None:
            output_config["format"] = output_format

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": self._system_blocks(prompt),
            "messages": [{"role": "user", "content": prompt.user}],
            "output_config": output_config,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            # Стрімимо завжди: при великому max_tokens нестрімовий запит
            # впирається в HTTP-таймаут SDK.
            with self._client.messages.stream(**kwargs) as stream:
                message = stream.get_final_message()
        except self._anthropic.APIStatusError as exc:
            raise BackendError(f"{self.name}: HTTP {exc.status_code}: {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise BackendError(f"{self.name}: мережа: {exc}") from exc

        # Перевіряємо відмову ДО читання content — при відмові він порожній
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise PermanentBackendError(
                f"{self.name}: запит відхилено класифікатором "
                f"({getattr(details, 'category', 'без категорії')})"
            )
        if message.stop_reason == "max_tokens":
            raise BackendError(
                f"{self.name}: обірвано на max_tokens={max_tokens} — підніми ліміт"
            )
        return message

    @staticmethod
    def _text_of(message: Any) -> str:
        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        ).strip()

    def _to_completion(self, message: Any, data: dict | None) -> Completion:
        usage = message.usage
        return Completion(
            text=self._text_of(message),
            data=data,
            usage=Usage(
                input_tokens=int(usage.input_tokens or 0),
                output_tokens=int(usage.output_tokens or 0),
                cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
                cache_write_tokens=int(
                    getattr(usage, "cache_creation_input_tokens", 0) or 0
                ),
            ),
            backend=self.name,
            model=self.model,
        )
