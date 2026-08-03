"""Бекенд Gemini (google-genai). Потрібен безкоштовний ключ із AI Studio.

Дві переваги над CLI-бекендом для цього завдання: немає накладних витрат на
системний промпт агента, і `response_schema` приймає pydantic-модель напряму.
Обмеження, спільне з Anthropic: структурований вивід **не поєднується** з
grounding-пошуком в одному виклику — тому прохід research і так розбитий на два.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ValidationError

from .base import BackendError, Completion, Prompt, Tools, Usage

__all__ = ["GeminiBackend"]

# Gemini керує глибиною роздумів рівнем, а не токенами
_EFFORT_TO_LEVEL = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


class GeminiBackend:
    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        api_key: str | None = None,
        timeout_s: int = 900,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - залежить від оточення
            raise BackendError(
                "не встановлено google-genai: pip install google-genai"
            ) from exc

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        if not key:
            raise BackendError(
                "не задано GEMINI_API_KEY. Безкоштовний ключ: "
                "https://aistudio.google.com/apikey"
            )
        from google.genai import types

        self._types = types
        self._client = genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=timeout_s * 1000)
        )
        self.model = model

    # ---------- публічний інтерфейс ----------

    def complete(
        self,
        prompt: Prompt,
        *,
        effort: str = "high",
        max_tokens: int = 32000,
        tools: Tools = Tools.NONE,
    ) -> Completion:
        config = self._config(prompt, effort=effort, max_tokens=max_tokens, tools=tools)
        response = self._generate(prompt, config)
        return self._to_completion(response, data=None)

    def complete_structured(
        self,
        prompt: Prompt,
        schema: type[BaseModel],
        *,
        effort: str = "high",
        max_tokens: int = 32000,
    ) -> Completion:
        config = self._config(
            prompt,
            effort=effort,
            max_tokens=max_tokens,
            tools=Tools.NONE,
            schema=schema,
        )
        response = self._generate(prompt, config)

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return self._to_completion(response, data=parsed.model_dump())

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise BackendError(f"{self.name}: порожня відповідь для {schema.__name__}")
        try:
            validated = schema.model_validate_json(text)
        except ValidationError as exc:
            raise BackendError(
                f"{self.name}: вивід не пройшов схему {schema.__name__}: {exc}"
            ) from exc
        return self._to_completion(response, data=validated.model_dump())

    def count_tokens(self, prompt: Prompt) -> int:
        try:
            result = self._client.models.count_tokens(
                model=self.model,
                contents=f"{prompt.stable_prefix}\n\n{prompt.user}",
            )
        except Exception:  # мережа/квота — не валимо оцінку через це
            return int(prompt.approx_chars() / 3.3)
        return int(getattr(result, "total_tokens", 0) or 0)

    # ---------- внутрішнє ----------

    def _config(
        self,
        prompt: Prompt,
        *,
        effort: str,
        max_tokens: int,
        tools: Tools,
        schema: type[BaseModel] | None = None,
    ):
        types = self._types
        kwargs: dict[str, Any] = {
            "system_instruction": prompt.stable_prefix,
            "max_output_tokens": max_tokens,
            "thinking_config": types.ThinkingConfig(
                thinking_level=_EFFORT_TO_LEVEL.get(effort, "high")
            ),
        }
        if schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = schema
        if tools is Tools.SEARCH:
            kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        return types.GenerateContentConfig(**kwargs)

    def _generate(self, prompt: Prompt, config):
        try:
            return self._client.models.generate_content(
                model=self.model, contents=prompt.user, config=config
            )
        except Exception as exc:
            raise BackendError(f"{self.name}: {type(exc).__name__}: {exc}") from exc

    def _to_completion(self, response: Any, data: dict | None) -> Completion:
        meta = getattr(response, "usage_metadata", None)
        return Completion(
            text=getattr(response, "text", None) or "",
            data=data,
            usage=Usage(
                input_tokens=int(getattr(meta, "prompt_token_count", 0) or 0),
                output_tokens=int(getattr(meta, "candidates_token_count", 0) or 0),
                cache_read_tokens=int(getattr(meta, "cached_content_token_count", 0) or 0),
            ),
            backend=self.name,
            model=self.model,
        )
