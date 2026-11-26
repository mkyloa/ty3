"""Бекенд через локальний `claude -p` — працює на підписці Claude Code, ключ не потрібен.

Заміряно на живих викликах: дефолтний системний промпт Claude Code додає ~26 тис.
токенів на **кожен** виклик. Тому адаптер завжди подає власний `--system-prompt`,
вимикає динамічні секції та інструменти — накладні витрати падають до ~6 тис.
Це не оптимізація «про запас»: на 80 викликах фільму різниця становить порядок.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from pydantic import BaseModel, ValidationError

from .base import BackendError, Completion, PermanentBackendError, Prompt, Tools, Usage

__all__ = ["ClaudeCliBackend"]

_SEARCH_TOOLS = "WebSearch,WebFetch"

# Присутнє в тексті відповіді при жорсткому місячному ліміті витрат орг-акаунта.
# На відміну від звичайного 429 (спрацьовує роллінговий ліміт і минає з часом),
# ця відмова не мине ні за 5 годин, ні за три повтори — лише за наступний
# розрахунковий період або після того, як адмін підніме ліміт.
_SPEND_LIMIT_MARKER = "spend limit"


class ClaudeCliBackend:
    name = "claude-cli"
    # Бампай, коли змінюється поведінка адаптера так, що старі кешовані
    # відповіді стають недійсними.
    #   rev2: додано --allowed-tools, без якого вебпошук мовчки відхилявся,
    #         і прохід research працював на памʼяті моделі;
    #   rev3: додано --max-budget-usd як запобіжник від нескінченного пошуку.
    revision = "3"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        executable: str = "claude",
        timeout_s: int = 900,
        search_budget_usd: float = 1.5,
    ) -> None:
        if shutil.which(executable) is None:
            raise BackendError(
                f"не знайдено {executable!r} у PATH. Встанови Claude Code або "
                f"вибери інший бекенд у config.toml"
            )
        self.model = model
        self.executable = executable
        self.timeout_s = timeout_s
        self.search_budget_usd = search_budget_usd

    # ---------- публічний інтерфейс ----------

    def complete(
        self,
        prompt: Prompt,
        *,
        effort: str = "high",
        max_tokens: int = 32000,
        tools: Tools = Tools.NONE,
    ) -> Completion:
        payload = self._invoke(prompt, effort=effort, tools=tools)
        return self._to_completion(payload, data=None)

    def complete_structured(
        self,
        prompt: Prompt,
        schema: type[BaseModel],
        *,
        effort: str = "high",
        max_tokens: int = 32000,
    ) -> Completion:
        json_schema = _to_strict_schema(schema)
        payload = self._invoke(
            prompt, effort=effort, tools=Tools.NONE, json_schema=json_schema
        )

        raw = payload.get("structured_output")
        if raw is None:
            # Схема не спрацювала — пробуємо витягти JSON із тексту, бо переклад
            # цілого чанка коштує задорого, щоб викидати його через формат.
            raw = _extract_json(payload.get("result", ""))
        if raw is None:
            raise BackendError(
                f"{self.name}: не повернув структурованого виводу за схемою "
                f"{schema.__name__}"
            )
        try:
            validated = schema.model_validate(raw)
        except ValidationError as exc:
            raise BackendError(f"{self.name}: вивід не пройшов схему {schema.__name__}: {exc}") from exc
        return self._to_completion(payload, data=validated.model_dump())

    def count_tokens(self, prompt: Prompt) -> int:
        """CLI не має ендпойнта підрахунку — оцінка за символами.

        ~3.3 символи на токен для змішаного EN/UK тексту; плюс заміряні ~6 тис.
        накладних витрат Claude Code на виклик.
        """
        return int(prompt.approx_chars() / 3.3) + 6000

    # ---------- внутрішнє ----------

    def _build_command(
        self,
        prompt: Prompt,
        *,
        effort: str,
        tools: Tools,
        json_schema: dict[str, Any] | None,
    ) -> list[str]:
        command = [
            self.executable,
            "-p",
            prompt.user,
            "--model",
            self.model,
            "--output-format",
            "json",
            "--system-prompt",
            prompt.stable_prefix,
            "--exclude-dynamic-system-prompt-sections",
            "--effort",
            effort,
            "--tools",
            _SEARCH_TOOLS if tools is Tools.SEARCH else "",
        ]
        if tools is Tools.SEARCH:
            # `--tools` лише робить інструменти доступними; дозвіл на виклик дає
            # окремий `--allowed-tools`. Без нього CLI відхиляє WebSearch і
            # модель тихо відповідає з памʼяті — перевірено.
            command += ["--allowed-tools", _SEARCH_TOOLS]
            # Запобіжник від нескінченного пошуку. Для маловідомого чи вигаданого
            # фільму модель шукає раз за разом, не знаходячи нічого; без ліміту
            # це впирається в 15-хвилинний таймаут і три марні спроби.
            command += ["--max-budget-usd", str(self.search_budget_usd)]
        if json_schema is not None:
            command += ["--json-schema", json.dumps(json_schema, sort_keys=True)]
        return command

    def _invoke(
        self,
        prompt: Prompt,
        *,
        effort: str,
        tools: Tools,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        command = self._build_command(
            prompt, effort=effort, tools=tools, json_schema=json_schema
        )
        try:
            done = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendError(
                f"{self.name}: виклик не завершився за {self.timeout_s} с"
            ) from exc

        if done.returncode != 0:
            # Навіть при ненульовому коду виходу claude CLI зазвичай пише JSON
            # у stdout — намагаємось розпізнати постійну відмову (місячний ліміт
            # витрат), перш ніж здатися на загальну помилку з сирим текстом.
            parsed = _try_parse_json(done.stdout)
            if parsed is not None:
                reason = _permanent_reason(parsed)
                if reason is not None:
                    raise PermanentBackendError(f"{self.name}: {reason}")
            detail = (done.stderr or done.stdout or "").strip()[:800]
            raise BackendError(f"{self.name}: код виходу {done.returncode}: {detail}")

        try:
            payload = json.loads(done.stdout)
        except json.JSONDecodeError as exc:
            raise BackendError(
                f"{self.name}: вивід не є JSON: {done.stdout[:400]!r}"
            ) from exc

        if payload.get("is_error") or payload.get("subtype") != "success":
            reason = _permanent_reason(payload)
            if reason is not None:
                raise PermanentBackendError(f"{self.name}: {reason}")
            raise BackendError(
                f"{self.name}: {payload.get('subtype')} / "
                f"{payload.get('api_error_status')}: {str(payload.get('result'))[:400]}"
            )
        return payload

    def _to_completion(self, payload: dict[str, Any], data: dict | None) -> Completion:
        raw_usage = payload.get("usage") or {}
        return Completion(
            text=payload.get("result") or "",
            data=data,
            usage=Usage(
                input_tokens=int(raw_usage.get("input_tokens") or 0),
                output_tokens=int(raw_usage.get("output_tokens") or 0),
                cache_read_tokens=int(raw_usage.get("cache_read_input_tokens") or 0),
                cache_write_tokens=int(raw_usage.get("cache_creation_input_tokens") or 0),
                cost_usd=payload.get("total_cost_usd"),
            ),
            backend=self.name,
            model=self.model,
        )


def _try_parse_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _permanent_reason(payload: dict[str, Any]) -> str | None:
    """Якщо помилка постійна (не минає з часом чи повтором), повертає причину.

    Спостережено на реальному прогоні: `429` з текстом про місячний ліміт витрат
    орг-акаунта. Це відрізняється від звичайного rate limit, який минає сам —
    тут повтори (і навіть очікування кількох годин) марні, доки не настане
    наступний розрахунковий період або адмін не підніме ліміт.
    """
    if payload.get("api_error_status") != 429:
        return None
    result_text = str(payload.get("result") or "")
    if _SPEND_LIMIT_MARKER in result_text.lower():
        return result_text.strip()
    return None


def _to_strict_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema з `additionalProperties: false` на кожному обʼєкті.

    Без цього структурований вивід дозволяє дописувати поля, і в дані просочується
    те, чого ми не просили.
    """
    generated = schema.model_json_schema()
    _harden(generated)
    return generated


def _harden(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" and "additionalProperties" not in node:
            node["additionalProperties"] = False
        for value in node.values():
            _harden(value)
    elif isinstance(node, list):
        for item in node:
            _harden(item)


def _extract_json(text: str) -> Any | None:
    """Виловлює JSON-обʼєкт із тексту: спершу цілий рядок, потім блок у ```."""
    text = text.strip()
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _json_candidates(text: str):
    yield text
    if "```" in text:
        for block in text.split("```")[1::2]:
            yield block.removeprefix("json").strip()
    first, last = text.find("{"), text.rfind("}")
    if 0 <= first < last:
        yield text[first : last + 1]
