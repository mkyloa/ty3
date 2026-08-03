"""Обгортка над бекендом: дисковий кеш, ретраї, лог витрат, паралельність.

Дисковий кеш — не оптимізація, а вимога до робочого процесу. Прогін фільму — це
десятки викликів; якщо на 70-му впала мережа або виявився поганий промпт, платити
за перші 69 удруге не потрібно. Ключ кешу включає все, що впливає на результат,
тому зміна промпта автоматично інвалідує тільки залежні кроки.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar

from pydantic import BaseModel

from .backends.base import (
    Backend,
    BackendError,
    Completion,
    PermanentBackendError,
    Prompt,
    Tools,
    Usage,
)

__all__ = ["Runner", "RunStats"]

T = TypeVar("T")


class RunStats:
    """Накопичує витрати за прогін."""

    def __init__(self) -> None:
        self.total = Usage()
        self.calls = 0
        self.cached_calls = 0
        self.by_pass: dict[str, Usage] = {}

    def add(self, pass_name: str, usage: Usage, *, from_cache: bool) -> None:
        self.calls += 1
        if from_cache:
            self.cached_calls += 1
            return
        self.total = self.total + usage
        self.by_pass[pass_name] = self.by_pass.get(pass_name, Usage()) + usage

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "billed_calls": self.calls - self.cached_calls,
            "input_tokens": self.total.input_tokens,
            "output_tokens": self.total.output_tokens,
            "cache_read_tokens": self.total.cache_read_tokens,
            "cache_write_tokens": self.total.cache_write_tokens,
            "cost_usd": self.total.cost_usd,
            "by_pass": {name: asdict(usage) for name, usage in self.by_pass.items()},
        }


class Runner:
    def __init__(
        self,
        backend: Backend,
        work_dir: Path,
        *,
        use_cache: bool = True,
        attempts: int = 3,
        backoff_s: float = 2.0,
        jobs: int = 1,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self.backend = backend
        self.cache_dir = Path(work_dir) / ".cache"
        self.log_path = Path(work_dir) / "run.log"
        self.use_cache = use_cache
        self.attempts = max(1, attempts)
        self.backoff_s = backoff_s
        self.jobs = max(1, jobs)
        self.stats = RunStats()
        self._on_event = on_event or (lambda _message: None)

    # ---------- публічний інтерфейс ----------

    def run(
        self,
        pass_name: str,
        prompt: Prompt,
        *,
        effort: str = "high",
        max_tokens: int = 32000,
        tools: Tools = Tools.NONE,
    ) -> Completion:
        key = self._key(pass_name, prompt, effort=effort, tools=tools.value, schema=None)
        return self._execute(
            pass_name,
            key,
            lambda: self.backend.complete(
                prompt, effort=effort, max_tokens=max_tokens, tools=tools
            ),
        )

    def run_structured(
        self,
        pass_name: str,
        prompt: Prompt,
        schema: type[BaseModel],
        *,
        effort: str = "high",
        max_tokens: int = 32000,
    ) -> Completion:
        key = self._key(
            pass_name, prompt, effort=effort, tools="none", schema=schema
        )
        return self._execute(
            pass_name,
            key,
            lambda: self.backend.complete_structured(
                prompt, schema, effort=effort, max_tokens=max_tokens
            ),
        )

    def map_structured(
        self,
        pass_name: str,
        prompts: Sequence[Prompt],
        schema: type[BaseModel],
        *,
        effort: str = "high",
        max_tokens: int = 32000,
    ) -> list[Completion]:
        """Паралельно (за jobs>1) виконує однотипні виклики, зберігаючи порядок."""
        if self.jobs == 1 or len(prompts) <= 1:
            return [
                self.run_structured(
                    pass_name, prompt, schema, effort=effort, max_tokens=max_tokens
                )
                for prompt in prompts
            ]
        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            futures = [
                pool.submit(
                    self.run_structured,
                    pass_name,
                    prompt,
                    schema,
                    effort=effort,
                    max_tokens=max_tokens,
                )
                for prompt in prompts
            ]
            return [future.result() for future in futures]

    def estimate_tokens(self, prompts: Iterable[Prompt]) -> int:
        return sum(self.backend.count_tokens(prompt) for prompt in prompts)

    # ---------- внутрішнє ----------

    def _key(
        self,
        pass_name: str,
        prompt: Prompt,
        *,
        effort: str,
        tools: str,
        schema: type[BaseModel] | None,
    ) -> str:
        material = {
            "backend": self.backend.name,
            "backend_revision": getattr(self.backend, "revision", "1"),
            "model": self.backend.model,
            "pass": pass_name,
            "effort": effort,
            "tools": tools,
            "schema": schema.__name__ if schema else None,
            "schema_shape": (
                json.dumps(schema.model_json_schema(), sort_keys=True) if schema else None
            ),
            "system": prompt.system,
            "context": prompt.context,
            "user": prompt.user,
        }
        blob = json.dumps(material, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def _cache_path(self, pass_name: str, key: str) -> Path:
        return self.cache_dir / f"{pass_name}-{key}.json"

    def _read_cache(self, pass_name: str, key: str) -> Completion | None:
        path = self._cache_path(pass_name, key)
        if not self.use_cache or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return Completion(
            text=payload.get("text", ""),
            data=payload.get("data"),
            usage=Usage(**payload.get("usage", {})),
            backend=payload.get("backend", ""),
            model=payload.get("model", ""),
            from_cache=True,
        )

    def _write_cache(self, pass_name: str, key: str, completion: Completion) -> None:
        if not self.use_cache:
            return
        path = self._cache_path(pass_name, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "text": completion.text,
            "data": completion.data,
            "usage": asdict(completion.usage),
            "backend": completion.backend,
            "model": completion.model,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)  # атомарно: обірваний запис не отруїть кеш

    def _execute(
        self, pass_name: str, key: str, call: Callable[[], Completion]
    ) -> Completion:
        cached = self._read_cache(pass_name, key)
        if cached is not None:
            self.stats.add(pass_name, cached.usage, from_cache=True)
            self._log(pass_name, cached, duration_s=0.0, from_cache=True)
            return cached

        last_error: BackendError | None = None
        for attempt in range(1, self.attempts + 1):
            started = time.monotonic()
            try:
                completion = call()
            except PermanentBackendError:
                raise
            except BackendError as exc:
                last_error = exc
                self._on_event(
                    f"  спроба {attempt}/{self.attempts} для {pass_name} не вдалася: {exc}"
                )
                if attempt < self.attempts:
                    time.sleep(self.backoff_s * attempt)
                continue

            duration = time.monotonic() - started
            self.stats.add(pass_name, completion.usage, from_cache=False)
            self._write_cache(pass_name, key, completion)
            self._log(pass_name, completion, duration_s=duration, from_cache=False)
            return completion

        raise BackendError(
            f"{pass_name}: {self.attempts} спроб(и) не вдалися; остання помилка: {last_error}"
        )

    def _log(
        self,
        pass_name: str,
        completion: Completion,
        *,
        duration_s: float,
        from_cache: bool,
    ) -> None:
        record = {
            "pass": pass_name,
            "backend": completion.backend or self.backend.name,
            "model": completion.model or self.backend.model,
            "from_cache": from_cache,
            "duration_s": round(duration_s, 2),
            "usage": asdict(completion.usage),
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # лог — не критичний шлях, не валимо через нього прогін
