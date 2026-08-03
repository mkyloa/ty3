"""Протокол провайдера. Проходи пайплайну бачать лише цей інтерфейс.

Промпт свідомо розділений на три частини, а не склеєний в один рядок:

    system  — роль і правила: однакові для всього прогону
    context — досьє фільму + біблія перекладу: однакові для всього прогону
    user    — конкретний чанк: змінюється на кожному виклику

Кеш промптів (де провайдер його має) працює лише на **незмінному префіксі**, тому
`system` + `context` мусять бути побайтово однакові між викликами. Через це вони
й винесені окремо: щоб не було спокуси вклеїти в них таймстемп або номер чанка.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

__all__ = [
    "Prompt",
    "Usage",
    "Completion",
    "Tools",
    "Backend",
    "BackendError",
    "PermanentBackendError",
]


class BackendError(RuntimeError):
    """Провайдер не зміг віддати відповідь (мережа, ліміт, відмова, поламаний JSON)."""


class PermanentBackendError(BackendError):
    """Повтор не допоможе: відмова класифікатора, немає ключа, невідома модель."""


class Tools(str, Enum):
    NONE = "none"
    SEARCH = "search"  # вебпошук для проходу research


@dataclass(frozen=True)
class Prompt:
    system: str
    context: str = ""
    user: str = ""

    @property
    def stable_prefix(self) -> str:
        """Частина, яку провайдер може кешувати."""
        return f"{self.system}\n\n{self.context}".strip()

    def approx_chars(self) -> int:
        return len(self.system) + len(self.context) + len(self.user)


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None

    def __add__(self, other: "Usage") -> "Usage":
        costs = [c for c in (self.cost_usd, other.cost_usd) if c is not None]
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=sum(costs) if costs else None,
        )


@dataclass(frozen=True)
class Completion:
    text: str
    data: dict | None = None  # розібраний структурований вивід
    usage: Usage = field(default_factory=Usage)
    backend: str = ""
    model: str = ""
    from_cache: bool = False


@runtime_checkable
class Backend(Protocol):
    """Мінімум, який мусить уміти будь-який провайдер."""

    name: str
    model: str
    # Версія поведінки адаптера. Входить у ключ дискового кешу, щоб виправлення
    # в самому адаптері (а не в промпті) інвалідувало старі відповіді.
    revision: str

    def complete(
        self,
        prompt: Prompt,
        *,
        effort: str = "high",
        max_tokens: int = 32000,
        tools: Tools = Tools.NONE,
    ) -> Completion:
        """Вільний текстовий вивід."""
        ...

    def complete_structured(
        self,
        prompt: Prompt,
        schema: type[BaseModel],
        *,
        effort: str = "high",
        max_tokens: int = 32000,
    ) -> Completion:
        """Вивід, що валідується схемою. `Completion.data` — розібраний обʼєкт."""
        ...

    def count_tokens(self, prompt: Prompt) -> int:
        """Оцінка вхідних токенів — для `ty3 estimate` без витрат."""
        ...
