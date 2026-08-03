"""Спільний тип знахідки для всіх лінтерів."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["Issue", "Severity"]


class Severity(str, Enum):
    ERROR = "error"  # треба виправити: порушує норму або матрицю
    WARNING = "warning"  # варте погляду людини
    INFO = "info"  # довідково, у звіт


@dataclass(frozen=True)
class Issue:
    cue: int
    code: str
    severity: Severity
    message: str
    detail: str | None = None

    @property
    def needs_fix(self) -> bool:
        """Чи віддавати цю репліку на точкове виправлення моделі."""
        return self.severity is Severity.ERROR

    def __str__(self) -> str:
        tail = f" ({self.detail})" if self.detail else ""
        return f"[{self.severity.value}] #{self.cue} {self.code}: {self.message}{tail}"
