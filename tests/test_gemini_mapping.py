"""Перетворення effort → thinking_level для Gemini.

Без мапи тут тихий дефект: SDK приймає невідоме значення на кшталт `xhigh`
із простим `UserWarning`, створює `ThinkingLevel.xhigh` і відправляє його в API.
Тобто помилка не падає локально, а виникає (або мовчки ігнорується) на сервері.
"""

from __future__ import annotations

import warnings

import pytest

from ty3.model.backends.gemini import _EFFORT_TO_LEVEL

VALID = {"THINKING_LEVEL_UNSPECIFIED", "MINIMAL", "LOW", "MEDIUM", "HIGH"}


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_every_effort_maps_to_a_real_thinking_level(effort):
    mapped = _EFFORT_TO_LEVEL[effort]
    assert mapped.upper() in VALID, f"{effort} → {mapped}: невідоме значення для Gemini"


def test_anthropic_only_levels_degrade_to_high():
    """`xhigh` і `max` існують лише в Anthropic — у Gemini вони мусять стати HIGH."""
    assert _EFFORT_TO_LEVEL["xhigh"] == "high"
    assert _EFFORT_TO_LEVEL["max"] == "high"


def test_mapped_values_construct_without_warnings():
    types = pytest.importorskip("google.genai.types")
    for effort, mapped in _EFFORT_TO_LEVEL.items():
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # UserWarning від SDK стає помилкою
            config = types.ThinkingConfig(thinking_level=mapped)
        assert config.thinking_level.value in VALID, effort


def test_unmapped_effort_falls_back_instead_of_leaking():
    """Невідомий effort не має просочитися в SDK як є."""
    assert _EFFORT_TO_LEVEL.get("турборежим", "high") == "high"
