import pytest

from src.agent import ExtractionMode
from src.main import get_parser_mode


@pytest.mark.parametrize(
    "mode,key,expected",
    [
        (None, None, ExtractionMode.RULES),
        (None, "test-key", ExtractionMode.RULES),
        ("rules", "test-key", ExtractionMode.RULES),
        ("llm", None, ExtractionMode.RULES),
        ("llm", "", ExtractionMode.RULES),
        ("llm", "   ", ExtractionMode.RULES),
        ("llm", "test-key", ExtractionMode.LLM),
        ("invalid", "test-key", ExtractionMode.RULES),
    ],
)
def test_parser_mode_requires_opt_in_and_key(monkeypatch, mode, key, expected):
    for name, value in (("PARSER_MODE", mode), ("ANTHROPIC_API_KEY", key)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert get_parser_mode() == expected
