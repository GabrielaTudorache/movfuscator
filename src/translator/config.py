"""Translator configuration and error types."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TranslatorConfig:
    """Configuration for the translator."""

    include_luts: bool = True
    include_scratch: bool = True
    lut_path: Optional[Path] = Path(__file__).resolve().parents[2] / "lut"


class TranslatorError(Exception):
    def __init__(self, message: str, line: int = 0):
        self.message = message
        self.line = line
        super().__init__(f"Translator error at line {line}: {message}")
