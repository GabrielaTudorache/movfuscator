"""Translator package public API."""

from .config import TranslatorConfig, TranslatorError
from .core import Translator, translate
from .output import OutputFormatter, format_program
from .metadata import (
    InstructionMapping,
    TranslationMetadata,
    TranslationPhase,
    metadata_to_dict,
    metadata_to_json,
)

__all__ = [
    "InstructionMapping",
    "OutputFormatter",
    "Translator",
    "TranslatorConfig",
    "TranslatorError",
    "TranslationMetadata",
    "TranslationPhase",
    "format_program",
    "metadata_to_dict",
    "metadata_to_json",
    "translate",
]
