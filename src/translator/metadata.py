"""Metadata structures for tracking translation phases."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List

from parser import Statement


@dataclass
class TranslationPhase:
    """A named group of output statements within one instruction's translation."""

    name: str
    category: str
    statements: List[Statement] = field(default_factory=list)


@dataclass
class InstructionMapping:
    """Maps one source instruction to its translated phases."""

    source_line: int
    source_text: str
    instruction_type: str
    phases: List[TranslationPhase] = field(default_factory=list)
    total_mov_count: int = 0


@dataclass
class TranslationMetadata:
    """Complete metadata for a translated program."""

    mappings: List[InstructionMapping] = field(default_factory=list)


def metadata_to_json(
    metadata: TranslationMetadata, formatter: "OutputFormatter"
) -> str:
    """Serialize translation metadata to JSON."""

    return json.dumps(metadata_to_dict(metadata, formatter), indent=2)


def metadata_to_dict(
    metadata: TranslationMetadata, formatter: "OutputFormatter"
) -> Dict[str, Any]:
    """Convert metadata to a JSON-serializable dict."""

    mappings: List[Dict[str, Any]] = []
    for m in metadata.mappings:
        phases: List[Dict[str, Any]] = []
        for p in m.phases:
            phase_instructions: List[str] = []
            for stmt in p.statements:
                text = formatter._format_statement(stmt).strip()
                if text:
                    phase_instructions.append(text)
            phases.append(
                {
                    "name": p.name,
                    "category": p.category,
                    "instructions": phase_instructions,
                    "instruction_count": len(phase_instructions),
                }
            )
        mappings.append(
            {
                "source_line": m.source_line,
                "source_text": m.source_text,
                "type": m.instruction_type,
                "phases": phases,
                "total_mov_count": m.total_mov_count,
            }
        )

    total_source = len(mappings)
    total_output = sum(m["total_mov_count"] for m in mappings)
    type_counts: Dict[str, int] = {}
    for m in mappings:
        t = m["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "mappings": mappings,
        "statistics": {
            "total_source_instructions": total_source,
            "total_output_instructions": total_output,
            "average_expansion_ratio": round(total_output / max(total_source, 1), 1),
            "instruction_type_counts": type_counts,
        },
    }
