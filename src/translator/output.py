"""Output formatting for translated programs."""

from typing import List, Optional, Set

from parser import (
    Program,
    Statement,
    Instruction,
    Label,
    Directive,
    Operand,
    Register,
    Immediate,
    Memory,
    LabelRef,
)
from scratch import ScratchMemory

from .config import TranslatorConfig


class OutputFormatter:
    """Formats AST back into assembly text."""

    def __init__(
        self,
        indent: str = "    ",
        config: Optional[TranslatorConfig] = None,
        required_luts: Optional[Set[str]] = None,
    ):
        self.indent = indent
        self.config = config or TranslatorConfig()
        self.required_luts = required_luts

    def format(self, program: Program) -> str:
        lines: List[str] = []

        if self.config.include_scratch:
            lines.append(ScratchMemory.to_assembly())
            lines.append("")

        if self.config.include_luts and self.config.lut_path:
            lut_content = self._load_luts()
            if lut_content:
                lines.append(lut_content)
                lines.append("")

        if self.config.include_scratch or self.config.include_luts:
            lines.append(".section .text")
            lines.append("")
            lines.append("# Translated Code")
            lines.append("")

        for stmt in program.statements:
            lines.append(self._format_statement(stmt))

        return "\n".join(lines) + "\n"

    def _load_luts(self) -> Optional[str]:
        """Load LUT assembly, including only tables required by the program."""
        lut_path = self.config.lut_path
        if lut_path is None:
            return None

        if self.required_luts is not None:
            if not self.required_luts:
                return None
            return self._load_selective_luts(self.required_luts)

        lut_file = lut_path / "all_luts.s"
        if not lut_file.exists():
            return None
        return lut_file.read_text()

    def _load_selective_luts(self, lut_names: Set[str]) -> Optional[str]:
        if self.config.lut_path is None:
            return None

        parts: List[str] = [".data", ""]
        loaded = 0
        for name in sorted(lut_names):
            lut_file = self.config.lut_path / f"{name}_lut.s"
            if not lut_file.exists():
                continue
            content = lut_file.read_text()

            stripped_lines: List[str] = []
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped.startswith("#") or stripped == ".data" or stripped == "":
                    continue
                stripped_lines.append(line)

            if stripped_lines:
                parts.append(f"# {name.upper()} LUT")
                parts.extend(stripped_lines)
                parts.append("")
                loaded += 1

        if loaded == 0:
            return None
        return "\n".join(parts)

    def _format_statement(self, stmt: Statement) -> str:
        if isinstance(stmt, Label):
            return f"{stmt.name}:"

        if isinstance(stmt, Directive):
            return self._format_directive(stmt)

        if isinstance(stmt, Instruction):
            return self._format_instruction(stmt)

        return ""

    def _format_directive(self, directive: Directive) -> str:
        if not directive.args:
            return f".{directive.name}"

        args_str = ", ".join(self._format_directive_arg(a) for a in directive.args)
        return f".{directive.name} {args_str}"

    def _format_directive_arg(self, arg) -> str:
        if isinstance(arg, str):
            return arg
        return str(arg)

    def _format_instruction(self, instr: Instruction) -> str:
        if not instr.operands:
            return f"{self.indent}{instr.mnemonic}"

        operands_str = ", ".join(self._format_operand(op) for op in instr.operands)
        return f"{self.indent}{instr.mnemonic} {operands_str}"

    def _format_operand(self, op: Operand) -> str:
        if isinstance(op, Register):
            return f"%{op.name}"

        if isinstance(op, Immediate):
            return f"${op.value}"

        if isinstance(op, Memory):
            return self._format_memory(op)

        if isinstance(op, LabelRef):
            return op.name

        return str(op)

    def _format_memory(self, mem: Memory) -> str:
        if mem.base is None and mem.index is None:
            return str(mem.displacement) if mem.displacement is not None else "0"

        parts: List[str] = []

        if mem.displacement is not None:
            parts.append(str(mem.displacement))

        parts.append("(")

        if mem.base:
            parts.append(f"%{mem.base.name}")

        if mem.index:
            if mem.base:
                parts.append(f", %{mem.index.name}, {mem.scale}")
            else:
                parts.append(f", %{mem.index.name}, {mem.scale}")

        parts.append(")")

        return "".join(parts)


def format_program(program: Program, config: Optional[TranslatorConfig] = None) -> str:
    formatter = OutputFormatter(config=config)
    return formatter.format(program)
