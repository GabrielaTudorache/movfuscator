"""Scratch memory layout for MOV-only operations."""

from dataclasses import dataclass
from typing import List


@dataclass
class ScratchVariable:
    """A scratch memory variable."""

    name: str
    size: str  # "long", "byte", "quad"
    initial_value: int = 0
    comment: str = ""


class ScratchMemory:
    """Manages scratch memory layout for MOV-only translations.

    Scratch memory provides temporary storage for:
    - Operand extraction (byte-by-byte processing)
    - Carry/borrow propagation
    - Intermediate results for multi-byte operations
    - Comparison flags for conditional jumps
    """

    # Core scratch variables for basic operations
    CORE_VARIABLES: List[ScratchVariable] = [
        ScratchVariable("scratch_a", "long", 0, "operand 1 (4 bytes)"),
        ScratchVariable("scratch_b", "long", 0, "operand 2 (4 bytes)"),
        ScratchVariable("scratch_r", "long", 0, "result (4 bytes)"),
        ScratchVariable("scratch_r_saved", "long", 0, "saved scratch_r (for LEA)"),
        ScratchVariable("scratch_c", "byte", 0, "carry/borrow (1 byte)"),
        ScratchVariable("scratch_t", "long", 0, "temporary (4 bytes)"),
        ScratchVariable("scratch_t2", "long", 0, "temporary 2 (4 bytes, for LEA)"),
        ScratchVariable("lea_base_saved", "long", 0, "saved LEA base (for LEA)"),
        ScratchVariable("lea_index_saved", "long", 0, "saved LEA index (for LEA)"),
        # Saved registers - used to preserve temp registers during translation
        ScratchVariable("save_eax", "long", 0, "saved eax"),
        ScratchVariable("save_ecx", "long", 0, "saved ecx"),
        ScratchVariable("save_edx", "long", 0, "saved edx"),
    ]

    # Variables for multi-byte comparison operations (per-byte intermediates)
    COMPARISON_VARIABLES: List[ScratchVariable] = [
        ScratchVariable("scratch_eq3", "byte", 0, "byte 3 equal?"),
        ScratchVariable("scratch_eq2", "byte", 0, "byte 2 equal?"),
        ScratchVariable("scratch_eq1", "byte", 0, "byte 1 equal?"),
        ScratchVariable("scratch_lt3", "byte", 0, "byte 3 less than? (unsigned)"),
        ScratchVariable("scratch_lt2", "byte", 0, "byte 2 less than?"),
        ScratchVariable("scratch_lt1", "byte", 0, "byte 1 less than?"),
        ScratchVariable("scratch_lt0", "byte", 0, "byte 0 less than?"),
        # CMP final results (computed from per-byte intermediates)
        ScratchVariable("scratch_cmp_eq", "byte", 0, "CMP result: 1 if equal"),
        ScratchVariable(
            "scratch_cmp_below", "byte", 0, "CMP result: 1 if dst < src (unsigned)"
        ),
        ScratchVariable(
            "scratch_cmp_sign_lt", "byte", 0, "CMP result: 1 if dst < src (signed)"
        ),
    ]

    # Variables for multiplication (64-bit result)
    MULTIPLICATION_VARIABLES: List[ScratchVariable] = [
        ScratchVariable("mul_op_a", "long", 0, "multiplication operand a"),
        ScratchVariable("mul_op_b", "long", 0, "multiplication operand b"),
        ScratchVariable("mul_accum", "quad", 0, "64-bit accumulator (8 bytes)"),
    ]

    # Variables for division (repeated subtraction)
    DIVISION_VARIABLES: List[ScratchVariable] = [
        ScratchVariable("div_remainder", "long", 0, "division remainder"),
        ScratchVariable("div_divisor", "long", 0, "division divisor"),
        ScratchVariable("div_quotient", "long", 0, "division quotient"),
    ]

    # Variables for shift operations
    SHIFT_VARIABLES: List[ScratchVariable] = [
        ScratchVariable(
            "shr_carry_inject", "byte", 0, "SHR carry map: carry=0 -> addend=0"
        ),
        ScratchVariable(
            "shr_carry_inject_hi",
            "byte",
            128,
            "SHR carry map: carry=1 -> addend=128",
        ),
    ]

    # Variables for control flow dispatcher
    CONTROL_FLOW_VARIABLES: List[ScratchVariable] = [
        ScratchVariable(
            "current_block", "long", 0, "current block index for dispatcher"
        ),
        ScratchVariable(
            "scratch_jcc_target", "long", 0, "conditional jump dispatch target address"
        ),
    ]

    @classmethod
    def get_all_variables(cls) -> List[ScratchVariable]:
        """Returns all scratch variables."""
        return (
            cls.CORE_VARIABLES
            + cls.COMPARISON_VARIABLES
            + cls.MULTIPLICATION_VARIABLES
            + cls.DIVISION_VARIABLES
            + cls.SHIFT_VARIABLES
            + cls.CONTROL_FLOW_VARIABLES
        )

    @classmethod
    def to_assembly(cls, indent: str = "    ") -> str:
        """Generate assembly declarations for all scratch variables.

        Args:
            indent: Indentation for variable definitions.

        Returns:
            Assembly text with .data section and variable definitions.
        """
        lines = [
            "# Scratch Memory",
            ".section .data",
            "",
        ]

        for var in cls.CORE_VARIABLES:
            lines.append(cls._format_variable(var, indent))

        lines.append("")
        for var in cls.COMPARISON_VARIABLES:
            lines.append(cls._format_variable(var, indent))

        lines.append("")
        for var in cls.MULTIPLICATION_VARIABLES:
            lines.append(cls._format_variable(var, indent))

        lines.append("")
        for var in cls.DIVISION_VARIABLES:
            lines.append(cls._format_variable(var, indent))

        lines.append("")
        for var in cls.SHIFT_VARIABLES:
            lines.append(cls._format_variable(var, indent))

        lines.append("")
        for var in cls.CONTROL_FLOW_VARIABLES:
            lines.append(cls._format_variable(var, indent))

        lines.append("")

        return "\n".join(lines)

    @classmethod
    def _format_variable(cls, var: ScratchVariable, indent: str) -> str:
        """Format a single variable declaration."""
        size_directive = {
            "byte": ".byte",
            "long": ".long",
            "quad": ".quad",
        }.get(var.size, ".long")

        comment = f"  # {var.comment}" if var.comment else ""
        return f"{var.name}:{indent}{size_directive} {var.initial_value}{comment}"

    @classmethod
    def get_variable(cls, name: str) -> ScratchVariable:
        """Get a scratch variable by name.

        Args:
            name: Variable name.

        Returns:
            The ScratchVariable object.

        Raises:
            KeyError: If variable not found.
        """
        for var in cls.get_all_variables():
            if var.name == name:
                return var
        raise KeyError(f"Unknown scratch variable: {name}")


def generate_scratch_memory() -> str:
    """Convenience function to generate scratch memory assembly."""
    return ScratchMemory.to_assembly()
