"""Lookup Table (LUT) generator for MOV-only assembly operations."""

from pathlib import Path
from typing import Callable, List


def signed_byte(b: int) -> int:
    """Convert unsigned byte (0-255) to signed value (-128 to 127)."""
    return b if b < 128 else b - 256


def generate_2d_lut(func: Callable[[int, int], int]) -> List[bytearray]:
    """Generate 256x256 LUT with given function."""
    rows = []
    for a in range(256):
        row = bytearray(256)
        for b in range(256):
            row[b] = func(a, b) & 0xFF
        rows.append(row)
    return rows


def generate_1d_lut(func: Callable[[int], int]) -> bytearray:
    """Generate 256-entry LUT with given function."""
    lut = bytearray(256)
    for a in range(256):
        lut[a] = func(a) & 0xFF
    return lut


def lut_2d_to_asm(rows: List[bytearray], name: str) -> str:
    """Convert 2D LUT to assembly format with row pointers."""
    lines = []

    # Row pointer table
    lines.append(f"{name}_row_ptrs:")
    for i in range(0, 256, 8):
        chunk = ", ".join(f"{name}_row_{j}" for j in range(i, min(i + 8, 256)))
        lines.append(f"    .long {chunk}")

    lines.append("")

    # Actual data rows
    for i, row in enumerate(rows):
        lines.append(f"{name}_row_{i}:")
        for j in range(0, 256, 16):
            chunk = ", ".join(str(row[k]) for k in range(j, min(j + 16, 256)))
            lines.append(f"    .byte {chunk}")

    return "\n".join(lines)


def lut_1d_to_asm(lut: bytearray, name: str) -> str:
    """Convert 1D LUT to assembly format."""
    lines = [f"{name}_lut:"]
    for i in range(0, 256, 16):
        chunk = ", ".join(str(lut[k]) for k in range(i, min(i + 16, 256)))
        lines.append(f"    .byte {chunk}")
    return "\n".join(lines)


class LUTGenerator:
    """Generates all LUTs needed for MOV-only operations."""

    # 2D LUT definitions: (name, lambda)
    LUT_2D_DEFS = {
        # Arithmetic
        "add": lambda a, b: a + b,
        "carry": lambda a, b: 1 if a + b > 255 else 0,
        "sub": lambda a, b: a - b,
        "borrow": lambda a, b: 1 if a < b else 0,
        # Logical
        "xor": lambda a, b: a ^ b,
        "or": lambda a, b: a | b,
        "and": lambda a, b: a & b,
        # Multiplication (8-bit)
        "mul8_lo": lambda a, b: a * b,
        "mul8_hi": lambda a, b: (a * b) >> 8,
        # Unsigned comparisons
        "jb": lambda a, b: 1 if a < b else 0,  # below (unsigned <)
        "je": lambda a, b: 1 if a == b else 0,  # equal
        # Signed comparisons
        "jl_signed": lambda a, b: 1 if signed_byte(a) < signed_byte(b) else 0,
    }

    # 1D LUT definitions: (name, lambda)
    LUT_1D_DEFS = {
        "is_zero": lambda a: 1 if a == 0 else 0,
        "is_not_zero": lambda a: 1 if a != 0 else 0,
        "shl1": lambda a: a << 1,
        "shl1_carry": lambda a: (a >> 7) & 1,
        "shr1": lambda a: a >> 1,
        "shr1_carry": lambda a: a & 1,
    }

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._luts_2d = {}
        self._luts_1d = {}

    def generate_all(self) -> None:
        """Generate all LUTs."""
        # Generate 2D LUTs
        for name, func in self.LUT_2D_DEFS.items():
            self._luts_2d[name] = generate_2d_lut(func)

        # Generate 1D LUTs
        for name, func in self.LUT_1D_DEFS.items():
            self._luts_1d[name] = generate_1d_lut(func)

    def save_all(self) -> None:
        """Save all LUTs to assembly files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save 2D LUTs
        for name, rows in self._luts_2d.items():
            asm = self._wrap_lut_file(lut_2d_to_asm(rows, name), name, "2D")
            filepath = self.output_dir / f"{name}_lut.s"
            filepath.write_text(asm)

        # Save 1D LUTs
        for name, lut in self._luts_1d.items():
            asm = self._wrap_lut_file(lut_1d_to_asm(lut, name), name, "1D")
            filepath = self.output_dir / f"{name}_lut.s"
            filepath.write_text(asm)

        # Generate combined file with all LUTs
        self._save_combined()

    def _wrap_lut_file(self, content: str, name: str, lut_type: str) -> str:
        """Wrap LUT content with header comment."""
        header = f"""# {name.upper()} Lookup Table ({lut_type})

.data
"""
        return header + content + "\n"

    def _save_combined(self) -> None:
        """Save all LUTs in a single combined file."""
        lines = [
            ".data",
            "",
        ]

        # Add 2D LUTs
        lines.append("# 2D LUTs")
        lines.append("")
        for name, rows in self._luts_2d.items():
            lines.append(f"# {name.upper()} LUT")
            lines.append(lut_2d_to_asm(rows, name))
            lines.append("")

        # Add 1D LUTs
        lines.append("# 1D LUTs")
        lines.append("")
        for name, lut in self._luts_1d.items():
            lines.append(f"# {name.upper()} LUT")
            lines.append(lut_1d_to_asm(lut, name))
            lines.append("")

        filepath = self.output_dir / "all_luts.s"
        filepath.write_text("\n".join(lines))


def generate_luts(output_dir: str = "lut") -> None:
    """Generate all LUTs and save to output directory."""
    generator = LUTGenerator(Path(output_dir))
    generator.generate_all()
    generator.save_all()


if __name__ == "__main__":
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "lut"
    generate_luts(output_dir)
    print(f"LUTs generated in {output_dir}/")
