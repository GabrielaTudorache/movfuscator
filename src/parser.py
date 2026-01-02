"""
MOVFUSCATOR Parser - Builds AST from token stream.
Parses x86 32-bit AT&T assembly into an Abstract Syntax Tree.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from enum import Enum

from lexer import Token, TokenType, Lexer

# Instructions that accept size suffixes (b/w/l/q)
SIZE_SUFFIXED_MNEMONICS = frozenset(
    {
        "mov",
        "movz",
        "movs",
        "add",
        "sub",
        "xor",
        "or",
        "and",
        "cmp",
        "test",
        "inc",
        "dec",
        "mul",
        "div",
        "shl",
        "shr",
        "push",
        "pop",
        "lea",
        "neg",
        "not",
        "imul",
        "idiv",
        "sal",
        "sar",
        "rol",
        "ror",
    }
)


class OperandSize(Enum):
    """Size suffix for instructions."""

    BYTE = "b"  # 8-bit
    WORD = "w"  # 16-bit
    LONG = "l"  # 32-bit
    QUAD = "q"  # 64-bit
    NONE = ""  # No suffix


@dataclass
class Register:
    """Represents a register operand."""

    name: str  # Register name without % (e.g., "eax", "al")

    @property
    def size(self) -> OperandSize:
        """Determine register size from name."""
        if self.name in ("al", "ah", "bl", "bh", "cl", "ch", "dl", "dh"):
            return OperandSize.BYTE
        if self.name in ("ax", "bx", "cx", "dx", "si", "di", "sp", "bp"):
            return OperandSize.WORD
        if self.name in ("eax", "ebx", "ecx", "edx", "esi", "edi", "esp", "ebp"):
            return OperandSize.LONG
        return OperandSize.NONE

    def __repr__(self) -> str:
        return f"Register(%{self.name})"


@dataclass
class Immediate:
    """Represents an immediate value operand."""

    value: Union[int, str]  # Numeric value or label name
    is_label: bool = False

    def __repr__(self) -> str:
        return f"Immediate(${self.value})"


@dataclass
class Memory:
    """
    Represents a memory operand in AT&T syntax.
    Format: displacement(base, index, scale)
    """

    displacement: Union[int, str, None] = None  # Offset or label
    base: Optional[Register] = None  # Base register
    index: Optional[Register] = None  # Index register
    scale: int = 1  # Scale factor (1, 2, 4, 8)

    def __repr__(self) -> str:
        parts = []
        if self.displacement is not None:
            parts.append(str(self.displacement))
        parts.append("(")
        if self.base:
            parts.append(f"%{self.base.name}")
        if self.index:
            parts.append(f", %{self.index.name}, {self.scale}")
        elif self.scale != 1 and self.base is None:
            # Format: disp(, index, scale) - no base
            parts.append(f", %{self.index.name if self.index else ''}, {self.scale}")
        parts.append(")")
        return f"Memory({''.join(parts)})"


@dataclass
class LabelRef:
    """Represents a label reference (for jumps, calls, or memory access)."""

    name: str

    def __repr__(self) -> str:
        return f"LabelRef({self.name})"


# Union type for all operand types
Operand = Union[Register, Immediate, Memory, LabelRef]


@dataclass
class Instruction:
    """Represents an assembly instruction."""

    mnemonic: str  # Instruction name (e.g., "mov", "add")
    operands: List[Operand] = field(default_factory=list)
    size_suffix: OperandSize = OperandSize.NONE
    line: int = 0

    @property
    def base_mnemonic(self) -> str:
        """Get mnemonic without size suffix."""
        m = self.mnemonic.lower()
        if m.endswith(("b", "w", "l", "q")) and len(m) > 1:
            base = m[:-1]
            if base in SIZE_SUFFIXED_MNEMONICS:
                return base
        return m

    def __repr__(self) -> str:
        ops = ", ".join(str(op) for op in self.operands)
        return f"Instruction({self.mnemonic}, [{ops}])"


@dataclass
class Label:
    """Represents a label definition."""

    name: str
    line: int = 0

    def __repr__(self) -> str:
        return f"Label({self.name}:)"


@dataclass
class Directive:
    """Represents an assembler directive."""

    name: str  # Directive name without . (e.g., "data", "long")
    args: List[Union[str, int]] = field(default_factory=list)
    line: int = 0

    def __repr__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"Directive(.{self.name} {args_str})"


# Union type for all statement types
Statement = Union[Instruction, Label, Directive]


@dataclass
class Program:
    """Represents a complete assembly program."""

    statements: List[Statement] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Program({len(self.statements)} statements)"


class ParserError(Exception):
    """Exception raised for parser errors."""

    def __init__(self, message: str, line: int, column: int = 0):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Parser error at line {line}: {message}")


class Parser:
    """
    Parser for x86 32-bit AT&T assembly.

    Usage:
        parser = Parser(tokens)
        program = parser.parse()
    """

    def __init__(self, tokens: List[Token]):
        """Initialize parser with token list."""
        self.tokens = tokens
        self.pos = 0
        self.current_line = 1

    def _current(self) -> Token:
        """Get current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # Return EOF

    def _peek(self, offset: int = 1) -> Token:
        """Peek at token at offset from current position."""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]

    def _advance(self) -> Token:
        """Advance to next token and return current."""
        token = self._current()
        if self.pos < len(self.tokens):
            self.current_line = token.line
            self.pos += 1
        return token

    def _match(self, *types: TokenType) -> bool:
        """Check if current token matches any of the given types."""
        return self._current().type in types

    def _expect(self, token_type: TokenType, message: str = "") -> Token:
        """Expect current token to be of given type, raise error otherwise."""
        if not self._match(token_type):
            msg = (
                message
                or f"Expected {token_type.name}, got {self._current().type.name}"
            )
            raise ParserError(msg, self._current().line, self._current().column)
        return self._advance()

    def _skip_newlines(self) -> None:
        """Skip newline tokens."""
        while self._match(TokenType.NEWLINE):
            self._advance()

    def _skip_comments_and_newlines(self) -> None:
        """Skip comment and newline tokens."""
        while self._match(TokenType.NEWLINE, TokenType.COMMENT):
            self._advance()

    def parse(self) -> Program:
        """
        Parse the token stream into a Program AST.

        Returns:
            Program AST
        """
        statements: List[Statement] = []

        while not self._match(TokenType.EOF):
            self._skip_comments_and_newlines()

            if self._match(TokenType.EOF):
                break

            stmt = self._parse_statement()
            if stmt is not None:
                statements.append(stmt)

        return Program(statements=statements)

    def _parse_statement(self) -> Optional[Statement]:
        """Parse a single statement (instruction, label, or directive)."""
        if self._match(TokenType.DIRECTIVE):
            return self._parse_directive()

        if self._match(TokenType.IDENTIFIER):
            # Could be label definition or instruction
            if self._peek().type == TokenType.COLON:
                return self._parse_label()
            return self._parse_instruction()

        if self._match(TokenType.NEWLINE, TokenType.COMMENT):
            self._advance()
            return None

        raise ParserError(
            f"Unexpected token: {self._current().value!r}",
            self._current().line,
            self._current().column,
        )

    def _parse_directive(self) -> Directive:
        """Parse an assembler directive."""
        token = self._expect(TokenType.DIRECTIVE)
        name = token.value[1:]  # Remove leading .
        line = token.line
        args: List[Union[str, int]] = []

        # Parse directive arguments until newline or EOF
        while not self._match(TokenType.NEWLINE, TokenType.EOF, TokenType.COMMENT):
            if self._match(TokenType.COMMA):
                self._advance()
                continue

            if self._match(TokenType.IDENTIFIER):
                args.append(self._advance().value)
            elif self._match(TokenType.NUMBER):
                args.append(self._parse_number(self._advance().value))
            elif self._match(TokenType.STRING):
                args.append(self._advance().value)
            elif self._match(TokenType.IMMEDIATE):
                # Handle immediate as directive arg (e.g., .long $label)
                # Strip the $ prefix and parse
                imm_value = self._advance().value[1:]  # Remove $
                if imm_value and (imm_value[0].isdigit() or imm_value[0] == "-"):
                    args.append(self._parse_number(imm_value))
                else:
                    args.append(imm_value)  # Label reference
            else:
                break

        return Directive(name=name, args=args, line=line)

    def _parse_label(self) -> Label:
        """Parse a label definition."""
        name_token = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.COLON)
        return Label(name=name_token.value, line=name_token.line)

    def _parse_instruction(self) -> Instruction:
        """Parse an instruction."""
        mnemonic_token = self._expect(TokenType.IDENTIFIER)
        mnemonic = mnemonic_token.value.lower()
        line = mnemonic_token.line

        # Determine size suffix only for instructions that use them
        size_suffix = OperandSize.NONE
        if len(mnemonic) > 1 and mnemonic[-1] in ("b", "w", "l", "q"):
            base = mnemonic[:-1]
            if base in SIZE_SUFFIXED_MNEMONICS:
                suffix_char = mnemonic[-1]
                if suffix_char == "b":
                    size_suffix = OperandSize.BYTE
                elif suffix_char == "w":
                    size_suffix = OperandSize.WORD
                elif suffix_char == "l":
                    size_suffix = OperandSize.LONG
                elif suffix_char == "q":
                    size_suffix = OperandSize.QUAD

        # Parse operands
        operands: List[Operand] = []

        while not self._match(TokenType.NEWLINE, TokenType.EOF, TokenType.COMMENT):
            if self._match(TokenType.COMMA):
                self._advance()
                continue

            operand = self._parse_operand()
            if operand is not None:
                operands.append(operand)
            else:
                break

        return Instruction(
            mnemonic=mnemonic, operands=operands, size_suffix=size_suffix, line=line
        )

    def _parse_operand(self) -> Optional[Operand]:
        """Parse a single operand."""
        # Indirect operand prefix: *%reg, *label, *(%reg), etc.
        # Used in AT&T syntax for indirect jumps/calls.
        if self._match(TokenType.STAR):
            return self._parse_indirect_operand()

        # Register: %eax
        if self._match(TokenType.REGISTER):
            return self._parse_register()

        # Immediate: $5, $label
        if self._match(TokenType.IMMEDIATE):
            return self._parse_immediate()

        # Memory or label reference
        if self._match(TokenType.IDENTIFIER):
            return self._parse_memory_or_label()

        # Memory with displacement: 8(%ebp)
        if self._match(TokenType.NUMBER):
            return self._parse_memory_with_displacement()

        # Memory without displacement: (%eax) or (, %eax, 4)
        if self._match(TokenType.LPAREN):
            return self._parse_memory()

        return None

    def _parse_indirect_operand(self) -> Operand:
        """Parse an indirect operand (prefixed with *).

        AT&T syntax uses * for indirect jump/call targets:
        - *%eax       -> indirect through register
        - *label      -> indirect through memory at label
        - *(%eax)     -> indirect through memory pointed by register
        - *4(%eax)    -> indirect through memory with displacement

        Since these are passthrough (jmp/call), we preserve the * prefix
        by wrapping the result in a LabelRef with '*' prefix.
        """
        self._advance()  # consume *

        # Parse the inner operand
        inner = self._parse_operand()
        if inner is None:
            raise ParserError(
                "Expected operand after '*'",
                self._current().line,
                self._current().column,
            )

        # Wrap with * prefix for the output formatter
        if isinstance(inner, Register):
            return LabelRef(f"*%{inner.name}")
        elif isinstance(inner, LabelRef):
            return LabelRef(f"*{inner.name}")
        elif isinstance(inner, Memory):
            # Format the memory operand and prefix with *
            return LabelRef(f"*{self._format_memory_for_indirect(inner)}")
        elif isinstance(inner, Immediate):
            return LabelRef(f"*${inner.value}")
        return inner

    @staticmethod
    def _format_memory_for_indirect(mem: Memory) -> str:
        """Format a Memory operand back to AT&T string for indirect wrapping."""
        parts = []
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

    def _parse_register(self) -> Register:
        """Parse a register operand."""
        token = self._expect(TokenType.REGISTER)
        name = token.value[1:]  # Remove leading %
        return Register(name=name)

    def _parse_immediate(self) -> Immediate:
        """Parse an immediate value operand."""
        token = self._expect(TokenType.IMMEDIATE)
        value_str = token.value[1:]  # Remove leading $

        # Check if it's a label reference
        if value_str and not value_str[0].isdigit() and value_str[0] != "-":
            return Immediate(value=value_str, is_label=True)

        return Immediate(value=self._parse_number(value_str), is_label=False)

    def _parse_memory_or_label(self) -> Operand:
        """Parse memory reference starting with identifier or plain label reference."""
        id_token = self._advance()
        name = id_token.value

        # Check if followed by memory addressing
        if self._match(TokenType.LPAREN):
            return self._parse_memory(displacement=name)

        # Plain label reference
        return LabelRef(name=name)

    def _parse_memory_with_displacement(self) -> Memory:
        """Parse memory with numeric displacement: 8(%ebp)."""
        num_token = self._advance()
        displacement = self._parse_number(num_token.value)

        if self._match(TokenType.LPAREN):
            return self._parse_memory(displacement=displacement)

        # Just a number (shouldn't happen in valid assembly, but handle it)
        return Memory(displacement=displacement)

    def _parse_memory(self, displacement: Union[int, str, None] = None) -> Memory:
        """
        Parse memory operand.
        Formats:
            (%base)
            (%base, %index, scale)
            (, %index, scale)
            disp(%base)
            disp(%base, %index, scale)
            disp(, %index, scale)
        """
        self._expect(TokenType.LPAREN)

        base: Optional[Register] = None
        index: Optional[Register] = None
        scale: int = 1

        # Check for empty base: (, %index, scale)
        if self._match(TokenType.COMMA):
            self._advance()
            # Parse index register
            if self._match(TokenType.REGISTER):
                index = self._parse_register()
            # Parse scale
            if self._match(TokenType.COMMA):
                self._advance()
                if self._match(TokenType.NUMBER):
                    scale = self._parse_number(self._advance().value)
        elif self._match(TokenType.REGISTER):
            # Parse base register
            base = self._parse_register()

            # Check for index and scale
            if self._match(TokenType.COMMA):
                self._advance()
                if self._match(TokenType.REGISTER):
                    index = self._parse_register()
                if self._match(TokenType.COMMA):
                    self._advance()
                    if self._match(TokenType.NUMBER):
                        scale = self._parse_number(self._advance().value)

        self._expect(TokenType.RPAREN)

        return Memory(displacement=displacement, base=base, index=index, scale=scale)

    def _parse_number(self, value: str) -> int:
        """Parse a numeric string to int."""
        value = value.strip()
        if value.startswith("0x") or value.startswith("-0x"):
            return int(value, 16)
        if value.startswith("0b") or value.startswith("-0b"):
            return int(value, 2)
        return int(value)


def parse(tokens: List[Token]) -> Program:
    """
    Convenience function to parse a token list.

    Args:
        tokens: List of tokens from lexer

    Returns:
        Program AST
    """
    parser = Parser(tokens)
    return parser.parse()


def parse_source(source: str) -> Program:
    """
    Convenience function to parse source code directly.

    Args:
        source: Assembly source code

    Returns:
        Program AST
    """
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    # Filter out comments
    tokens = Lexer.filter_tokens(tokens, [TokenType.COMMENT])
    return parse(tokens)


def parse_file(filepath: str) -> Program:
    """
    Parse an assembly file.

    Args:
        filepath: Path to the assembly file

    Returns:
        Program AST
    """
    with open(filepath, "r") as f:
        source = f.read()
    return parse_source(source)
