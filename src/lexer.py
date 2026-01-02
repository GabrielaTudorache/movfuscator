"""
MOVFUSCATOR Lexer - Tokenizer for x86 AT&T Assembly
Transforms assembly source code into a stream of tokens.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Iterator
import re


class TokenType(Enum):
    # Directives
    DIRECTIVE = auto()  # .data, .text, .global, .long, .ascii, .space, etc.

    # Operands
    REGISTER = auto()  # %eax, %ebx, %ecx, %edx, %esi, %edi, %esp, %ebp
    IMMEDIATE = auto()  # $5, $0x80, $-1
    IDENTIFIER = auto()  # labels, variable names, instruction mnemonics

    # Memory addressing
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    COMMA = auto()  # ,

    # Labels
    COLON = auto()  # : (after label definition)

    # String literals
    STRING = auto()  # "string content"

    # Numbers (without $ prefix - used in directives and memory offsets)
    NUMBER = auto()  # 5, 0x80, -10

    # Indirect operand prefix
    STAR = auto()  # * (for indirect jumps/calls: jmp *%eax, call *label)

    # Structure
    NEWLINE = auto()
    COMMENT = auto()  # # or ; comment
    EOF = auto()


@dataclass
class Token:
    """Represents a single token from the lexer."""

    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.column})"


class LexerError(Exception):
    """Exception raised for lexer errors."""

    def __init__(self, message: str, line: int, column: int):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Lexer error at line {line}, column {column}: {message}")


class Lexer:
    """
    Tokenizer for x86 32-bit AT&T assembly syntax.

    Usage:
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
    """

    # Token patterns (more specific patterns first)
    TOKEN_PATTERNS = [
        # Comments (must come before other patterns)
        (r"#[^\n]*", TokenType.COMMENT),
        (r";[^\n]*", TokenType.COMMENT),
        # String literals
        (r'"(?:[^"\\]|\\.)*"', TokenType.STRING),
        # Local labels (GAS-style .L prefix, must come before directives)
        (r"\.L[a-zA-Z0-9_]*", TokenType.IDENTIFIER),
        # Directives (start with .)
        (r"\.[a-zA-Z_][a-zA-Z0-9_]*", TokenType.DIRECTIVE),
        # Immediate values (start with $)
        (r"\$-?0x[0-9a-fA-F]+", TokenType.IMMEDIATE),  # Hex: $0x80
        (r"\$-?0b[01]+", TokenType.IMMEDIATE),  # Binary: $0b1010
        (r"\$-?[0-9]+", TokenType.IMMEDIATE),  # Decimal: $5, $-1
        (
            r"\$[a-zA-Z_][a-zA-Z0-9_]*",
            TokenType.IMMEDIATE,
        ),  # Label as immediate: $label
        # Registers (start with %)
        (r"%[a-zA-Z]+", TokenType.REGISTER),
        # Numbers (for directives like .long 10, memory offsets like 8(%ebp))
        (r"-?0x[0-9a-fA-F]+", TokenType.NUMBER),
        (r"-?0b[01]+", TokenType.NUMBER),
        (r"-?[0-9]+", TokenType.NUMBER),
        # Identifiers (instructions, labels, variable names)
        (r"[a-zA-Z_][a-zA-Z0-9_]*", TokenType.IDENTIFIER),
        # Punctuation
        (r"\*", TokenType.STAR),
        (r"\(", TokenType.LPAREN),
        (r"\)", TokenType.RPAREN),
        (r",", TokenType.COMMA),
        (r":", TokenType.COLON),
        # Newlines
        (r"\n", TokenType.NEWLINE),
    ]

    # Whitespace pattern (to be skipped, but not newlines)
    WHITESPACE = re.compile(r"[ \t\r]+")

    def __init__(self, source: str):
        """
        Initialize the lexer with source code.

        Args:
            source: The assembly source code to tokenize
        """
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []

        # Compile all patterns
        self._compiled_patterns = [
            (re.compile(pattern), token_type)
            for pattern, token_type in self.TOKEN_PATTERNS
        ]

    def _advance(self, count: int = 1) -> None:
        """Advance position by count characters, updating line/column."""
        for _ in range(count):
            if self.pos < len(self.source):
                if self.source[self.pos] == "\n":
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
                self.pos += 1

    def _skip_whitespace(self) -> None:
        """Skip whitespace (but not newlines)."""
        match = self.WHITESPACE.match(self.source, self.pos)
        if match:
            length = len(match.group())
            self.column += length
            self.pos += length

    def _match_token(self) -> Optional[Token]:
        """Try to match a token at the current position."""
        for pattern, token_type in self._compiled_patterns:
            match = pattern.match(self.source, self.pos)
            if match:
                value = match.group()
                token = Token(
                    type=token_type, value=value, line=self.line, column=self.column
                )

                # Advance position
                if token_type == TokenType.NEWLINE:
                    self.line += 1
                    self.column = 1
                    self.pos += 1
                else:
                    self.column += len(value)
                    self.pos += len(value)

                return token

        return None

    def tokenize(self) -> List[Token]:
        """
        Tokenize the entire source code.

        Returns:
            List of tokens

        Raises:
            LexerError: If an unrecognized character is encountered
        """
        self.tokens = []
        self.pos = 0
        self.line = 1
        self.column = 1

        while self.pos < len(self.source):
            # Skip whitespace (not newlines)
            self._skip_whitespace()

            if self.pos >= len(self.source):
                break

            # Try to match a token
            token = self._match_token()
            if token:
                self.tokens.append(token)
            else:
                # Unrecognized character
                char = self.source[self.pos]
                raise LexerError(
                    f"Unexpected character: {char!r}", self.line, self.column
                )

        # Add EOF token
        self.tokens.append(
            Token(type=TokenType.EOF, value="", line=self.line, column=self.column)
        )

        return self.tokens

    def tokenize_iter(self) -> Iterator[Token]:
        """
        Tokenize and yield tokens one at a time.

        Yields:
            Tokens one at a time
        """
        for token in self.tokenize():
            yield token

    @staticmethod
    def filter_tokens(
        tokens: List[Token], exclude_types: Optional[List[TokenType]] = None
    ) -> List[Token]:
        """
        Filter out unwanted token types.

        Args:
            tokens: List of tokens to filter
            exclude_types: Token types to exclude (default: COMMENT)

        Returns:
            Filtered list of tokens
        """
        if exclude_types is None:
            exclude_types = [TokenType.COMMENT]

        return [t for t in tokens if t.type not in exclude_types]


def tokenize(source: str) -> List[Token]:
    """
    Convenience function to tokenize source code.

    Args:
        source: Assembly source code

    Returns:
        List of tokens
    """
    lexer = Lexer(source)
    return lexer.tokenize()


def tokenize_file(filepath: str) -> List[Token]:
    """
    Tokenize an assembly file.

    Args:
        filepath: Path to the assembly file

    Returns:
        List of tokens
    """
    with open(filepath, "r") as f:
        source = f.read()
    return tokenize(source)
