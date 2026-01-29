"""MOVFUSCATOR - Transforms x86 assembly to MOV-only assembly."""

import argparse
import sys
from pathlib import Path

from lexer import Lexer, TokenType
from parser import Parser
from translator import Translator, OutputFormatter, TranslatorConfig, TranslatorError


from typing import Optional


def translate_file(
    input_path: Path,
    output_path: Path,
    include_luts: bool = True,
    include_scratch: bool = True,
    lut_path: Optional[Path] = None,
) -> None:
    source = input_path.read_text()

    lexer = Lexer(source)
    tokens = lexer.tokenize()
    tokens = Lexer.filter_tokens(tokens, [TokenType.COMMENT])

    parser = Parser(tokens)
    program = parser.parse()

    config = TranslatorConfig(
        include_luts=include_luts, include_scratch=include_scratch, lut_path=lut_path
    )
    translator = Translator(config)
    translated = translator.translate(program)

    formatter = OutputFormatter(config=config, required_luts=translator.required_luts)
    output = formatter.format(translated)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output)


def main() -> int:
    argparser = argparse.ArgumentParser(
        description="Transform x86 assembly to MOV-only assembly"
    )
    argparser.add_argument("input", type=Path, help="Input assembly file")
    argparser.add_argument("-o", "--output", type=Path, help="Output assembly file")
    argparser.add_argument(
        "--no-luts",
        action="store_true",
        help="Omit LUT data from output (only needed LUTs are included by default)",
    )
    argparser.add_argument(
        "--no-scratch",
        action="store_true",
        help="Omit scratch memory declarations from output",
    )
    argparser.add_argument(
        "--lut-path",
        type=Path,
        default=Path(__file__).parent.parent / "lut",
        help="Path to LUT directory (default: ../lut)",
    )

    args = argparser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    output_path = args.output
    if output_path is None:
        output_path = args.input.with_suffix(".mov.s")

    try:
        translate_file(
            args.input,
            output_path,
            include_luts=not args.no_luts,
            include_scratch=not args.no_scratch,
            lut_path=args.lut_path,
        )
        print(f"Translated: {args.input} -> {output_path}")
        return 0
    except TranslatorError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
