# Movfuscator Architecture

Movfuscator translates x86-32 AT&T assembly into semantically equivalent
assembly that uses only MOV instructions (and variants). It is based on the
observation, proved by Stephen Dolan in 2013, that the x86 MOV instruction is
Turing-complete: any computation can be expressed as a sequence of memory reads,
lookup-table transformations, and memory writes - all of which MOV can perform.

## Translation pipeline

```
Source.s → Lexer → Tokens → Parser → AST → Translator → OutputFormatter → Output.s
                                              ↑              ↑
                                         LUTGenerator   ScratchMemory
```

An input assembly file passes through four stages:

1. **Lexer** - tokenises AT&T assembly text into a flat list of tokens.
2. **Parser** - builds an abstract syntax tree (AST) of statements from tokens.
3. **Translator** - rewrites every non-MOV instruction into an equivalent
   sequence of MOV instructions, using pre-computed lookup tables and scratch
   memory.
4. **OutputFormatter** - emits the final assembly, prepending scratch-memory
   declarations and whichever lookup tables the translated program actually
   needs.

Labels and assembler directives from the input pass through unchanged.
However, some translations (conditional jumps, LOOP, DIV) inject new labels
and `.section` directives for dispatch tables and loop constructs. Only
instructions are rewritten to MOV-only equivalents.

## The core idea: computation through table lookup

A traditional CPU computes `a + b` with ALU circuitry. Movfuscator instead
pre-computes every possible byte-level result into a 256×256 lookup table. To
add two bytes it performs two MOV instructions:

```asm
movl  add_row_ptrs(, %ecx, 4), %ecx   # row pointer for first byte value
movzbl (%ecx, %edx, 1), %eax           # look up result using second byte
```

For 32-bit values the translator processes all four bytes in sequence and
propagates carry between them. This byte-at-a-time strategy keeps table sizes
practical - a 256×256 table is 64 KB, whereas a 2³²×2³² table would be 16 TB.

## Lexer

`src/lexer.py`

The lexer scans raw assembly text with compiled regular expressions, matched in
priority order (most specific first). It produces a list of `Token` objects,
each carrying a type, value, line number, and column.

Token types include `DIRECTIVE` (`.data`, `.text`, `.global`, …),
`IDENTIFIER` (mnemonics, label names), `REGISTER` (`%eax`, `%bl`, …),
`IMMEDIATE` (`$5`, `$0xFF`, `$label`), `NUMBER` (bare numeric values used in
directives and memory displacements), `STRING`, `LPAREN`/`RPAREN`/`COMMA`
(memory addressing punctuation), `COLON`, `NEWLINE`, `COMMENT`, and `EOF`.

Comments and whitespace can be filtered out before parsing with
`Lexer.filter_tokens()`.

## Parser

`src/parser.py`

The parser consumes the token list and produces a `Program` - a list of
`Statement` nodes, each of which is an `Instruction`, `Label`, or `Directive`.

### AST node types

**Operands** - the building blocks of instruction arguments:

| Node | Fields | Example |
|------|--------|---------|
| `Register` | `name` (without `%`) | `eax`, `bl` |
| `Immediate` | `value`, `is_label` | `5`, `0xFF`, `label_name` |
| `Memory` | `displacement`, `base`, `index`, `scale` | `8(%ebp, %eax, 4)` |
| `LabelRef` | `name` | bare label in a jump |

**Statements:**

| Node | Fields |
|------|--------|
| `Instruction` | `mnemonic`, `operands: [src, dst]`, `size_suffix: OperandSize` |
| `Label` | `name` |
| `Directive` | `name` (without `.`), `args` |

`OperandSize` is an enum (`BYTE`, `WORD`, `LONG`, `QUAD`, `NONE`) extracted
from size suffixes on mnemonics - for example `addl` is parsed as mnemonic
`addl` with size `LONG`.

### Memory operand parsing

AT&T syntax encodes complex addressing as
`displacement(base, index, scale)`. The parser handles all combinations:

- `8(%ebp)` → displacement 8, base ebp
- `(%eax, %ebx, 4)` → base eax, index ebx, scale 4
- `label(%ebp)` → displacement "label", base ebp
- bare `label` → parsed as `LabelRef`, not `Memory`

## Scratch memory

`src/scratch.py`

Every non-MOV translation needs temporary storage. Scratch memory is a block of
labelled variables emitted in the `.data` section at the top of the output file.
The translator accesses them by name (e.g. `movl %eax, scratch_a`).

### Variables

| Group | Variables | Purpose |
|-------|-----------|---------|
| **Operands** | `scratch_a`, `scratch_b` (.long) | Hold the two operands being processed |
| **Result** | `scratch_r` (.long) | Holds the computed result |
| **Carry** | `scratch_c` (.byte) | Carry or borrow flag between byte steps |
| **Temporaries** | `scratch_t`, `scratch_t2` (.long) | General-purpose scratch space |
| **Register saves** | `save_eax`, `save_ecx`, `save_edx` (.long) | Preserve registers clobbered by the translation sequence |
| **Comparison** | `scratch_eq1`–`eq3`, `scratch_lt0`–`lt3` (.byte) | Per-byte equality and less-than results |
| **Comparison finals** | `scratch_cmp_eq`, `scratch_cmp_below`, `scratch_cmp_sign_lt` (.byte) | Combined comparison outcomes |
| **Multiplication** | `mul_op_a`, `mul_op_b` (.long), `mul_accum` (.quad) | 32×32→64-bit multiply workspace |
| **Division** | `div_remainder`, `div_divisor`, `div_quotient` (.long) | Long-division workspace |
| **Shift** | `shr_carry_inject`, `shr_carry_inject_hi` (.byte) | Carry injection constants for shift operations |
| **Control flow** | `current_block` (.long), `scratch_jcc_target` (.long) | Block dispatcher index and conditional jump target address |

32 variables totalling 93 bytes.

## Lookup tables

`src/lut_generator.py`, `lut/*.s`

The `lut/` directory contains 18 pre-computed assembly files plus a combined
`all_luts.s`. The individual files total a few MB. Each file defines either a
2D table (256×256 entries, one byte each) or a 1D table (256 entries).

### How 2D tables are laid out

A 2D table such as `add_lut` consists of a 256-entry pointer array followed by
256 rows of 256 bytes:

```asm
add_row_ptrs:
    .long add_row_0
    .long add_row_1
    ...
    .long add_row_255

add_row_0:   .byte 0, 1, 2, 3, ... 255    # 0 + 0..255
add_row_1:   .byte 1, 2, 3, 4, ... 0      # 1 + 0..255
...
add_row_255: .byte 255, 0, 1, 2, ... 254   # 255 + 0..255
```

This two-level layout exists because x86 addressing modes support a maximum
scale factor of 8, which is too small to index into a flat 256×256 array in one
step. Instead, two MOV instructions resolve a lookup:

```asm
movl   add_row_ptrs(, %ecx, 4), %ecx   # scale=4: pointer table (4-byte entries)
movzbl (%ecx, %edx, 1), %eax            # scale=1: row data (1-byte entries)
```

No arithmetic is needed - x86 hardware addressing does the work.

### 1D tables

A 1D table is a flat 256-byte array accessed with a single MOV:

```asm
movzbl is_zero_lut(, %ecx, 1), %eax   # result = is_zero_lut[value]
```

### Table catalogue

**2D tables (256×256):**

| Table | Formula | Used by |
|-------|---------|---------|
| `add_lut` | `(a + b) & 0xFF` | ADD, INC, SUB, POP, LEA, MUL, SHL, SHR, DIV |
| `carry_lut` | `1 if a + b > 255` | ADD, POP (via ADD), LEA (via ADD), MUL |
| `sub_lut` | `(a - b) & 0xFF` | SUB, DEC, PUSH, LOOP |
| `borrow_lut` | `1 if a < b` | SUB, PUSH (via SUB), DIV (via SUB) |
| `xor_lut` | `a ^ b` | XOR |
| `or_lut` | `a \| b` | OR, CMP, TEST, conditional jumps (flag combination) |
| `and_lut` | `a & b` | AND, TEST, CMP, INC, DEC, LOOP (flag/carry combination) |
| `mul8_lo_lut` | `(a * b) & 0xFF` | MUL |
| `mul8_hi_lut` | `(a * b) >> 8` | MUL |
| `je_lut` | `1 if a == b` | CMP |
| `jb_lut` | `1 if a < b (unsigned)` | CMP |
| `jl_signed_lut` | `1 if a < b (signed)` | CMP (MSB byte only) |

**1D tables (256):**

| Table | Formula | Used by |
|-------|---------|---------|
| `is_zero_lut` | `1 if a == 0` | Conditional jumps, LOOP, INC, DEC |
| `is_not_zero_lut` | `1 if a != 0` | ADD, SUB, DIV, PUSH, POP, LEA (carry propagation) |
| `shl1_lut` | `(a << 1) & 0xFF` | SHL |
| `shl1_carry_lut` | `(a >> 7) & 1` | SHL |
| `shr1_lut` | `a >> 1` | SHR |
| `shr1_carry_lut` | `a & 1` | SHR |

### Table generation

`LUTGenerator` in `src/lut_generator.py` generates every table from a lambda.
Tables are pre-generated and checked into the repository; regeneration is only
needed if a table definition changes.

## Translator

`src/translator/core.py`

The translator is implemented as a small package under `src/translator/`. It
walks the AST, passes MOV instructions through unchanged, and rewrites
everything else into MOV-only sequences. Each translated instruction follows a
common pattern:

1. **Save** temporary registers (`eax`, `ecx`, `edx`) to scratch memory.
2. **Store** operands into `scratch_a` / `scratch_b`. If an operand happens to
   be one of the temp registers, reload its original value from the saved
   location.
3. **Process** byte-by-byte (four bytes for 32-bit), looking up each byte pair
   in the appropriate LUT and propagating carry or borrow to the next byte.
4. **Store** the result to `scratch_r` (for arithmetic/logic) or to the
   `scratch_cmp_*` flags (for comparisons).
5. **Write back** the result to the destination operand. Comparisons skip this
   step since `CMP` only sets flags.
6. **Restore** saved registers, skipping the destination register if it received
   the result.

Some translations intentionally keep outputs in temp registers (e.g. `mul`
returns `edx:eax`, `div` returns `eax`/`edx`, `loop` updates `ecx`) and restore
only the registers that must be preserved.

### Supported instructions

| Category | Instructions |
|----------|-------------|
| Movement | `mov`, `movl`, `movb`, `movw`, `movzbl`, `movsbl`, `movzwl`, `movswl`, `movzbw`, `movsbw`, `lea` |
| Arithmetic | `add`, `sub`, `inc`, `dec`, `mul`, `div` |
| Logic | `xor`, `or`, `and` |
| Shift | `shl`, `sal`, `shr` |
| Comparison | `cmp`, `test` |
| Control flow | `jmp`, `je`/`jz`, `jne`/`jnz`, `jl`, `jle`, `jg`, `jge`, `jb`, `jbe`, `ja`, `jae`, `loop` |
| Stack | `push`, `pop`, `leave` |
| Misc | `nop` |
| Passthrough | `int`, `call`, `ret` (emitted unchanged) |

All arithmetic and logic translations operate on 32-bit values only.

### Worked example: ADD

`addl %eax, %ebx` produces about 80 MOV instructions:

```asm
# 1. Save temporary registers
movl %eax, save_eax
movl %ecx, save_ecx
movl %edx, save_edx

# 2. Load operands into scratch
movl %eax, scratch_a          # source
movl %ebx, scratch_b          # destination (also the accumulator)

# 3. Clear carry
movb $0, scratch_c

# 4. Process byte 0
movzbl scratch_a, %ecx                 # first byte of A
movl   add_row_ptrs(, %ecx, 4), %ecx   # LUT row pointer
movzbl scratch_b, %edx                 # first byte of B
movzbl (%ecx, %edx, 1), %eax           # add_lut[A0][B0]
movb   %al, scratch_r                  # store result byte 0

# 4a. Compute carry for byte 1
movzbl scratch_a, %ecx
movl   carry_row_ptrs(, %ecx, 4), %ecx
movzbl scratch_b, %edx
movzbl (%ecx, %edx, 1), %eax           # carry_lut[A0][B0]
movb   %al, scratch_c

# 4b. Process byte 1: add carry to A1, then add B1
movzbl scratch_a+1, %ecx
movl   add_row_ptrs(, %ecx, 4), %ecx
movzbl scratch_c, %edx
movzbl (%ecx, %edx, 1), %eax           # A1 + carry
# ... then add B1 to that result, compute new carry ...
# ... repeat for bytes 2 and 3 ...

# 5. Write result back to destination
movl scratch_r, %ebx

# 6. Restore temporary registers (skip ebx - it got the result)
movl save_eax, %eax
movl save_ecx, %ecx
movl save_edx, %edx
```

### How comparison works

`CMP` is the most complex translation (~90 MOV instructions). It compares two
32-bit values by computing per-byte equality and less-than flags, then combining
them.

**Per-byte step (repeated for bytes 0–3):**

```asm
movzbl scratch_b+N, %ecx
movl   je_row_ptrs(, %ecx, 4), %ecx
movzbl scratch_a+N, %edx
movzbl (%ecx, %edx, 1), %eax       # 1 if dst byte == src byte
movb   %al, scratch_eqN             # byte 0 goes to scratch_t, bytes 1-3 to scratch_eq1-3

movzbl scratch_b+N, %ecx
movl   jb_row_ptrs(, %ecx, 4), %ecx
movzbl scratch_a+N, %edx
movzbl (%ecx, %edx, 1), %eax       # 1 if dst byte < src byte (unsigned)
movb   %al, scratch_ltN
```

**Combining the per-byte results:**

Equality requires all four bytes to match:

```
eq = AND(eq3, AND(eq2, AND(eq1, scratch_t)))
```

Unsigned less-than checks from the most significant byte down: if the high bytes
are equal, the next byte decides, and so on:

```
below = lt3 OR (eq3 AND (lt2 OR (eq2 AND (lt1 OR (eq1 AND lt0)))))
```

Signed less-than works similarly but uses `jl_signed_lut` for the most
significant byte (where the sign bit lives) and unsigned comparison for the
rest.

The AND and OR LUT tables combine these per-byte flags without any arithmetic.
Final results land in `scratch_cmp_eq`, `scratch_cmp_below`, and
`scratch_cmp_sign_lt`.

### How conditional jumps work

A conditional jump like `jge label` cannot use CPU flags - they were never set.
Instead it reads the comparison results from scratch memory and dispatches
through a two-entry lookup table:

```asm
# Emitted in the .data section
_jcc_table_0:
    .long _jcc_fallthrough_0    # index 0: condition false
    .long label                 # index 1: condition true

# Emitted in the .text section
movl   %eax, save_eax
movzbl scratch_cmp_sign_lt, %eax          # load the relevant flag
movzbl is_zero_lut(, %eax, 1), %eax       # invert: 0→1, non-zero→0
movl   _jcc_table_0(, %eax, 4), %eax      # pick target address
movl   %eax, scratch_jcc_target
movl   save_eax, %eax
jmp    *scratch_jcc_target                 # indirect dispatch
_jcc_fallthrough_0:                        # execution continues here if false
```

Each condition type reads the appropriate scratch flag and applies `is_zero_lut`
or `is_not_zero_lut` to convert it to a 0-or-1 table index.

### Other translations in brief

| Instruction | Strategy |
|-------------|----------|
| **XOR / OR / AND** | Same byte-by-byte loop as ADD, but no carry propagation - each byte is independent. |
| **INC / DEC** | Sets an initial carry/borrow of 1 and processes all four bytes using `add_lut[byte][carry]` (INC) or `sub_lut[byte][borrow]` (DEC). Carry/borrow propagation uses `is_zero_lut` and `and_lut`. |
| **MUL** | 32×32→64-bit multiplication decomposed into 16 partial 8×8 products using `mul8_lo_lut` and `mul8_hi_lut`, accumulated with carry. |
| **DIV** | Repeated subtraction loop using CMP/DEC patterns to build the quotient. |
| **SHL / SHR** | Unrolled loop of 1-bit shifts using `shl1_lut` / `shr1_lut` with carry propagation between bytes. |
| **PUSH** | Subtract 4 from ESP (via `sub_lut`), then MOV value to `(%esp)`. |
| **POP** | MOV `(%esp)` to destination, then add 4 to ESP (via `add_lut`). |
| **LEA** | Computes the effective address by delegating to ADD for base + displacement and SHL for index × scale (scale 2/4/8 become left-shifts of 1/2/3). |
| **LOOP** | Decrement ECX, test for zero, dispatch via jump table (combines DEC and conditional-jump patterns). |
| **JMP** | Passed through unchanged. |
| **INT / CALL / RET** | Passed through unchanged - these are system interface instructions that cannot be expressed as MOV. |

### Register preservation

Every translation clobbers `eax`, `ecx`, and `edx` for LUT indexing. To keep
this invisible to the surrounding code:

- All three are saved to `save_eax`, `save_ecx`, `save_edx` at the start.
- If an operand *is* one of these registers, the translator loads from the saved
  copy instead.
- At the end, all three are restored - except the destination register, which
  already holds the correct result.

### Smart LUT inclusion

The translator infers which lookup tables are required by scanning the emitted translated AST for LUT label references (e.g. `add_row_ptrs`, `is_zero_lut`). The output formatter then includes only the corresponding `lut/*_lut.s` files. This keeps output self-contained while avoiding unnecessary LUT payload.

## Output formatter

`OutputFormatter` (defined in `src/translator/output.py`) assembles the final output:

1. **Scratch memory** - the `.data` section with all scratch variable
   declarations.
2. **Lookup tables** - only the `.s` files for tables the program needs.
3. **Translated code** - the `.text` section with the rewritten instructions.

Configuration is held in `TranslatorConfig` (see `src/translator/config.py`):

```python
TranslatorConfig(
    include_luts=True,       # prepend LUT data
    include_scratch=True,    # prepend scratch memory
    lut_path=Path("lut")     # directory containing LUT .s files
)
```

## Limitations

- **32-bit only.** Arithmetic and logic translations operate on `.long` (32-bit)
  values. 8-bit and 16-bit variants are not yet supported.
- **No floating point.** There is no SSE or x87 support.
- **Passthrough instructions.** `int`, `call`, and `ret` are emitted as-is;
  they are part of the system calling convention and cannot be replaced with MOV.
- **Code size.** A single `addl` expands to about 80 MOV instructions. A `cmpl`
  followed by a conditional jump produces about 100. Programs grow substantially.

## Project layout

| File | Purpose |
|------|---------|
| `src/lexer.py` | Tokeniser |
| `src/parser.py` | AST builder |
| `src/translator/` | Translator package (`core.py`, `output.py`, `config.py`) |
| `src/scratch.py` | Scratch memory variable definitions |
| `src/lut_generator.py` | Lookup table generator |
| `src/main.py` | CLI entry point |
| `lut/*.s` | Pre-computed lookup tables (18 individual + 1 combined, a few MB) |
| `samples/in/` | Test input programs |
| `test.sh` | Integration test runner |
