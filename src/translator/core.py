"""Translates x86 AST into MOV-only assembly AST."""

from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

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
    OperandSize,
)

from .config import TranslatorConfig, TranslatorError


class Translator:
    """Translates x86 assembly to MOV-only assembly."""

    MOV_MNEMONICS = frozenset(
        {
            "mov",
            "movl",
            "movb",
            "movw",
            "movzbl",
            "movsbl",
            "movzwl",
            "movswl",
            "movzbw",
            "movsbw",
        }
    )

    ADD_MNEMONICS_32 = frozenset({"add", "addl"})
    ADD_MNEMONICS_UNSUPPORTED = frozenset({"addb", "addw", "addq"})

    SUB_MNEMONICS_32 = frozenset({"sub", "subl"})
    SUB_MNEMONICS_UNSUPPORTED = frozenset({"subb", "subw", "subq"})

    XOR_MNEMONICS_32 = frozenset({"xor", "xorl"})
    XOR_MNEMONICS_UNSUPPORTED = frozenset({"xorb", "xorw", "xorq"})

    OR_MNEMONICS_32 = frozenset({"or", "orl"})
    OR_MNEMONICS_UNSUPPORTED = frozenset({"orb", "orw", "orq"})

    AND_MNEMONICS_32 = frozenset({"and", "andl"})
    AND_MNEMONICS_UNSUPPORTED = frozenset({"andb", "andw", "andq"})

    INC_MNEMONICS_32 = frozenset({"inc", "incl"})
    INC_MNEMONICS_UNSUPPORTED = frozenset({"incb", "incw", "incq"})

    DEC_MNEMONICS_32 = frozenset({"dec", "decl"})
    DEC_MNEMONICS_UNSUPPORTED = frozenset({"decb", "decw", "decq"})

    CMP_MNEMONICS_32 = frozenset({"cmp", "cmpl"})
    CMP_MNEMONICS_UNSUPPORTED = frozenset({"cmpb", "cmpw", "cmpq"})

    TEST_MNEMONICS_32 = frozenset({"test", "testl"})
    TEST_MNEMONICS_UNSUPPORTED = frozenset({"testb", "testw", "testq"})

    # Signed conditional jumps.
    #
    # Includes common aliases used by assemblers:
    # - jnge == jl
    # - jnl  == jge
    # - jng  == jle
    # - jnle == jg
    JCC_SIGNED_MNEMONICS = frozenset(
        {"jl", "jle", "jg", "jge", "jnge", "jnl", "jng", "jnle"}
    )

    # Unsigned conditional jumps.
    #
    # Includes common aliases used by assemblers:
    # - jc / jnae  == jb
    # - jna        == jbe
    # - jnbe       == ja
    # - jnb / jnc  == jae
    JCC_UNSIGNED_MNEMONICS = frozenset(
        {"jb", "jbe", "ja", "jae", "jc", "jnae", "jna", "jnbe", "jnb", "jnc"}
    )

    # Equality conditional jumps
    JCC_EQUALITY_MNEMONICS = frozenset({"je", "jne", "jz", "jnz"})

    # Unconditional jump (passthrough)
    JMP_MNEMONICS = frozenset({"jmp"})

    # LOOP instruction (decrement ECX, jump if not zero)
    LOOP_MNEMONICS = frozenset({"loop"})

    MUL_MNEMONICS_32 = frozenset({"mul", "mull"})
    MUL_MNEMONICS_UNSUPPORTED = frozenset({"mulb", "mulw", "mulq"})

    DIV_MNEMONICS_32 = frozenset({"div", "divl"})
    DIV_MNEMONICS_UNSUPPORTED = frozenset({"divb", "divw", "divq"})

    SHL_MNEMONICS_32 = frozenset({"shl", "shll"})
    SHL_MNEMONICS_UNSUPPORTED = frozenset({"shlb", "shlw", "shlq"})

    # SAL is an alias for SHL
    SAL_MNEMONICS_32 = frozenset({"sal", "sall"})
    SAL_MNEMONICS_UNSUPPORTED = frozenset({"salb", "salw", "salq"})

    SHR_MNEMONICS_32 = frozenset({"shr", "shrl"})
    SHR_MNEMONICS_UNSUPPORTED = frozenset({"shrb", "shrw", "shrq"})

    PUSH_MNEMONICS_32 = frozenset({"push", "pushl"})
    PUSH_MNEMONICS_UNSUPPORTED = frozenset({"pushb", "pushw", "pushq"})
    POP_MNEMONICS_32 = frozenset({"pop", "popl"})
    POP_MNEMONICS_UNSUPPORTED = frozenset({"popb", "popw", "popq"})

    # LEA (Load Effective Address) - translates to MOV with immediate address
    LEA_MNEMONICS = frozenset({"lea", "leal"})

    # Function epilogue helper
    LEAVE_MNEMONICS = frozenset({"leave"})

    # No-op
    NOP_MNEMONICS = frozenset({"nop"})

    # Passthrough instructions (emitted as-is, cannot be MOV-ified)
    PASSTHROUGH_MNEMONICS = frozenset({"int", "call", "ret"})

    # Temp registers used internally - must be saved/restored
    TEMP_REGS = frozenset({"eax", "ecx", "edx"})

    def __init__(self, config: Optional[TranslatorConfig] = None):
        self.config = config or TranslatorConfig()
        self._output_statements: List[Statement] = []
        self._jcc_counter: int = 0
        self._required_luts: Set[str] = set()
        self._dispatch, self._unsupported = self._build_dispatch_tables()

    def _build_dispatch_tables(
        self,
    ) -> Tuple[
        Dict[str, Callable[[Instruction], List[Statement]]],
        Dict[str, Tuple[str, str]],
    ]:
        dispatch: Dict[str, Callable[[Instruction], List[Statement]]] = {}
        unsupported: Dict[str, Tuple[str, str]] = {}

        def add_all(
            mnems: Iterable[str],
            handler: Callable[[Instruction], List[Statement]],
        ) -> None:
            for m in mnems:
                dispatch[m] = handler

        def add_unsupported(mnems: Iterable[str], op: str, hint: str) -> None:
            for m in mnems:
                unsupported[m] = (op, hint)

        add_all(self.NOP_MNEMONICS, self._translate_nop)
        add_all(self.MOV_MNEMONICS, self._translate_passthrough)
        add_all(self.JMP_MNEMONICS, self._translate_passthrough)
        add_all(self.PASSTHROUGH_MNEMONICS, self._translate_passthrough)

        add_all(self.ADD_MNEMONICS_32, self._translate_add)
        add_unsupported(self.ADD_MNEMONICS_UNSUPPORTED, "ADD", "addl")

        add_all(self.SUB_MNEMONICS_32, self._translate_sub)
        add_unsupported(self.SUB_MNEMONICS_UNSUPPORTED, "SUB", "subl")

        add_all(self.XOR_MNEMONICS_32, self._translate_xor)
        add_unsupported(self.XOR_MNEMONICS_UNSUPPORTED, "XOR", "xorl")

        add_all(self.OR_MNEMONICS_32, self._translate_or)
        add_unsupported(self.OR_MNEMONICS_UNSUPPORTED, "OR", "orl")

        add_all(self.AND_MNEMONICS_32, self._translate_and)
        add_unsupported(self.AND_MNEMONICS_UNSUPPORTED, "AND", "andl")

        add_all(self.INC_MNEMONICS_32, self._translate_inc)
        add_unsupported(self.INC_MNEMONICS_UNSUPPORTED, "INC", "incl")

        add_all(self.DEC_MNEMONICS_32, self._translate_dec)
        add_unsupported(self.DEC_MNEMONICS_UNSUPPORTED, "DEC", "decl")

        add_all(self.CMP_MNEMONICS_32, self._translate_cmp)
        add_unsupported(self.CMP_MNEMONICS_UNSUPPORTED, "CMP", "cmpl")

        add_all(self.TEST_MNEMONICS_32, self._translate_test)
        add_unsupported(self.TEST_MNEMONICS_UNSUPPORTED, "TEST", "testl")

        add_all(self.JCC_SIGNED_MNEMONICS, self._translate_jcc_signed)
        add_all(self.JCC_UNSIGNED_MNEMONICS, self._translate_jcc_unsigned)
        add_all(self.JCC_EQUALITY_MNEMONICS, self._translate_jcc_equality)

        add_all(self.LOOP_MNEMONICS, self._translate_loop)

        add_all(self.MUL_MNEMONICS_32, self._translate_mul)
        add_unsupported(self.MUL_MNEMONICS_UNSUPPORTED, "MUL", "mull")

        add_all(self.DIV_MNEMONICS_32, self._translate_div)
        add_unsupported(self.DIV_MNEMONICS_UNSUPPORTED, "DIV", "divl")

        add_all(self.SHL_MNEMONICS_32, self._translate_shl)
        add_all(self.SAL_MNEMONICS_32, self._translate_shl)
        add_unsupported(self.SHL_MNEMONICS_UNSUPPORTED, "SHL", "shll")
        add_unsupported(self.SAL_MNEMONICS_UNSUPPORTED, "SAL", "sall")

        add_all(self.SHR_MNEMONICS_32, self._translate_shr)
        add_unsupported(self.SHR_MNEMONICS_UNSUPPORTED, "SHR", "shrl")

        add_all(self.PUSH_MNEMONICS_32, self._translate_push)
        add_unsupported(self.PUSH_MNEMONICS_UNSUPPORTED, "PUSH", "pushl")
        add_all(self.POP_MNEMONICS_32, self._translate_pop)
        add_unsupported(self.POP_MNEMONICS_UNSUPPORTED, "POP", "popl")

        add_all(self.LEA_MNEMONICS, self._translate_lea)
        add_all(self.LEAVE_MNEMONICS, self._translate_leave)

        return dispatch, unsupported

    def _translate_nop(self, instr: Instruction) -> List[Statement]:
        return []

    def _translate_passthrough(self, instr: Instruction) -> List[Statement]:
        return [instr]

    @property
    def required_luts(self) -> Set[str]:
        """Return the set of LUT base names required by the translated program."""
        return set(self._required_luts)

    @staticmethod
    def _scan_required_luts(program: Program) -> Set[str]:
        """Infer required LUT base names by scanning emitted operands."""
        required: Set[str] = set()

        def maybe_add_from_disp(disp: Optional[str]) -> None:
            if not disp:
                return
            base = disp.split("+", 1)[0]
            if base.endswith("_row_ptrs"):
                required.add(base[: -len("_row_ptrs")])
                return
            if base.endswith("_lut"):
                required.add(base[: -len("_lut")])
                return

        def walk_operand(op: Operand) -> None:
            if isinstance(op, Memory):
                if isinstance(op.displacement, str):
                    maybe_add_from_disp(op.displacement)
                return

        for stmt in program.statements:
            if isinstance(stmt, Instruction):
                for op in stmt.operands:
                    walk_operand(op)

        return required

    def translate(self, program: Program) -> Program:
        self._output_statements = []
        self._required_luts = set()

        for stmt in program.statements:
            translated = self._translate_statement(stmt)
            self._output_statements.extend(translated)

        out = Program(statements=self._output_statements)
        self._required_luts = self._scan_required_luts(out)
        return out

    def _translate_statement(self, stmt: Statement) -> List[Statement]:
        if isinstance(stmt, Label):
            return [stmt]

        if isinstance(stmt, Directive):
            return [stmt]

        if isinstance(stmt, Instruction):
            return self._translate_instruction(stmt)

        return [stmt]

    def _translate_instruction(self, instr: Instruction) -> List[Statement]:
        mnemonic = instr.mnemonic.lower()

        handler = self._dispatch.get(mnemonic)
        if handler is not None:
            return handler(instr)

        unsupported = self._unsupported.get(mnemonic)
        if unsupported is not None:
            op, hint = unsupported
            raise TranslatorError(
                f"Only 32-bit {op} ({hint}) is supported, got: {mnemonic}", instr.line
            )

        raise TranslatorError(f"Unsupported instruction: {mnemonic}", instr.line)

    def _translate_leave(self, instr: Instruction) -> List[Statement]:
        """Translate LEAVE to MOV-only sequence.

        LEAVE  =>  movl %ebp, %esp; popl %ebp

        This matches the common x86-32 function epilogue and intentionally does
        not update scratch-based flags.
        """

        if instr.operands:
            raise TranslatorError(
                f"LEAVE takes no operands, got {len(instr.operands)}", instr.line
            )

        statements: List[Statement] = []
        statements.append(
            Instruction(
                mnemonic="movl",
                operands=[Register("ebp"), Register("esp")],
                line=instr.line,
            )
        )

        pop_ebp = Instruction(
            mnemonic="popl",
            operands=[Register("ebp")],
            size_suffix=OperandSize.LONG,
            line=instr.line,
        )
        statements.extend(self._translate_pop(pop_ebp))
        return statements

    def _emit_update_zf(self, statements: List[Statement], line: int) -> None:
        """Update scratch_cmp_eq (ZF) based on scratch_r.

        Computes: scratch_cmp_eq = 1 if all 4 bytes of scratch_r are zero, else 0.
        Uses: eax, ecx, edx (must already be saved by caller).
        """

        def mov(mnem: str, s: Operand, d: Operand) -> Instruction:
            return Instruction(mnemonic=mnem, operands=[s, d], line=line)

        # OR all 4 bytes of scratch_r together
        statements.append(
            mov("movzbl", Memory(displacement="scratch_r"), Register("ecx"))
        )
        statements.append(
            mov(
                "movl",
                Memory(displacement="or_row_ptrs", index=Register("ecx"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_r+1"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # OR with byte 2
        statements.append(
            mov(
                "movl",
                Memory(displacement="or_row_ptrs", index=Register("eax"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_r+2"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # OR with byte 3
        statements.append(
            mov(
                "movl",
                Memory(displacement="or_row_ptrs", index=Register("eax"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_r+3"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # Apply is_zero_lut: 1 if all bytes were zero (ZF=1)
        statements.append(
            mov(
                "movzbl",
                Memory(displacement="is_zero_lut", index=Register("eax"), scale=1),
                Register("eax"),
            )
        )

        # Store to scratch_cmp_eq
        statements.append(
            mov("movb", Register("al"), Memory(displacement="scratch_cmp_eq"))
        )

    @staticmethod
    def _normalize_operand(op: Operand) -> Operand:
        """Convert LabelRef operands to Memory (bare labels are memory references)."""
        if isinstance(op, LabelRef):
            return Memory(displacement=op.name)
        return op

    @staticmethod
    def _mov_instr(
        line: int, mnemonic: str, src_op: Operand, dst_op: Operand
    ) -> Instruction:
        return Instruction(mnemonic=mnemonic, operands=[src_op, dst_op], line=line)

    @staticmethod
    def _scratch_mem(name: str, byte_offset: int = 0) -> Memory:
        if byte_offset == 0:
            return Memory(displacement=name)
        return Memory(displacement=f"{name}+{byte_offset}")

    @staticmethod
    def _temp_save_locations() -> Dict[str, Memory]:
        return {
            "eax": Memory(displacement="save_eax"),
            "ecx": Memory(displacement="save_ecx"),
            "edx": Memory(displacement="save_edx"),
        }

    def _emit_save_temps(
        self, statements: List[Statement], line: int
    ) -> Dict[str, Memory]:
        save_map = self._temp_save_locations()
        statements.append(
            self._mov_instr(line, "movl", Register("eax"), save_map["eax"])
        )
        statements.append(
            self._mov_instr(line, "movl", Register("ecx"), save_map["ecx"])
        )
        statements.append(
            self._mov_instr(line, "movl", Register("edx"), save_map["edx"])
        )
        return save_map

    def _emit_restore_temps(
        self,
        statements: List[Statement],
        line: int,
        dst_reg_name: Optional[str],
        save_map: Dict[str, Memory],
    ) -> None:
        exclude: Set[str] = set()
        if dst_reg_name is not None:
            exclude.add(dst_reg_name)
        self._emit_restore_temps_except(statements, line, exclude, save_map)

    def _emit_restore_temps_except(
        self,
        statements: List[Statement],
        line: int,
        exclude: Set[str],
        save_map: Dict[str, Memory],
    ) -> None:
        for reg in ("eax", "ecx", "edx"):
            if reg in exclude:
                continue
            statements.append(
                self._mov_instr(line, "movl", save_map[reg], Register(reg))
            )

    def _emit_store_operand_long_to_memory(
        self,
        statements: List[Statement],
        src: Operand,
        dst_mem: Memory,
        line: int,
        save_map: Dict[str, Memory],
    ) -> None:
        """Store a 32-bit value from src into dst_mem.

        Uses save_* for eax/ecx/edx operands (assumes caller already saved temps).
        """
        if isinstance(src, Register):
            if src.name in self.TEMP_REGS:
                statements.append(
                    self._mov_instr(line, "movl", save_map[src.name], Register("eax"))
                )
                statements.append(
                    self._mov_instr(line, "movl", Register("eax"), dst_mem)
                )
            else:
                statements.append(self._mov_instr(line, "movl", src, dst_mem))
            return

        if isinstance(src, Immediate):
            statements.append(self._mov_instr(line, "movl", src, dst_mem))
            return

        if isinstance(src, Memory):
            statements.append(self._mov_instr(line, "movl", src, Register("eax")))
            statements.append(self._mov_instr(line, "movl", Register("eax"), dst_mem))
            return

        raise TranslatorError("Unsupported operand type", line)

    def _emit_store_operand_long_to_scratch(
        self,
        statements: List[Statement],
        src: Operand,
        scratch_name: str,
        line: int,
        save_map: Dict[str, Memory],
    ) -> None:
        """Store a 32-bit value from src into scratch memory.

        Uses save_* for eax/ecx/edx operands (assumes caller already saved temps).
        """
        self._emit_store_operand_long_to_memory(
            statements,
            src,
            self._scratch_mem(scratch_name),
            line,
            save_map,
        )

    def _validate_32bit_operand(
        self, op: Operand, instr: Instruction, context: str
    ) -> None:
        if isinstance(op, Register):
            if op.size != OperandSize.LONG:
                raise TranslatorError(
                    f"{context} requires 32-bit operands; got register %{op.name}",
                    instr.line,
                )
        elif isinstance(op, Memory):
            if op.base and op.base.size != OperandSize.LONG:
                raise TranslatorError(
                    f"{context} requires 32-bit base register; got %{op.base.name}",
                    instr.line,
                )
            if op.index and op.index.size != OperandSize.LONG:
                raise TranslatorError(
                    f"{context} requires 32-bit index register; got %{op.index.name}",
                    instr.line,
                )

    def _validate_32bit_operands(
        self, instr: Instruction, operands: List[Operand], context: str
    ) -> None:
        # Reject explicit non-32-bit suffixes.
        if instr.size_suffix in (OperandSize.BYTE, OperandSize.WORD, OperandSize.QUAD):
            raise TranslatorError(
                f"Only 32-bit {context} is supported, got size suffix {instr.size_suffix.value!r}",
                instr.line,
            )

        for op in operands:
            self._validate_32bit_operand(op, instr, context)

    def _emit_store_long_to_memory(
        self,
        statements: List[Statement],
        src_mem: Memory,
        dst_mem: Memory,
        save_eax: Memory,
        save_ecx: Memory,
        save_edx: Memory,
        line: int,
    ) -> None:
        """Store a .long value from src_mem into dst_mem.

        If dst_mem uses one of the temp regs (eax/ecx/edx) as base/index, this
        restores the needed address regs from save_* and uses an unused temp reg
        for the value to avoid clobbering the address.
        """

        addr_regs: Set[str] = set()
        if (
            dst_mem.base
            and isinstance(dst_mem.base, Register)
            and dst_mem.base.name in self.TEMP_REGS
        ):
            addr_regs.add(dst_mem.base.name)
        if (
            dst_mem.index
            and isinstance(dst_mem.index, Register)
            and dst_mem.index.name in self.TEMP_REGS
        ):
            addr_regs.add(dst_mem.index.name)

        save_map = {"eax": save_eax, "ecx": save_ecx, "edx": save_edx}
        for reg in sorted(addr_regs):
            statements.append(
                self._mov_instr(line, "movl", save_map[reg], Register(reg))
            )

        value_reg = next(r for r in ("eax", "ecx", "edx") if r not in addr_regs)
        statements.append(self._mov_instr(line, "movl", src_mem, Register(value_reg)))
        statements.append(self._mov_instr(line, "movl", Register(value_reg), dst_mem))

    def _translate_add(
        self,
        instr: Instruction,
        update_flags: bool = True,
        save_temps: bool = True,
    ) -> List[Statement]:
        """Translate ADD src, dst to a MOV-only sequence."""
        if len(instr.operands) != 2:
            raise TranslatorError(
                f"ADD requires 2 operands, got {len(instr.operands)}", instr.line
            )

        src, dst = (
            self._normalize_operand(instr.operands[0]),
            self._normalize_operand(instr.operands[1]),
        )

        self._validate_32bit_operands(instr, [src, dst], "ADD")

        statements: List[Statement] = []
        line = instr.line

        def mov(mnemonic: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnemonic, src_op, dst_op)

        scratch_r = self._scratch_mem("scratch_r")
        scratch_c = self._scratch_mem("scratch_c")

        # Determine which register is the destination (if any)
        dst_reg_name = dst.name if isinstance(dst, Register) else None

        save_map = (
            self._emit_save_temps(statements, line)
            if save_temps
            else self._temp_save_locations()
        )

        self._emit_store_operand_long_to_scratch(
            statements, src, "scratch_a", line, save_map
        )

        if isinstance(dst, Immediate):
            raise TranslatorError("ADD destination must be register or memory", line)
        self._emit_store_operand_long_to_scratch(
            statements, dst, "scratch_b", line, save_map
        )

        statements.append(mov("movb", Immediate(0), scratch_c))

        for byte_idx in range(4):
            a_byte = self._scratch_mem("scratch_a", byte_idx)
            b_byte = self._scratch_mem("scratch_b", byte_idx)
            r_byte = self._scratch_mem("scratch_r", byte_idx)

            statements.append(mov("movzbl", a_byte, Register("ecx")))

            # Get add_lut row pointer: add_row_ptrs(, %ecx, 4)
            add_row_ptr = Memory(
                displacement="add_row_ptrs", index=Register("ecx"), scale=4
            )
            statements.append(mov("movl", add_row_ptr, Register("ecx")))

            # Load byte from scratch_b into edx (zero-extended)
            statements.append(mov("movzbl", b_byte, Register("edx")))

            # Lookup result: (%ecx, %edx, 1)
            add_lookup = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
            statements.append(mov("movzbl", add_lookup, Register("eax")))

            # Now we need to add the carry from previous byte
            if byte_idx > 0:
                # If carry was set, we need to add 1 to the result
                # We do this by looking up add_lut[result][carry]
                # Since carry is 0 or 1, this effectively adds 0 or 1

                # Load carry into edx
                statements.append(mov("movzbl", scratch_c, Register("edx")))

                # Get row pointer for current result
                add_row_ptr2 = Memory(
                    displacement="add_row_ptrs", index=Register("eax"), scale=4
                )
                statements.append(mov("movl", add_row_ptr2, Register("ecx")))

                # Lookup add_lut[result][carry]
                add_carry_lookup = Memory(
                    base=Register("ecx"), index=Register("edx"), scale=1
                )
                statements.append(mov("movzbl", add_carry_lookup, Register("eax")))

            # Store result byte
            statements.append(mov("movb", Register("al"), r_byte))

            # Calculate carry for next byte (if not last byte)
            if byte_idx < 3:
                # Carry = carry_lut[a_byte][b_byte]
                # But we also need to handle carry chain from adding previous carry

                # Reload a_byte and b_byte
                statements.append(mov("movzbl", a_byte, Register("ecx")))
                carry_row_ptr = Memory(
                    displacement="carry_row_ptrs", index=Register("ecx"), scale=4
                )
                statements.append(mov("movl", carry_row_ptr, Register("ecx")))
                statements.append(mov("movzbl", b_byte, Register("edx")))

                # Lookup carry_lut[a][b]
                carry_lookup = Memory(
                    base=Register("ecx"), index=Register("edx"), scale=1
                )
                statements.append(mov("movzbl", carry_lookup, Register("eax")))

                if byte_idx > 0:
                    # Combine carry from (a+b) and carry from (+old_carry).

                    # Store carry1 in scratch_t
                    scratch_t = Memory(displacement="scratch_t")
                    statements.append(mov("movb", Register("al"), scratch_t))

                    # Compute carry_out = carry_lut[sum][old_carry] OR carry_lut[a][b]
                    statements.append(mov("movzbl", r_byte, Register("ecx")))

                    # Load add_lut[a][b] (sum without carry added)
                    statements.append(mov("movzbl", a_byte, Register("ecx")))
                    add_row_ptr3 = Memory(
                        displacement="add_row_ptrs", index=Register("ecx"), scale=4
                    )
                    statements.append(mov("movl", add_row_ptr3, Register("ecx")))
                    statements.append(mov("movzbl", b_byte, Register("edx")))
                    sum_lookup = Memory(
                        base=Register("ecx"), index=Register("edx"), scale=1
                    )
                    statements.append(mov("movzbl", sum_lookup, Register("ecx")))

                    # Get carry_lut[sum][old_carry]
                    carry_row_ptr2 = Memory(
                        displacement="carry_row_ptrs", index=Register("ecx"), scale=4
                    )
                    statements.append(mov("movl", carry_row_ptr2, Register("ecx")))
                    statements.append(mov("movzbl", scratch_c, Register("edx")))
                    carry_from_carry = Memory(
                        base=Register("ecx"), index=Register("edx"), scale=1
                    )
                    statements.append(mov("movzbl", carry_from_carry, Register("eax")))

                    # OR for 0/1 is equivalent to (a+b)!=0

                    # Load scratch_t (carry1)
                    statements.append(mov("movzbl", scratch_t, Register("edx")))

                    # Use add_lut to add carry1 + carry2
                    add_row_ptr4 = Memory(
                        displacement="add_row_ptrs", index=Register("eax"), scale=4
                    )
                    statements.append(mov("movl", add_row_ptr4, Register("ecx")))
                    or_result = Memory(
                        base=Register("ecx"), index=Register("edx"), scale=1
                    )
                    statements.append(mov("movzbl", or_result, Register("eax")))

                    # If result > 0, carry = 1, else carry = 0
                    # Use is_not_zero_lut
                    is_nz = Memory(
                        displacement="is_not_zero_lut", index=Register("eax"), scale=1
                    )
                    statements.append(mov("movzbl", is_nz, Register("eax")))

                # Store carry for next byte
                statements.append(mov("movb", Register("al"), scratch_c))

        if update_flags:
            self._emit_update_zf(statements, line)

        if isinstance(dst, Register):
            statements.append(mov("movl", scratch_r, dst))
        elif isinstance(dst, Memory):
            self._emit_store_long_to_memory(
                statements,
                scratch_r,
                dst,
                save_map["eax"],
                save_map["ecx"],
                save_map["edx"],
                line,
            )

        if save_temps:
            self._emit_restore_temps(statements, line, dst_reg_name, save_map)

        return statements

    def _translate_logical_op(
        self, instr: Instruction, lut_name: str, op_name: str, update_flags: bool = True
    ) -> List[Statement]:
        """Translate a logical op (XOR/OR/AND) to a MOV-only sequence."""
        if len(instr.operands) != 2:
            raise TranslatorError(
                f"{op_name} requires 2 operands, got {len(instr.operands)}", instr.line
            )

        src, dst = (
            self._normalize_operand(instr.operands[0]),
            self._normalize_operand(instr.operands[1]),
        )

        self._validate_32bit_operands(instr, [src, dst], op_name)

        statements: List[Statement] = []
        line = instr.line

        def mov(mnemonic: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnemonic, src_op, dst_op)

        scratch_r = self._scratch_mem("scratch_r")

        # Determine which register is the destination (if any)
        dst_reg_name = dst.name if isinstance(dst, Register) else None

        save_map = self._emit_save_temps(statements, line)

        self._emit_store_operand_long_to_scratch(
            statements, src, "scratch_a", line, save_map
        )

        if isinstance(dst, Immediate):
            raise TranslatorError(
                f"{op_name} destination must be register or memory", line
            )
        self._emit_store_operand_long_to_scratch(
            statements, dst, "scratch_b", line, save_map
        )

        lut_row_ptrs = f"{lut_name}_row_ptrs"
        for byte_idx in range(4):
            a_byte = self._scratch_mem("scratch_a", byte_idx)
            b_byte = self._scratch_mem("scratch_b", byte_idx)
            r_byte = self._scratch_mem("scratch_r", byte_idx)

            statements.append(mov("movzbl", a_byte, Register("ecx")))

            lut_row_ptr = Memory(
                displacement=lut_row_ptrs, index=Register("ecx"), scale=4
            )
            statements.append(mov("movl", lut_row_ptr, Register("ecx")))

            statements.append(mov("movzbl", b_byte, Register("edx")))

            lut_lookup = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
            statements.append(mov("movzbl", lut_lookup, Register("eax")))

            statements.append(mov("movb", Register("al"), r_byte))

        if update_flags:
            self._emit_update_zf(statements, line)

        if isinstance(dst, Register):
            statements.append(mov("movl", scratch_r, dst))
        elif isinstance(dst, Memory):
            self._emit_store_long_to_memory(
                statements,
                scratch_r,
                dst,
                save_map["eax"],
                save_map["ecx"],
                save_map["edx"],
                line,
            )

        self._emit_restore_temps(statements, line, dst_reg_name, save_map)

        return statements

    def _translate_xor(self, instr: Instruction) -> List[Statement]:
        """Translate XOR instruction to MOV-only sequence.

        Special case: XOR %reg, %reg (same register) is a common idiom for zeroing
        a register, which we optimize to a single MOV.
        """
        if len(instr.operands) == 2:
            src, dst = instr.operands
            # Check for XOR %reg, %reg (self-XOR to zero)
            if (
                isinstance(src, Register)
                and isinstance(dst, Register)
                and src.name == dst.name
            ):
                self._validate_32bit_operands(instr, [src, dst], "XOR")
                # XOR %reg, %reg => MOV $0, %reg
                return [
                    Instruction(
                        mnemonic="movl", operands=[Immediate(0), dst], line=instr.line
                    )
                ]

        return self._translate_logical_op(instr, "xor", "XOR")

    def _translate_or(self, instr: Instruction) -> List[Statement]:
        """Translate OR instruction to MOV-only sequence."""
        return self._translate_logical_op(instr, "or", "OR")

    def _translate_and(self, instr: Instruction) -> List[Statement]:
        """Translate AND instruction to MOV-only sequence."""
        return self._translate_logical_op(instr, "and", "AND")

    def _translate_inc(
        self, instr: Instruction, update_flags: bool = True
    ) -> List[Statement]:
        """Translate INC dst to a MOV-only sequence."""
        if len(instr.operands) != 1:
            raise TranslatorError(
                f"INC requires 1 operand, got {len(instr.operands)}", instr.line
            )

        dst = self._normalize_operand(instr.operands[0])

        self._validate_32bit_operands(instr, [dst], "INC")

        statements: List[Statement] = []
        line = instr.line

        def mov(mnemonic: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnemonic, src_op, dst_op)

        scratch_r = self._scratch_mem("scratch_r")
        scratch_c = self._scratch_mem("scratch_c")

        # Determine which register is the destination (if any)
        dst_reg_name = dst.name if isinstance(dst, Register) else None

        save_map = self._emit_save_temps(statements, line)

        if isinstance(dst, Immediate):
            raise TranslatorError("INC destination must be register or memory", line)
        self._emit_store_operand_long_to_scratch(
            statements, dst, "scratch_b", line, save_map
        )

        statements.append(mov("movb", Immediate(1), scratch_c))

        for byte_idx in range(4):
            b_byte = self._scratch_mem("scratch_b", byte_idx)
            r_byte = self._scratch_mem("scratch_r", byte_idx)

            statements.append(mov("movzbl", scratch_c, Register("ecx")))

            # Add 0 or 1 based on carry using add_lut
            statements.append(mov("movzbl", b_byte, Register("eax")))

            # Get add_lut row pointer for original byte
            add_row_ptr = Memory(
                displacement="add_row_ptrs", index=Register("eax"), scale=4
            )
            statements.append(mov("movl", add_row_ptr, Register("eax")))

            # Lookup add_lut[byte][carry] = byte + carry (0 or 1)
            add_lookup = Memory(base=Register("eax"), index=Register("ecx"), scale=1)
            statements.append(mov("movzbl", add_lookup, Register("eax")))

            # Store result byte
            statements.append(mov("movb", Register("al"), r_byte))

            # Calculate carry for next byte (if not last byte)
            if byte_idx < 3:
                # First get is_zero of result
                is_zero = Memory(
                    displacement="is_zero_lut", index=Register("eax"), scale=1
                )
                statements.append(mov("movzbl", is_zero, Register("eax")))

                # Now AND with old carry (in ecx): use and_lut
                and_row_ptr = Memory(
                    displacement="and_row_ptrs", index=Register("eax"), scale=4
                )
                statements.append(mov("movl", and_row_ptr, Register("eax")))
                and_lookup = Memory(
                    base=Register("eax"), index=Register("ecx"), scale=1
                )
                statements.append(mov("movzbl", and_lookup, Register("eax")))

                # Store new carry
                statements.append(mov("movb", Register("al"), scratch_c))

        if update_flags:
            self._emit_update_zf(statements, line)

        if isinstance(dst, Register):
            statements.append(mov("movl", scratch_r, dst))
        elif isinstance(dst, Memory):
            self._emit_store_long_to_memory(
                statements,
                scratch_r,
                dst,
                save_map["eax"],
                save_map["ecx"],
                save_map["edx"],
                line,
            )

        self._emit_restore_temps(statements, line, dst_reg_name, save_map)

        return statements

    def _translate_dec(
        self, instr: Instruction, update_flags: bool = True
    ) -> List[Statement]:
        """Translate DEC dst to a MOV-only sequence."""
        if len(instr.operands) != 1:
            raise TranslatorError(
                f"DEC requires 1 operand, got {len(instr.operands)}", instr.line
            )

        dst = self._normalize_operand(instr.operands[0])

        self._validate_32bit_operands(instr, [dst], "DEC")

        statements: List[Statement] = []
        line = instr.line

        def mov(mnemonic: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnemonic, src_op, dst_op)

        scratch_r = self._scratch_mem("scratch_r")
        scratch_c = self._scratch_mem("scratch_c")

        # Determine which register is the destination (if any)
        dst_reg_name = dst.name if isinstance(dst, Register) else None

        save_map = self._emit_save_temps(statements, line)

        if isinstance(dst, Immediate):
            raise TranslatorError("DEC destination must be register or memory", line)
        self._emit_store_operand_long_to_scratch(
            statements, dst, "scratch_b", line, save_map
        )

        statements.append(mov("movb", Immediate(1), scratch_c))

        for byte_idx in range(4):
            b_byte = self._scratch_mem("scratch_b", byte_idx)
            r_byte = self._scratch_mem("scratch_r", byte_idx)

            # Load byte from scratch_b into eax (zero-extended)
            statements.append(mov("movzbl", b_byte, Register("eax")))

            statements.append(mov("movzbl", scratch_c, Register("ecx")))

            # Use sub_lut[byte][borrow] = byte - borrow
            sub_row_ptr = Memory(
                displacement="sub_row_ptrs", index=Register("eax"), scale=4
            )
            statements.append(mov("movl", sub_row_ptr, Register("eax")))
            sub_lookup = Memory(base=Register("eax"), index=Register("ecx"), scale=1)
            statements.append(mov("movzbl", sub_lookup, Register("eax")))

            # Store result byte
            statements.append(mov("movb", Register("al"), r_byte))

            # Calculate borrow for next byte (if not last byte)
            if byte_idx < 3:
                # Load original byte
                statements.append(mov("movzbl", b_byte, Register("eax")))

                # Get is_zero of original byte
                is_zero = Memory(
                    displacement="is_zero_lut", index=Register("eax"), scale=1
                )
                statements.append(mov("movzbl", is_zero, Register("eax")))

                # AND with old borrow (in ecx): use and_lut
                and_row_ptr = Memory(
                    displacement="and_row_ptrs", index=Register("eax"), scale=4
                )
                statements.append(mov("movl", and_row_ptr, Register("eax")))
                and_lookup = Memory(
                    base=Register("eax"), index=Register("ecx"), scale=1
                )
                statements.append(mov("movzbl", and_lookup, Register("eax")))

                # Store new borrow
                statements.append(mov("movb", Register("al"), scratch_c))

        # 4. Update ZF (scratch_cmp_eq) from result
        if update_flags:
            self._emit_update_zf(statements, instr.line)

        if isinstance(dst, Register):
            statements.append(mov("movl", scratch_r, dst))
        elif isinstance(dst, Memory):
            self._emit_store_long_to_memory(
                statements,
                scratch_r,
                dst,
                save_map["eax"],
                save_map["ecx"],
                save_map["edx"],
                line,
            )

        self._emit_restore_temps(statements, line, dst_reg_name, save_map)

        return statements

    def _translate_sub(
        self, instr: Instruction, update_flags: bool = True
    ) -> List[Statement]:
        """Translate SUB src, dst to a MOV-only sequence."""
        if len(instr.operands) != 2:
            raise TranslatorError(
                f"SUB requires 2 operands, got {len(instr.operands)}", instr.line
            )

        src, dst = (
            self._normalize_operand(instr.operands[0]),
            self._normalize_operand(instr.operands[1]),
        )

        self._validate_32bit_operands(instr, [src, dst], "SUB")

        statements: List[Statement] = []
        line = instr.line

        def mov(mnemonic: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnemonic, src_op, dst_op)

        scratch_r = self._scratch_mem("scratch_r")
        scratch_c = self._scratch_mem("scratch_c")

        # Determine which register is the destination (if any)
        dst_reg_name = dst.name if isinstance(dst, Register) else None

        save_map = self._emit_save_temps(statements, line)

        self._emit_store_operand_long_to_scratch(
            statements, src, "scratch_a", line, save_map
        )

        if isinstance(dst, Immediate):
            raise TranslatorError("SUB destination must be register or memory", line)
        self._emit_store_operand_long_to_scratch(
            statements, dst, "scratch_b", line, save_map
        )

        statements.append(mov("movb", Immediate(0), scratch_c))

        for byte_idx in range(4):
            a_byte = self._scratch_mem("scratch_a", byte_idx)
            b_byte = self._scratch_mem("scratch_b", byte_idx)
            r_byte = self._scratch_mem("scratch_r", byte_idx)

            statements.append(mov("movzbl", b_byte, Register("ecx")))

            # Get sub_lut row pointer: sub_row_ptrs(, %ecx, 4)
            sub_row_ptr = Memory(
                displacement="sub_row_ptrs", index=Register("ecx"), scale=4
            )
            statements.append(mov("movl", sub_row_ptr, Register("ecx")))

            # Load subtrahend byte (a) into edx (zero-extended)
            statements.append(mov("movzbl", a_byte, Register("edx")))

            # Lookup result: (%ecx, %edx, 1) = sub_lut[b][a] = b - a
            sub_lookup = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
            statements.append(mov("movzbl", sub_lookup, Register("eax")))

            # Handle borrow from previous byte
            if byte_idx > 0:
                # If borrow was set, we need to subtract 1 from the result
                # We do this by looking up sub_lut[result][borrow]
                # Since borrow is 0 or 1, this effectively subtracts 0 or 1

                # Load borrow into edx
                statements.append(mov("movzbl", scratch_c, Register("edx")))

                # Get row pointer for current result
                sub_row_ptr2 = Memory(
                    displacement="sub_row_ptrs", index=Register("eax"), scale=4
                )
                statements.append(mov("movl", sub_row_ptr2, Register("ecx")))

                # Lookup sub_lut[result][borrow] = result - borrow
                sub_borrow_lookup = Memory(
                    base=Register("ecx"), index=Register("edx"), scale=1
                )
                statements.append(mov("movzbl", sub_borrow_lookup, Register("eax")))

            # Store result byte
            statements.append(mov("movb", Register("al"), r_byte))

            # Calculate borrow for next byte (if not last byte)
            if byte_idx < 3:
                # Borrow = borrow_lut[b_byte][a_byte] (1 if b < a)
                # But we also need to handle borrow chain from subtracting previous borrow

                # Reload b_byte (minuend) and a_byte (subtrahend)
                statements.append(mov("movzbl", b_byte, Register("ecx")))
                borrow_row_ptr = Memory(
                    displacement="borrow_row_ptrs", index=Register("ecx"), scale=4
                )
                statements.append(mov("movl", borrow_row_ptr, Register("ecx")))
                statements.append(mov("movzbl", a_byte, Register("edx")))

                # Lookup borrow_lut[b][a] (1 if b < a)
                borrow_lookup = Memory(
                    base=Register("ecx"), index=Register("edx"), scale=1
                )
                statements.append(mov("movzbl", borrow_lookup, Register("eax")))

                if byte_idx > 0:
                    # Also check if subtracting the borrow caused another borrow
                    # borrow_out = borrow_lut[b][a] OR borrow_lut[b-a][old_borrow]
                    # Simplified: if old_borrow was 1 and (b-a)==0, then we borrow again

                    # Store borrow1 in scratch_t
                    scratch_t = Memory(displacement="scratch_t")
                    statements.append(mov("movb", Register("al"), scratch_t))

                    # Get sub_lut[b][a] (the difference before borrow)
                    statements.append(mov("movzbl", b_byte, Register("ecx")))
                    sub_row_ptr3 = Memory(
                        displacement="sub_row_ptrs", index=Register("ecx"), scale=4
                    )
                    statements.append(mov("movl", sub_row_ptr3, Register("ecx")))
                    statements.append(mov("movzbl", a_byte, Register("edx")))
                    diff_lookup = Memory(
                        base=Register("ecx"), index=Register("edx"), scale=1
                    )
                    statements.append(mov("movzbl", diff_lookup, Register("ecx")))

                    # Get borrow_lut[diff][old_borrow]
                    borrow_row_ptr2 = Memory(
                        displacement="borrow_row_ptrs", index=Register("ecx"), scale=4
                    )
                    statements.append(mov("movl", borrow_row_ptr2, Register("ecx")))
                    statements.append(mov("movzbl", scratch_c, Register("edx")))
                    borrow_from_borrow = Memory(
                        base=Register("ecx"), index=Register("edx"), scale=1
                    )
                    statements.append(
                        mov("movzbl", borrow_from_borrow, Register("eax"))
                    )

                    # OR with borrow1 (in scratch_t)
                    # Use: (borrow1 + borrow2) > 0 via is_not_zero_lut
                    statements.append(mov("movzbl", scratch_t, Register("edx")))
                    add_row_ptr = Memory(
                        displacement="add_row_ptrs", index=Register("eax"), scale=4
                    )
                    statements.append(mov("movl", add_row_ptr, Register("ecx")))
                    or_result = Memory(
                        base=Register("ecx"), index=Register("edx"), scale=1
                    )
                    statements.append(mov("movzbl", or_result, Register("eax")))

                    # If result > 0, borrow = 1, else borrow = 0
                    is_nz = Memory(
                        displacement="is_not_zero_lut", index=Register("eax"), scale=1
                    )
                    statements.append(mov("movzbl", is_nz, Register("eax")))

                # Store borrow for next byte
                statements.append(mov("movb", Register("al"), scratch_c))

        if update_flags:
            self._emit_update_zf(statements, line)
            self._emit_cmp_below_and_sign_lt(statements, line)

        if isinstance(dst, Register):
            statements.append(mov("movl", scratch_r, dst))
        elif isinstance(dst, Memory):
            self._emit_store_long_to_memory(
                statements,
                scratch_r,
                dst,
                save_map["eax"],
                save_map["ecx"],
                save_map["edx"],
                line,
            )

        self._emit_restore_temps(statements, line, dst_reg_name, save_map)

        return statements

    def _translate_cmp(self, instr: Instruction) -> List[Statement]:
        """Translate CMP src, dst to MOV-only sequence (flags only)."""
        if len(instr.operands) != 2:
            raise TranslatorError(
                f"CMP requires 2 operands, got {len(instr.operands)}", instr.line
            )

        src, dst = (
            self._normalize_operand(instr.operands[0]),
            self._normalize_operand(instr.operands[1]),
        )

        self._validate_32bit_operands(instr, [src, dst], "CMP")

        statements: List[Statement] = []
        line = instr.line

        def mov(mnemonic: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnemonic, src_op, dst_op)

        save_map = self._emit_save_temps(statements, line)

        self._emit_store_operand_long_to_scratch(
            statements, src, "scratch_a", line, save_map
        )

        if isinstance(dst, Immediate):
            raise TranslatorError("CMP destination must be register or memory", line)
        self._emit_store_operand_long_to_scratch(
            statements, dst, "scratch_b", line, save_map
        )

        # 3. Per-byte comparison: equality and unsigned less-than
        for byte_idx in range(4):
            a_byte = Memory(
                displacement=f"scratch_a+{byte_idx}" if byte_idx > 0 else "scratch_a"
            )
            b_byte = Memory(
                displacement=f"scratch_b+{byte_idx}" if byte_idx > 0 else "scratch_b"
            )

            # --- Equality: je_lut[dst_byte][src_byte] ---
            statements.append(mov("movzbl", b_byte, Register("ecx")))
            je_ptr = Memory(displacement="je_row_ptrs", index=Register("ecx"), scale=4)
            statements.append(mov("movl", je_ptr, Register("ecx")))
            statements.append(mov("movzbl", a_byte, Register("edx")))
            je_lookup = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
            statements.append(mov("movzbl", je_lookup, Register("eax")))

            # Store equality result
            if byte_idx == 0:
                # eq0 stored in scratch_t (temporary for final equality calc)
                eq_dest = Memory(displacement="scratch_t")
            else:
                eq_dest = Memory(displacement=f"scratch_eq{byte_idx}")
            statements.append(mov("movb", Register("al"), eq_dest))

            # --- Unsigned less-than: jb_lut[dst_byte][src_byte] ---
            statements.append(mov("movzbl", b_byte, Register("ecx")))
            jb_ptr = Memory(displacement="jb_row_ptrs", index=Register("ecx"), scale=4)
            statements.append(mov("movl", jb_ptr, Register("ecx")))
            statements.append(mov("movzbl", a_byte, Register("edx")))
            jb_lookup = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
            statements.append(mov("movzbl", jb_lookup, Register("eax")))

            lt_dest = Memory(displacement=f"scratch_lt{byte_idx}")
            statements.append(mov("movb", Register("al"), lt_dest))

        # 4. Signed less-than for byte 3 (MSB): jl_signed_lut[dst_byte3][src_byte3]
        b_byte3 = Memory(displacement="scratch_b+3")
        a_byte3 = Memory(displacement="scratch_a+3")
        statements.append(mov("movzbl", b_byte3, Register("ecx")))
        jl_ptr = Memory(
            displacement="jl_signed_row_ptrs", index=Register("ecx"), scale=4
        )
        statements.append(mov("movl", jl_ptr, Register("ecx")))
        statements.append(mov("movzbl", a_byte3, Register("edx")))
        jl_lookup = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
        statements.append(mov("movzbl", jl_lookup, Register("eax")))
        # Store signed lt3 in scratch_t+1
        statements.append(
            mov("movb", Register("al"), Memory(displacement="scratch_t+1"))
        )

        # 5. Compute final equality: eq3 AND eq2 AND eq1 AND eq0
        # Start: and_lut[eq3][eq2]
        statements.append(
            mov("movzbl", Memory(displacement="scratch_eq3"), Register("ecx"))
        )
        and_ptr = Memory(displacement="and_row_ptrs", index=Register("ecx"), scale=4)
        statements.append(mov("movl", and_ptr, Register("ecx")))
        statements.append(
            mov("movzbl", Memory(displacement="scratch_eq2"), Register("edx"))
        )
        and_lookup = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
        statements.append(mov("movzbl", and_lookup, Register("eax")))

        # AND with eq1
        and_ptr2 = Memory(displacement="and_row_ptrs", index=Register("eax"), scale=4)
        statements.append(mov("movl", and_ptr2, Register("ecx")))
        statements.append(
            mov("movzbl", Memory(displacement="scratch_eq1"), Register("edx"))
        )
        and_lookup2 = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
        statements.append(mov("movzbl", and_lookup2, Register("eax")))

        # AND with eq0 (in scratch_t)
        and_ptr3 = Memory(displacement="and_row_ptrs", index=Register("eax"), scale=4)
        statements.append(mov("movl", and_ptr3, Register("ecx")))
        statements.append(
            mov("movzbl", Memory(displacement="scratch_t"), Register("edx"))
        )
        and_lookup3 = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
        statements.append(mov("movzbl", and_lookup3, Register("eax")))

        # Store final equality result
        statements.append(
            mov("movb", Register("al"), Memory(displacement="scratch_cmp_eq"))
        )

        # 6. Compute final unsigned less-than (cascading from LSB to MSB):
        # result = lt3 OR (eq3 AND (lt2 OR (eq2 AND (lt1 OR (eq1 AND lt0)))))
        #
        # Step 1: temp = and_lut[eq1][lt0]
        statements.append(
            mov("movzbl", Memory(displacement="scratch_eq1"), Register("ecx"))
        )
        statements.append(
            mov(
                "movl",
                Memory(displacement="and_row_ptrs", index=Register("ecx"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_lt0"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # Step 2: temp = or_lut[temp][lt1]  (OR is symmetric, temp in eax)
        statements.append(
            mov(
                "movl",
                Memory(displacement="or_row_ptrs", index=Register("eax"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_lt1"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # Step 3: temp = and_lut[temp][eq2]
        statements.append(
            mov(
                "movl",
                Memory(displacement="and_row_ptrs", index=Register("eax"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_eq2"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # Step 4: temp = or_lut[temp][lt2]
        statements.append(
            mov(
                "movl",
                Memory(displacement="or_row_ptrs", index=Register("eax"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_lt2"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # Step 5: temp = and_lut[temp][eq3]
        statements.append(
            mov(
                "movl",
                Memory(displacement="and_row_ptrs", index=Register("eax"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_eq3"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # Save cascade result for signed computation (scratch_t+2)
        statements.append(
            mov("movb", Register("al"), Memory(displacement="scratch_t+2"))
        )

        # Step 6: result = or_lut[temp][lt3]  (unsigned)
        statements.append(
            mov(
                "movl",
                Memory(displacement="or_row_ptrs", index=Register("eax"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_lt3"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # Store final unsigned below result
        statements.append(
            mov("movb", Register("al"), Memory(displacement="scratch_cmp_below"))
        )

        # 7. Compute final signed less-than
        # Same cascade result (scratch_t+2), but OR with signed_lt3 (scratch_t+1)
        statements.append(
            mov("movzbl", Memory(displacement="scratch_t+2"), Register("eax"))
        )
        statements.append(
            mov(
                "movl",
                Memory(displacement="or_row_ptrs", index=Register("eax"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_t+1"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # Store final signed less-than result
        statements.append(
            mov("movb", Register("al"), Memory(displacement="scratch_cmp_sign_lt"))
        )

        self._emit_restore_temps(statements, line, None, save_map)

        return statements

    def _translate_test(self, instr: Instruction) -> List[Statement]:
        """Translate TEST src, dst to MOV-only sequence (flags only)."""
        if len(instr.operands) != 2:
            raise TranslatorError(
                f"TEST requires 2 operands, got {len(instr.operands)}", instr.line
            )

        src, dst = (
            self._normalize_operand(instr.operands[0]),
            self._normalize_operand(instr.operands[1]),
        )

        self._validate_32bit_operands(instr, [src, dst], "TEST")

        statements: List[Statement] = []
        line = instr.line

        def mov(mnemonic: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnemonic, src_op, dst_op)

        scratch_r = self._scratch_mem("scratch_r")
        save_map = self._emit_save_temps(statements, line)

        self._emit_store_operand_long_to_scratch(
            statements, src, "scratch_a", line, save_map
        )

        if isinstance(dst, Immediate):
            raise TranslatorError("TEST destination must be register or memory", line)
        self._emit_store_operand_long_to_scratch(
            statements, dst, "scratch_b", line, save_map
        )

        for byte_idx in range(4):
            a_byte = self._scratch_mem("scratch_a", byte_idx)
            b_byte = self._scratch_mem("scratch_b", byte_idx)
            r_byte = self._scratch_mem("scratch_r", byte_idx)

            statements.append(mov("movzbl", a_byte, Register("ecx")))
            and_ptr = Memory(
                displacement="and_row_ptrs", index=Register("ecx"), scale=4
            )
            statements.append(mov("movl", and_ptr, Register("ecx")))
            statements.append(mov("movzbl", b_byte, Register("edx")))
            and_lookup = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
            statements.append(mov("movzbl", and_lookup, Register("eax")))
            statements.append(mov("movb", Register("al"), r_byte))

        self._emit_update_zf(statements, line)
        self._emit_restore_temps(statements, line, None, save_map)
        return statements

    def _translate_jcc_signed(self, instr: Instruction) -> List[Statement]:
        """Translate signed Jcc using scratch_cmp_* dispatch.

        Supported mnemonics: jl/jle/jg/jge and their common aliases.
        """
        if len(instr.operands) != 1:
            raise TranslatorError(
                f"Conditional jump requires 1 operand (label), got {len(instr.operands)}",
                instr.line,
            )

        target = instr.operands[0]
        if not isinstance(target, LabelRef):
            raise TranslatorError(
                f"Conditional jump operand must be a label", instr.line
            )

        mnemonic = instr.mnemonic.lower()
        counter = self._jcc_counter
        self._jcc_counter += 1

        table_label = f"_jcc_table_{counter}"
        fallthrough_label = f"_jcc_fallthrough_{counter}"
        target_label = target.name

        statements: List[Statement] = []
        line = instr.line

        def mov(mnem: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnem, src_op, dst_op)

        save_map = self._emit_save_temps(statements, line)

        # 1. Emit dispatch table in .data section
        statements.append(Directive(name="section", args=[".data"], line=instr.line))
        statements.append(Label(name=table_label, line=instr.line))
        statements.append(
            Directive(name="long", args=[fallthrough_label], line=instr.line)
        )
        statements.append(Directive(name="long", args=[target_label], line=instr.line))
        statements.append(Directive(name="section", args=[".text"], line=instr.line))

        # 2. Save temp registers

        # 3. Compute condition index (0 = not taken, 1 = taken)
        if mnemonic in ("jl", "jnge"):
            # index = scratch_cmp_sign_lt
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="scratch_cmp_sign_lt"),
                    Register("eax"),
                )
            )

        elif mnemonic in ("jge", "jnl"):
            # index = NOT(scratch_cmp_sign_lt) = is_zero_lut[sign_lt]
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="scratch_cmp_sign_lt"),
                    Register("eax"),
                )
            )
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="is_zero_lut", index=Register("eax"), scale=1),
                    Register("eax"),
                )
            )

        elif mnemonic in ("jle", "jng"):
            # index = OR(scratch_cmp_sign_lt, scratch_cmp_eq)
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="scratch_cmp_sign_lt"),
                    Register("eax"),
                )
            )
            statements.append(
                mov(
                    "movl",
                    Memory(displacement="or_row_ptrs", index=Register("eax"), scale=4),
                    Register("eax"),
                )
            )
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="scratch_cmp_eq"),
                    Register("ecx"),
                )
            )
            statements.append(
                mov(
                    "movzbl",
                    Memory(base=Register("eax"), index=Register("ecx"), scale=1),
                    Register("eax"),
                )
            )

        elif mnemonic in ("jg", "jnle"):
            # index = NOT(OR(scratch_cmp_sign_lt, scratch_cmp_eq))
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="scratch_cmp_sign_lt"),
                    Register("eax"),
                )
            )
            statements.append(
                mov(
                    "movl",
                    Memory(displacement="or_row_ptrs", index=Register("eax"), scale=4),
                    Register("eax"),
                )
            )
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="scratch_cmp_eq"),
                    Register("ecx"),
                )
            )
            statements.append(
                mov(
                    "movzbl",
                    Memory(base=Register("eax"), index=Register("ecx"), scale=1),
                    Register("eax"),
                )
            )
            # Invert: NOT JLE = JG
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="is_zero_lut", index=Register("eax"), scale=1),
                    Register("eax"),
                )
            )

        # 4. Load target address from dispatch table
        dispatch_lookup = Memory(
            displacement=table_label, index=Register("eax"), scale=4
        )
        statements.append(mov("movl", dispatch_lookup, Register("eax")))

        # 5. Store target address to scratch_jcc_target
        statements.append(
            mov("movl", Register("eax"), Memory(displacement="scratch_jcc_target"))
        )

        # 6. Restore temp registers
        self._emit_restore_temps(statements, line, None, save_map)

        # 7. Indirect jump through scratch_jcc_target
        statements.append(
            Instruction(
                mnemonic="jmp",
                operands=[LabelRef(name="*scratch_jcc_target")],
                line=line,
            )
        )

        # 8. Fallthrough label (not-taken path continues here)
        statements.append(Label(name=fallthrough_label, line=line))

        return statements

    def _translate_jcc_unsigned(self, instr: Instruction) -> List[Statement]:
        """Translate unsigned Jcc using scratch_cmp_* dispatch.

        Supported mnemonics: jb/jbe/ja/jae and their common aliases.
        """
        if len(instr.operands) != 1:
            raise TranslatorError(
                f"Conditional jump requires 1 operand (label), got {len(instr.operands)}",
                instr.line,
            )

        target = instr.operands[0]
        if not isinstance(target, LabelRef):
            raise TranslatorError(
                f"Conditional jump operand must be a label", instr.line
            )

        mnemonic = instr.mnemonic.lower()
        counter = self._jcc_counter
        self._jcc_counter += 1

        table_label = f"_jcc_table_{counter}"
        fallthrough_label = f"_jcc_fallthrough_{counter}"
        target_label = target.name

        statements: List[Statement] = []
        line = instr.line

        def mov(mnem: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnem, src_op, dst_op)

        save_map = self._emit_save_temps(statements, line)

        # 1. Emit dispatch table in .data section
        statements.append(Directive(name="section", args=[".data"], line=instr.line))
        statements.append(Label(name=table_label, line=instr.line))
        statements.append(
            Directive(name="long", args=[fallthrough_label], line=instr.line)
        )
        statements.append(Directive(name="long", args=[target_label], line=instr.line))
        statements.append(Directive(name="section", args=[".text"], line=instr.line))

        # 2. Save temp registers

        # 3. Compute condition index (0 = not taken, 1 = taken)
        if mnemonic in ("jb", "jc", "jnae"):
            # index = scratch_cmp_below
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="scratch_cmp_below"),
                    Register("eax"),
                )
            )

        elif mnemonic in ("jae", "jnb", "jnc"):
            # index = NOT(scratch_cmp_below) = is_zero_lut[below]
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="scratch_cmp_below"),
                    Register("eax"),
                )
            )
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="is_zero_lut", index=Register("eax"), scale=1),
                    Register("eax"),
                )
            )

        elif mnemonic in ("jbe", "jna"):
            # index = OR(scratch_cmp_below, scratch_cmp_eq)
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="scratch_cmp_below"),
                    Register("eax"),
                )
            )
            statements.append(
                mov(
                    "movl",
                    Memory(displacement="or_row_ptrs", index=Register("eax"), scale=4),
                    Register("eax"),
                )
            )
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="scratch_cmp_eq"),
                    Register("ecx"),
                )
            )
            statements.append(
                mov(
                    "movzbl",
                    Memory(base=Register("eax"), index=Register("ecx"), scale=1),
                    Register("eax"),
                )
            )

        elif mnemonic in ("ja", "jnbe"):
            # index = NOT(OR(scratch_cmp_below, scratch_cmp_eq))
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="scratch_cmp_below"),
                    Register("eax"),
                )
            )
            statements.append(
                mov(
                    "movl",
                    Memory(displacement="or_row_ptrs", index=Register("eax"), scale=4),
                    Register("eax"),
                )
            )
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="scratch_cmp_eq"),
                    Register("ecx"),
                )
            )
            statements.append(
                mov(
                    "movzbl",
                    Memory(base=Register("eax"), index=Register("ecx"), scale=1),
                    Register("eax"),
                )
            )
            # Invert: NOT JBE = JA
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="is_zero_lut", index=Register("eax"), scale=1),
                    Register("eax"),
                )
            )

        # 4. Load target address from dispatch table
        dispatch_lookup = Memory(
            displacement=table_label, index=Register("eax"), scale=4
        )
        statements.append(mov("movl", dispatch_lookup, Register("eax")))

        # 5. Store target address to scratch_jcc_target
        statements.append(
            mov("movl", Register("eax"), Memory(displacement="scratch_jcc_target"))
        )

        # 6. Restore temp registers
        self._emit_restore_temps(statements, line, None, save_map)

        # 7. Indirect jump through scratch_jcc_target
        statements.append(
            Instruction(
                mnemonic="jmp",
                operands=[LabelRef(name="*scratch_jcc_target")],
                line=line,
            )
        )

        # 8. Fallthrough label (not-taken path continues here)
        statements.append(Label(name=fallthrough_label, line=line))

        return statements

    def _translate_loop(self, instr: Instruction) -> List[Statement]:
        """Translate LOOP (dec ecx; jnz target) to MOV-only sequence."""
        if len(instr.operands) != 1:
            raise TranslatorError(
                f"LOOP requires 1 operand (label), got {len(instr.operands)}",
                instr.line,
            )

        target = instr.operands[0]
        if not isinstance(target, LabelRef):
            raise TranslatorError(f"LOOP operand must be a label", instr.line)

        counter = self._jcc_counter
        self._jcc_counter += 1

        table_label = f"_jcc_table_{counter}"
        fallthrough_label = f"_jcc_fallthrough_{counter}"
        target_label = target.name

        statements: List[Statement] = []
        line = instr.line

        def mov(mnem: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnem, src_op, dst_op)

        scratch_r = self._scratch_mem("scratch_r")
        scratch_c = self._scratch_mem("scratch_c")

        save_map = self._emit_save_temps(statements, line)

        # 1. Emit dispatch table in .data section
        statements.append(Directive(name="section", args=[".data"], line=line))
        statements.append(Label(name=table_label, line=line))
        statements.append(Directive(name="long", args=[fallthrough_label], line=line))
        statements.append(Directive(name="long", args=[target_label], line=line))
        statements.append(Directive(name="section", args=[".text"], line=line))

        # 2. Load ECX into scratch_b (avoid mem-to-mem)
        self._emit_store_operand_long_to_scratch(
            statements, Register("ecx"), "scratch_b", line, save_map
        )

        # 3. Perform 32-bit DEC on scratch_b -> scratch_r
        statements.append(mov("movb", Immediate(1), scratch_c))

        for byte_idx in range(4):
            b_byte = self._scratch_mem("scratch_b", byte_idx)
            r_byte = self._scratch_mem("scratch_r", byte_idx)

            # Load byte from scratch_b into eax (zero-extended)
            statements.append(mov("movzbl", b_byte, Register("eax")))

            # Load borrow into ecx
            statements.append(mov("movzbl", scratch_c, Register("ecx")))

            # Use sub_lut[byte][borrow] = byte - borrow
            sub_row_ptr = Memory(
                displacement="sub_row_ptrs", index=Register("eax"), scale=4
            )
            statements.append(mov("movl", sub_row_ptr, Register("eax")))
            sub_lookup = Memory(base=Register("eax"), index=Register("ecx"), scale=1)
            statements.append(mov("movzbl", sub_lookup, Register("eax")))

            # Store result byte
            statements.append(mov("movb", Register("al"), r_byte))

            # Calculate borrow for next byte (if not last byte)
            if byte_idx < 3:
                # Borrow occurs if: original_byte == 0 AND current_borrow == 1
                statements.append(mov("movzbl", b_byte, Register("eax")))
                is_zero = Memory(
                    displacement="is_zero_lut", index=Register("eax"), scale=1
                )
                statements.append(mov("movzbl", is_zero, Register("eax")))
                and_row_ptr = Memory(
                    displacement="and_row_ptrs", index=Register("eax"), scale=4
                )
                statements.append(mov("movl", and_row_ptr, Register("eax")))
                and_lookup = Memory(
                    base=Register("eax"), index=Register("ecx"), scale=1
                )
                statements.append(mov("movzbl", and_lookup, Register("eax")))
                statements.append(mov("movb", Register("al"), scratch_c))

        # 4. Compute dispatch index: 1 if ecx!=0 else 0
        self._emit_update_zf(statements, line)
        statements.append(
            mov("movzbl", Memory(displacement="scratch_cmp_eq"), Register("eax"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(displacement="is_zero_lut", index=Register("eax"), scale=1),
                Register("eax"),
            )
        )

        # 5. Load target address from dispatch table
        dispatch_lookup = Memory(
            displacement=table_label, index=Register("eax"), scale=4
        )
        statements.append(mov("movl", dispatch_lookup, Register("eax")))

        # 6. Store target address to scratch_jcc_target
        statements.append(
            mov("movl", Register("eax"), Memory(displacement="scratch_jcc_target"))
        )

        # 7. Store decremented value back to ECX
        statements.append(mov("movl", scratch_r, Register("ecx")))

        # 8. Restore temps except ecx (ecx holds the decremented value)
        self._emit_restore_temps_except(statements, line, {"ecx"}, save_map)

        # 9. Indirect jump through scratch_jcc_target
        statements.append(
            Instruction(
                mnemonic="jmp",
                operands=[LabelRef(name="*scratch_jcc_target")],
                line=line,
            )
        )

        # 10. Fallthrough label (ecx == 0, exit loop)
        statements.append(Label(name=fallthrough_label, line=line))

        return statements

    def _emit_accum_add_byte(
        self,
        statements: List[Statement],
        pos: int,
        val_mem: Memory,
        max_pos: int,
        line: int,
    ) -> None:
        """Emit MOV-only instructions to add byte val_mem to mul_accum[pos].

        Includes carry propagation from pos+1 through max_pos.
        Uses scratch_c for running carry, scratch_t+2 as carry temporary.
        """

        def mov(mnemonic: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return Instruction(mnemonic=mnemonic, operands=[src_op, dst_op], line=line)

        def accum_byte(k: int) -> Memory:
            return Memory(displacement=f"mul_accum+{k}" if k > 0 else "mul_accum")

        scratch_c = Memory(displacement="scratch_c")
        carry_temp = Memory(displacement="scratch_t+2")

        # --- Compute carry BEFORE modifying accum[pos] ---
        statements.append(mov("movzbl", accum_byte(pos), Register("ecx")))
        carry_row = Memory(
            displacement="carry_row_ptrs", index=Register("ecx"), scale=4
        )
        statements.append(mov("movl", carry_row, Register("ecx")))
        statements.append(mov("movzbl", val_mem, Register("edx")))
        carry_lookup = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
        statements.append(mov("movzbl", carry_lookup, Register("eax")))
        statements.append(mov("movb", Register("al"), scratch_c))

        # --- Compute sum and store ---
        statements.append(mov("movzbl", accum_byte(pos), Register("ecx")))
        add_row = Memory(displacement="add_row_ptrs", index=Register("ecx"), scale=4)
        statements.append(mov("movl", add_row, Register("ecx")))
        statements.append(mov("movzbl", val_mem, Register("edx")))
        add_lookup = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
        statements.append(mov("movzbl", add_lookup, Register("eax")))
        statements.append(mov("movb", Register("al"), accum_byte(pos)))

        # --- Propagate carry through pos+1 .. max_pos ---
        for k in range(pos + 1, max_pos + 1):
            # carry_new = carry_lut[accum[k]][carry_old]
            statements.append(mov("movzbl", accum_byte(k), Register("ecx")))
            carry_row_k = Memory(
                displacement="carry_row_ptrs", index=Register("ecx"), scale=4
            )
            statements.append(mov("movl", carry_row_k, Register("ecx")))
            statements.append(mov("movzbl", scratch_c, Register("edx")))
            carry_lookup_k = Memory(
                base=Register("ecx"), index=Register("edx"), scale=1
            )
            statements.append(mov("movzbl", carry_lookup_k, Register("eax")))
            statements.append(mov("movb", Register("al"), carry_temp))

            # accum[k] = add_lut[accum[k]][carry_old]
            statements.append(mov("movzbl", accum_byte(k), Register("ecx")))
            add_row_k = Memory(
                displacement="add_row_ptrs", index=Register("ecx"), scale=4
            )
            statements.append(mov("movl", add_row_k, Register("ecx")))
            statements.append(mov("movzbl", scratch_c, Register("edx")))
            add_lookup_k = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
            statements.append(mov("movzbl", add_lookup_k, Register("eax")))
            statements.append(mov("movb", Register("al"), accum_byte(k)))

            # Update carry for next iteration
            statements.append(mov("movzbl", carry_temp, Register("eax")))
            statements.append(mov("movb", Register("al"), scratch_c))

    def _translate_mul(self, instr: Instruction) -> List[Statement]:
        """Translate unsigned MUL op (EDX:EAX = EAX * op) to MOV-only sequence."""
        if len(instr.operands) != 1:
            raise TranslatorError(
                f"MUL requires 1 operand, got {len(instr.operands)}", instr.line
            )

        operand = self._normalize_operand(instr.operands[0])

        if isinstance(operand, Immediate):
            raise TranslatorError("MUL does not accept immediate operand", instr.line)

        self._validate_32bit_operands(instr, [operand], "MUL")

        statements: List[Statement] = []
        line = instr.line

        def mov(mnemonic: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnemonic, src_op, dst_op)

        mul_op_a = Memory(displacement="mul_op_a")
        mul_op_b = Memory(displacement="mul_op_b")
        mul_accum = Memory(displacement="mul_accum")
        mul_accum_hi = Memory(displacement="mul_accum+4")

        save_map = self._emit_save_temps(statements, line)

        # 1. Store EAX (multiplicand) to mul_op_a
        statements.append(mov("movl", save_map["eax"], Register("eax")))
        statements.append(mov("movl", Register("eax"), mul_op_a))

        # 2. Store operand (multiplier) to mul_op_b
        self._emit_store_operand_long_to_memory(
            statements, operand, mul_op_b, line, save_map
        )

        # 3. Clear 64-bit accumulator
        statements.append(mov("movl", Immediate(0), mul_accum))
        statements.append(mov("movl", Immediate(0), mul_accum_hi))

        # 4. Process 16 partial products (schoolbook multiplication)
        for i in range(4):
            for j in range(4):
                pos = i + j  # Output byte position (0-6)

                # Memory references for operand bytes
                a_byte = Memory(displacement=f"mul_op_a+{i}" if i > 0 else "mul_op_a")
                b_byte = Memory(displacement=f"mul_op_b+{j}" if j > 0 else "mul_op_b")

                lo_temp = Memory(displacement="scratch_t")
                hi_temp = Memory(displacement="scratch_t+1")

                # --- Compute lo byte: mul8_lo_lut[Ai][Bj] ---
                statements.append(mov("movzbl", a_byte, Register("ecx")))
                lo_row_ptr = Memory(
                    displacement="mul8_lo_row_ptrs",
                    index=Register("ecx"),
                    scale=4,
                )
                statements.append(mov("movl", lo_row_ptr, Register("ecx")))
                statements.append(mov("movzbl", b_byte, Register("edx")))
                lo_lookup = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
                statements.append(mov("movzbl", lo_lookup, Register("eax")))
                statements.append(mov("movb", Register("al"), lo_temp))

                # --- Compute hi byte: mul8_hi_lut[Ai][Bj] ---
                statements.append(mov("movzbl", a_byte, Register("ecx")))
                hi_row_ptr = Memory(
                    displacement="mul8_hi_row_ptrs",
                    index=Register("ecx"),
                    scale=4,
                )
                statements.append(mov("movl", hi_row_ptr, Register("ecx")))
                statements.append(mov("movzbl", b_byte, Register("edx")))
                hi_lookup = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
                statements.append(mov("movzbl", hi_lookup, Register("eax")))
                statements.append(mov("movb", Register("al"), hi_temp))

                # --- Add lo byte to accumulator at position pos ---
                self._emit_accum_add_byte(statements, pos, lo_temp, 7, line)

                # --- Add hi byte to accumulator at position pos+1 ---
                self._emit_accum_add_byte(statements, pos + 1, hi_temp, 7, line)

        # 5. Load result: EAX = accum low 32 bits, EDX = accum high 32 bits
        statements.append(mov("movl", mul_accum, Register("eax")))
        statements.append(mov("movl", mul_accum_hi, Register("edx")))

        # 6. Restore ecx only (eax and edx contain the MUL result)
        self._emit_restore_temps_except(statements, line, {"eax", "edx"}, save_map)

        return statements

    def _emit_cmp_below_and_sign_lt(
        self, statements: List[Statement], line: int
    ) -> None:
        """Emit inline comparison: scratch_cmp_below and scratch_cmp_sign_lt.

        Computes unsigned below and signed less-than from scratch_a/scratch_b.
        Prereq: scratch_a = src value, scratch_b = dst value
        Result: scratch_cmp_below = 1 if dst < src (unsigned)
                scratch_cmp_sign_lt = 1 if dst < src (signed)
        """

        def mov(mnem: str, s: Operand, d: Operand) -> Instruction:
            return Instruction(mnemonic=mnem, operands=[s, d], line=line)

        # Per-byte comparisons
        for byte_idx in range(4):
            a_byte = Memory(
                displacement=f"scratch_a+{byte_idx}" if byte_idx > 0 else "scratch_a"
            )
            b_byte = Memory(
                displacement=f"scratch_b+{byte_idx}" if byte_idx > 0 else "scratch_b"
            )

            # Equality for bytes 1-3 (needed for below cascade)
            if byte_idx > 0:
                statements.append(mov("movzbl", b_byte, Register("ecx")))
                je_ptr = Memory(
                    displacement="je_row_ptrs", index=Register("ecx"), scale=4
                )
                statements.append(mov("movl", je_ptr, Register("ecx")))
                statements.append(mov("movzbl", a_byte, Register("edx")))
                je_lookup = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
                statements.append(mov("movzbl", je_lookup, Register("eax")))
                statements.append(
                    mov(
                        "movb",
                        Register("al"),
                        Memory(displacement=f"scratch_eq{byte_idx}"),
                    )
                )

            # Unsigned less-than for all bytes
            statements.append(mov("movzbl", b_byte, Register("ecx")))
            jb_ptr = Memory(displacement="jb_row_ptrs", index=Register("ecx"), scale=4)
            statements.append(mov("movl", jb_ptr, Register("ecx")))
            statements.append(mov("movzbl", a_byte, Register("edx")))
            jb_lookup = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
            statements.append(mov("movzbl", jb_lookup, Register("eax")))
            statements.append(
                mov(
                    "movb",
                    Register("al"),
                    Memory(displacement=f"scratch_lt{byte_idx}"),
                )
            )

        # Signed less-than for byte 3 (MSB): jl_signed_lut[dst_byte3][src_byte3]
        b_byte3 = Memory(displacement="scratch_b+3")
        a_byte3 = Memory(displacement="scratch_a+3")
        statements.append(mov("movzbl", b_byte3, Register("ecx")))
        jl_ptr = Memory(
            displacement="jl_signed_row_ptrs", index=Register("ecx"), scale=4
        )
        statements.append(mov("movl", jl_ptr, Register("ecx")))
        statements.append(mov("movzbl", a_byte3, Register("edx")))
        jl_lookup = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
        statements.append(mov("movzbl", jl_lookup, Register("eax")))
        # Store signed lt3 in scratch_t+1
        statements.append(
            mov("movb", Register("al"), Memory(displacement="scratch_t+1"))
        )

        # Cascade: result = lt3 OR (eq3 AND (lt2 OR (eq2 AND (lt1 OR (eq1 AND lt0)))))

        # Step 1: temp = and_lut[eq1][lt0]
        statements.append(
            mov("movzbl", Memory(displacement="scratch_eq1"), Register("ecx"))
        )
        statements.append(
            mov(
                "movl",
                Memory(displacement="and_row_ptrs", index=Register("ecx"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_lt0"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # Step 2: temp = or_lut[temp][lt1]
        statements.append(
            mov(
                "movl",
                Memory(displacement="or_row_ptrs", index=Register("eax"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_lt1"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # Step 3: temp = and_lut[temp][eq2]
        statements.append(
            mov(
                "movl",
                Memory(displacement="and_row_ptrs", index=Register("eax"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_eq2"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # Step 4: temp = or_lut[temp][lt2]
        statements.append(
            mov(
                "movl",
                Memory(displacement="or_row_ptrs", index=Register("eax"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_lt2"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # Step 5: temp = and_lut[temp][eq3]
        statements.append(
            mov(
                "movl",
                Memory(displacement="and_row_ptrs", index=Register("eax"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_eq3"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # Save cascade intermediate for signed computation
        statements.append(
            mov("movb", Register("al"), Memory(displacement="scratch_t+2"))
        )

        # Step 6: result = or_lut[temp][lt3] (unsigned below)
        statements.append(
            mov(
                "movl",
                Memory(displacement="or_row_ptrs", index=Register("eax"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_lt3"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # Store final unsigned below result
        statements.append(
            mov("movb", Register("al"), Memory(displacement="scratch_cmp_below"))
        )

        # Compute signed less-than: or_lut[cascade_intermediate][signed_lt3]
        statements.append(
            mov("movzbl", Memory(displacement="scratch_t+2"), Register("eax"))
        )
        statements.append(
            mov(
                "movl",
                Memory(displacement="or_row_ptrs", index=Register("eax"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="scratch_t+1"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )

        # Store final signed less-than result
        statements.append(
            mov("movb", Register("al"), Memory(displacement="scratch_cmp_sign_lt"))
        )

    def _emit_sub_bytes(self, statements: List[Statement], line: int) -> None:
        """Emit inline SUB: scratch_r = scratch_b - scratch_a (4 bytes).

        Prereq: scratch_a = subtrahend, scratch_b = minuend
        Result: scratch_r = minuend - subtrahend
        Uses: scratch_c (borrow), scratch_t, eax, ecx, edx
        """

        def mov(mnem: str, s: Operand, d: Operand) -> Instruction:
            return Instruction(mnemonic=mnem, operands=[s, d], line=line)

        scratch_c = Memory(displacement="scratch_c")

        # Clear borrow
        statements.append(mov("movb", Immediate(0), scratch_c))

        for byte_idx in range(4):
            a_byte = Memory(
                displacement=f"scratch_a+{byte_idx}" if byte_idx > 0 else "scratch_a"
            )
            b_byte = Memory(
                displacement=f"scratch_b+{byte_idx}" if byte_idx > 0 else "scratch_b"
            )
            r_byte = Memory(
                displacement=f"scratch_r+{byte_idx}" if byte_idx > 0 else "scratch_r"
            )

            # sub_lut[minuend][subtrahend]
            statements.append(mov("movzbl", b_byte, Register("ecx")))
            sub_row_ptr = Memory(
                displacement="sub_row_ptrs", index=Register("ecx"), scale=4
            )
            statements.append(mov("movl", sub_row_ptr, Register("ecx")))
            statements.append(mov("movzbl", a_byte, Register("edx")))
            sub_lookup = Memory(base=Register("ecx"), index=Register("edx"), scale=1)
            statements.append(mov("movzbl", sub_lookup, Register("eax")))

            # Handle borrow from previous byte
            if byte_idx > 0:
                statements.append(mov("movzbl", scratch_c, Register("edx")))
                sub_row_ptr2 = Memory(
                    displacement="sub_row_ptrs",
                    index=Register("eax"),
                    scale=4,
                )
                statements.append(mov("movl", sub_row_ptr2, Register("ecx")))
                sub_borrow_lookup = Memory(
                    base=Register("ecx"), index=Register("edx"), scale=1
                )
                statements.append(mov("movzbl", sub_borrow_lookup, Register("eax")))

            # Store result byte
            statements.append(mov("movb", Register("al"), r_byte))

            # Borrow for next byte
            if byte_idx < 3:
                statements.append(mov("movzbl", b_byte, Register("ecx")))
                borrow_row_ptr = Memory(
                    displacement="borrow_row_ptrs",
                    index=Register("ecx"),
                    scale=4,
                )
                statements.append(mov("movl", borrow_row_ptr, Register("ecx")))
                statements.append(mov("movzbl", a_byte, Register("edx")))
                borrow_lookup = Memory(
                    base=Register("ecx"), index=Register("edx"), scale=1
                )
                statements.append(mov("movzbl", borrow_lookup, Register("eax")))

                if byte_idx > 0:
                    scratch_t = Memory(displacement="scratch_t")
                    statements.append(mov("movb", Register("al"), scratch_t))

                    # Get diff before borrow: sub_lut[b][a]
                    statements.append(mov("movzbl", b_byte, Register("ecx")))
                    sub_row_ptr3 = Memory(
                        displacement="sub_row_ptrs",
                        index=Register("ecx"),
                        scale=4,
                    )
                    statements.append(mov("movl", sub_row_ptr3, Register("ecx")))
                    statements.append(mov("movzbl", a_byte, Register("edx")))
                    diff_lookup = Memory(
                        base=Register("ecx"), index=Register("edx"), scale=1
                    )
                    statements.append(mov("movzbl", diff_lookup, Register("ecx")))

                    # borrow_lut[diff][old_borrow]
                    borrow_row_ptr2 = Memory(
                        displacement="borrow_row_ptrs",
                        index=Register("ecx"),
                        scale=4,
                    )
                    statements.append(mov("movl", borrow_row_ptr2, Register("ecx")))
                    statements.append(mov("movzbl", scratch_c, Register("edx")))
                    borrow_from_borrow = Memory(
                        base=Register("ecx"), index=Register("edx"), scale=1
                    )
                    statements.append(
                        mov("movzbl", borrow_from_borrow, Register("eax"))
                    )

                    # OR: (borrow1 + borrow2) > 0
                    statements.append(mov("movzbl", scratch_t, Register("edx")))
                    add_row_ptr = Memory(
                        displacement="add_row_ptrs",
                        index=Register("eax"),
                        scale=4,
                    )
                    statements.append(mov("movl", add_row_ptr, Register("ecx")))
                    or_result = Memory(
                        base=Register("ecx"), index=Register("edx"), scale=1
                    )
                    statements.append(mov("movzbl", or_result, Register("eax")))

                    is_nz = Memory(
                        displacement="is_not_zero_lut",
                        index=Register("eax"),
                        scale=1,
                    )
                    statements.append(mov("movzbl", is_nz, Register("eax")))

                statements.append(mov("movb", Register("al"), scratch_c))

    def _emit_inc_bytes(self, statements: List[Statement], line: int) -> None:
        """Emit inline INC: scratch_r = scratch_b + 1 (4 bytes).

        Prereq: scratch_b = value to increment
        Result: scratch_r = value + 1
        Uses: scratch_c (carry), eax, ecx
        """

        def mov(mnem: str, s: Operand, d: Operand) -> Instruction:
            return Instruction(mnemonic=mnem, operands=[s, d], line=line)

        scratch_c = Memory(displacement="scratch_c")

        # Set carry to 1 (add 1)
        statements.append(mov("movb", Immediate(1), scratch_c))

        for byte_idx in range(4):
            b_byte = Memory(
                displacement=f"scratch_b+{byte_idx}" if byte_idx > 0 else "scratch_b"
            )
            r_byte = Memory(
                displacement=f"scratch_r+{byte_idx}" if byte_idx > 0 else "scratch_r"
            )

            # Load byte
            statements.append(mov("movzbl", b_byte, Register("eax")))

            # Load carry
            statements.append(mov("movzbl", scratch_c, Register("ecx")))

            # add_lut[byte][carry]
            add_row_ptr = Memory(
                displacement="add_row_ptrs", index=Register("eax"), scale=4
            )
            statements.append(mov("movl", add_row_ptr, Register("eax")))
            add_lookup = Memory(base=Register("eax"), index=Register("ecx"), scale=1)
            statements.append(mov("movzbl", add_lookup, Register("eax")))

            # Store result byte
            statements.append(mov("movb", Register("al"), r_byte))

            # Carry for next byte
            if byte_idx < 3:
                is_zero = Memory(
                    displacement="is_zero_lut",
                    index=Register("eax"),
                    scale=1,
                )
                statements.append(mov("movzbl", is_zero, Register("eax")))

                and_row_ptr = Memory(
                    displacement="and_row_ptrs",
                    index=Register("eax"),
                    scale=4,
                )
                statements.append(mov("movl", and_row_ptr, Register("eax")))
                and_lookup = Memory(
                    base=Register("eax"), index=Register("ecx"), scale=1
                )
                statements.append(mov("movzbl", and_lookup, Register("eax")))

                statements.append(mov("movb", Register("al"), scratch_c))

    def _translate_div(self, instr: Instruction) -> List[Statement]:
        """Translate unsigned DIV r/m32 using MOV-only code.

        Implements full x86-32 semantics for the 64-bit dividend EDX:EAX.

        - Inputs:  unsigned dividend in EDX:EAX, unsigned divisor in operand
        - Outputs: quotient in EAX, remainder in EDX
        - Traps:   divisor == 0 or quotient overflow (EDX >= divisor) -> INT $0

        Notes:
        - We intentionally model the #DE exception via `int $0`.
        - We do not attempt to preserve EAX/EDX (they are outputs). ECX is
          preserved (restored) like other translations.
        """
        if len(instr.operands) != 1:
            raise TranslatorError(
                f"DIV requires 1 operand, got {len(instr.operands)}", instr.line
            )

        operand = self._normalize_operand(instr.operands[0])

        if isinstance(operand, Immediate):
            raise TranslatorError("DIV does not accept immediate operand", instr.line)

        self._validate_32bit_operands(instr, [operand], "DIV")

        # Generate unique labels
        counter = self._jcc_counter
        self._jcc_counter += 1
        loop_label = f"_div_loop_{counter}"
        after_sub_label = f"_div_after_sub_{counter}"
        do_sub_label = f"_div_do_sub_{counter}"
        skip_sub_label = f"_div_skip_sub_{counter}"
        exit_label = f"_div_exit_{counter}"
        divzero_check_label = f"_div_zchk_{counter}"
        divzero_ok_label = f"_div_zok_{counter}"
        overflow_check_label = f"_div_ovchk_{counter}"
        overflow_ok_label = f"_div_ovok_{counter}"
        dispatch_sub_label = f"_div_subdisp_{counter}"
        dispatch_loop_label = f"_div_loopdisp_{counter}"
        div_trap_label = f"_div_trap_{counter}"

        statements: List[Statement] = []
        line = instr.line

        def mov(mnem: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnem, src_op, dst_op)

        div_remainder = Memory(displacement="div_remainder")
        div_dividend_lo = Memory(displacement="div_dividend_lo")
        div_divisor = Memory(displacement="div_divisor")
        div_quotient = Memory(displacement="div_quotient")
        div_counter = Memory(displacement="div_counter")
        scratch_a = Memory(displacement="scratch_a")
        scratch_b = Memory(displacement="scratch_b")
        scratch_r = Memory(displacement="scratch_r")
        scratch_t = Memory(displacement="scratch_t")

        save_map = self._emit_save_temps(statements, line)

        # 1. Store dividend hi/lo (EDX:EAX) to div state
        statements.append(mov("movl", save_map["edx"], Register("eax")))
        statements.append(mov("movl", Register("eax"), div_remainder))
        statements.append(mov("movl", save_map["eax"], Register("eax")))
        statements.append(mov("movl", Register("eax"), div_dividend_lo))

        # 2. Store divisor to div_divisor
        self._emit_store_operand_long_to_memory(
            statements, operand, div_divisor, line, save_map
        )

        # 3. Clear quotient
        statements.append(mov("movl", Immediate(0), div_quotient))

        # 3a. Division-by-zero check: if divisor == 0, trap
        # OR all 4 bytes of div_divisor together, then check is_zero
        statements.append(
            mov("movzbl", Memory(displacement="div_divisor"), Register("ecx"))
        )
        statements.append(
            mov(
                "movl",
                Memory(displacement="or_row_ptrs", index=Register("ecx"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="div_divisor+1"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )
        statements.append(
            mov(
                "movl",
                Memory(displacement="or_row_ptrs", index=Register("eax"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="div_divisor+2"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )
        statements.append(
            mov(
                "movl",
                Memory(displacement="or_row_ptrs", index=Register("eax"), scale=4),
                Register("ecx"),
            )
        )
        statements.append(
            mov("movzbl", Memory(displacement="div_divisor+3"), Register("edx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(base=Register("ecx"), index=Register("edx"), scale=1),
                Register("eax"),
            )
        )
        # eax = OR of all divisor bytes. If 0, divisor is zero.
        # is_zero_lut[eax]: 1 if zero (error), 0 if non-zero (ok)
        statements.append(
            mov(
                "movzbl",
                Memory(displacement="is_zero_lut", index=Register("eax"), scale=1),
                Register("eax"),
            )
        )
        # Dispatch: index=0 -> continue (non-zero divisor), index=1 -> divide error
        statements.append(Directive(name="section", args=[".data"], line=instr.line))
        statements.append(Label(name=divzero_check_label, line=instr.line))
        statements.append(
            Directive(name="long", args=[divzero_ok_label], line=instr.line)
        )
        # Entry 1 points to a label that does int $0.
        statements.append(
            Directive(name="long", args=[div_trap_label], line=instr.line)
        )
        statements.append(Directive(name="section", args=[".text"], line=instr.line))
        dispatch_zchk = Memory(
            displacement=divzero_check_label, index=Register("eax"), scale=4
        )
        statements.append(mov("movl", dispatch_zchk, Register("eax")))
        statements.append(
            mov("movl", Register("eax"), Memory(displacement="scratch_jcc_target"))
        )
        statements.append(
            Instruction(
                mnemonic="jmp",
                operands=[LabelRef(name="*scratch_jcc_target")],
                line=instr.line,
            )
        )
        # Divide error trap: INT $0 (matches real x86 #DE exception)
        statements.append(Label(name=div_trap_label, line=line))
        statements.append(mov("movl", save_map["eax"], Register("eax")))
        statements.append(mov("movl", save_map["ecx"], Register("ecx")))
        statements.append(mov("movl", save_map["edx"], Register("edx")))
        statements.append(
            Instruction(mnemonic="int", operands=[Immediate(0)], line=line)
        )
        # Continue after zero-check passed
        statements.append(Label(name=divzero_ok_label, line=line))

        # 3b. Quotient overflow check: if dividend_hi >= divisor, trap
        # Compute below = (dividend_hi < divisor) using CMP emulation.
        statements.append(mov("movl", div_divisor, Register("eax")))
        statements.append(mov("movl", Register("eax"), scratch_a))
        statements.append(mov("movl", div_remainder, Register("eax")))
        statements.append(mov("movl", Register("eax"), scratch_b))
        self._emit_cmp_below_and_sign_lt(statements, line)

        # Dispatch: index=0 (not below) -> trap, index=1 (below) -> ok
        statements.append(Directive(name="section", args=[".data"], line=line))
        statements.append(Label(name=overflow_check_label, line=line))
        statements.append(Directive(name="long", args=[div_trap_label], line=line))
        statements.append(Directive(name="long", args=[overflow_ok_label], line=line))
        statements.append(Directive(name="section", args=[".text"], line=line))
        statements.append(
            mov(
                "movzbl",
                Memory(displacement="scratch_cmp_below"),
                Register("eax"),
            )
        )
        dispatch_ov = Memory(
            displacement=overflow_check_label, index=Register("eax"), scale=4
        )
        statements.append(mov("movl", dispatch_ov, Register("eax")))
        statements.append(
            mov("movl", Register("eax"), Memory(displacement="scratch_jcc_target"))
        )
        statements.append(
            Instruction(
                mnemonic="jmp",
                operands=[LabelRef(name="*scratch_jcc_target")],
                line=line,
            )
        )
        statements.append(Label(name=overflow_ok_label, line=line))

        # 4. Initialize quotient and loop counter (32 quotient bits)
        statements.append(mov("movl", Immediate(0), div_quotient))
        statements.append(mov("movl", Immediate(32), div_counter))

        # 5. Dispatch table for conditional subtract (based on scratch_cmp_below)
        statements.append(Directive(name="section", args=[".data"], line=line))
        statements.append(Label(name=dispatch_sub_label, line=line))
        # below=0: remainder >= divisor -> do_sub
        statements.append(Directive(name="long", args=[do_sub_label], line=line))
        # below=1: remainder < divisor -> skip_sub
        statements.append(Directive(name="long", args=[skip_sub_label], line=line))
        statements.append(Directive(name="section", args=[".text"], line=line))

        # 6. Dispatch table for loop continuation (based on counter != 0)
        statements.append(Directive(name="section", args=[".data"], line=line))
        statements.append(Label(name=dispatch_loop_label, line=line))
        # idx=0: counter == 0 -> exit
        statements.append(Directive(name="long", args=[exit_label], line=line))
        # idx=1: counter != 0 -> loop
        statements.append(Directive(name="long", args=[loop_label], line=line))
        statements.append(Directive(name="section", args=[".text"], line=line))

        # 7. Long division loop (32 iterations)
        statements.append(Label(name=loop_label, line=line))

        # 7a. inject = (div_dividend_lo >> 31) & 1
        statements.append(
            mov("movzbl", Memory(displacement="div_dividend_lo+3"), Register("ecx"))
        )
        statements.append(
            mov(
                "movzbl",
                Memory(displacement="shl1_carry_lut", index=Register("ecx"), scale=1),
                Register("eax"),
            )
        )
        # scratch_t = inject (as a 32-bit 0/1)
        statements.append(mov("movl", Immediate(0), scratch_t))
        statements.append(mov("movb", Register("al"), scratch_t))

        # 7b. remainder = (remainder << 1) + inject
        shl_rem = Instruction(
            mnemonic="shll",
            operands=[Immediate(1), div_remainder],
            line=line,
        )
        statements.extend(
            self._translate_shl(shl_rem, update_flags=False, save_temps=False)
        )
        add_inject = Instruction(
            mnemonic="addl",
            operands=[scratch_t, div_remainder],
            line=line,
        )
        statements.extend(
            self._translate_add(add_inject, update_flags=False, save_temps=False)
        )

        # 7c. dividend_lo <<= 1
        shl_lo = Instruction(
            mnemonic="shll",
            operands=[Immediate(1), div_dividend_lo],
            line=line,
        )
        statements.extend(
            self._translate_shl(shl_lo, update_flags=False, save_temps=False)
        )

        # 7d. quotient <<= 1
        shl_q = Instruction(
            mnemonic="shll",
            operands=[Immediate(1), div_quotient],
            line=line,
        )
        statements.extend(
            self._translate_shl(shl_q, update_flags=False, save_temps=False)
        )

        # 7e. Compare remainder with divisor (unsigned): below = remainder < divisor
        statements.append(mov("movl", div_divisor, Register("eax")))
        statements.append(mov("movl", Register("eax"), scratch_a))
        statements.append(mov("movl", div_remainder, Register("eax")))
        statements.append(mov("movl", Register("eax"), scratch_b))
        self._emit_cmp_below_and_sign_lt(statements, line)

        # 7f. Dispatch on below: do_sub if below==0 else skip
        statements.append(
            mov(
                "movzbl",
                Memory(displacement="scratch_cmp_below"),
                Register("eax"),
            )
        )
        sub_dispatch = Memory(
            displacement=dispatch_sub_label, index=Register("eax"), scale=4
        )
        statements.append(mov("movl", sub_dispatch, Register("eax")))
        statements.append(
            mov("movl", Register("eax"), Memory(displacement="scratch_jcc_target"))
        )
        statements.append(
            Instruction(
                mnemonic="jmp",
                operands=[LabelRef(name="*scratch_jcc_target")],
                line=line,
            )
        )

        # 7g. do_sub: remainder -= divisor; quotient++
        statements.append(Label(name=do_sub_label, line=line))
        statements.append(mov("movl", div_divisor, Register("eax")))
        statements.append(mov("movl", Register("eax"), scratch_a))
        statements.append(mov("movl", div_remainder, Register("eax")))
        statements.append(mov("movl", Register("eax"), scratch_b))
        self._emit_sub_bytes(statements, line)
        statements.append(mov("movl", scratch_r, Register("eax")))
        statements.append(mov("movl", Register("eax"), div_remainder))

        statements.append(mov("movl", div_quotient, Register("eax")))
        statements.append(mov("movl", Register("eax"), scratch_b))
        self._emit_inc_bytes(statements, line)
        statements.append(mov("movl", scratch_r, Register("eax")))
        statements.append(mov("movl", Register("eax"), div_quotient))
        statements.append(
            Instruction(
                mnemonic="jmp", operands=[LabelRef(name=after_sub_label)], line=line
            )
        )

        # 7h. skip_sub
        statements.append(Label(name=skip_sub_label, line=line))
        statements.append(Label(name=after_sub_label, line=line))

        # 7i. counter--
        statements.append(mov("movl", Immediate(1), scratch_a))
        statements.append(mov("movl", div_counter, Register("eax")))
        statements.append(mov("movl", Register("eax"), scratch_b))
        self._emit_sub_bytes(statements, line)
        statements.append(mov("movl", scratch_r, Register("eax")))
        statements.append(mov("movl", Register("eax"), div_counter))
        self._emit_update_zf(statements, line)

        # idx = (counter != 0) = is_zero_lut[scratch_cmp_eq]
        statements.append(
            mov(
                "movzbl",
                Memory(displacement="scratch_cmp_eq"),
                Register("eax"),
            )
        )
        statements.append(
            mov(
                "movzbl",
                Memory(displacement="is_zero_lut", index=Register("eax"), scale=1),
                Register("eax"),
            )
        )
        loop_dispatch = Memory(
            displacement=dispatch_loop_label, index=Register("eax"), scale=4
        )
        statements.append(mov("movl", loop_dispatch, Register("eax")))
        statements.append(
            mov("movl", Register("eax"), Memory(displacement="scratch_jcc_target"))
        )
        statements.append(
            Instruction(
                mnemonic="jmp",
                operands=[LabelRef(name="*scratch_jcc_target")],
                line=line,
            )
        )

        # 8. Exit: load results and restore ECX only (EAX/EDX are outputs)
        statements.append(Label(name=exit_label, line=line))
        statements.append(mov("movl", div_quotient, Register("eax")))
        statements.append(mov("movl", div_remainder, Register("edx")))
        statements.append(mov("movl", save_map["ecx"], Register("ecx")))

        return statements

    def _translate_jcc_equality(self, instr: Instruction) -> List[Statement]:
        """Translate equality Jcc (je/jne/jz/jnz) using scratch_cmp_eq dispatch."""
        if len(instr.operands) != 1:
            raise TranslatorError(
                f"Conditional jump requires 1 operand (label), got {len(instr.operands)}",
                instr.line,
            )

        target = instr.operands[0]
        if not isinstance(target, LabelRef):
            raise TranslatorError(
                f"Conditional jump operand must be a label", instr.line
            )

        mnemonic = instr.mnemonic.lower()
        counter = self._jcc_counter
        self._jcc_counter += 1

        table_label = f"_jcc_table_{counter}"
        fallthrough_label = f"_jcc_fallthrough_{counter}"
        target_label = target.name

        statements: List[Statement] = []
        line = instr.line

        def mov(mnem: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnem, src_op, dst_op)

        save_map = self._emit_save_temps(statements, line)

        # 1. Emit dispatch table
        statements.append(Directive(name="section", args=[".data"], line=line))
        statements.append(Label(name=table_label, line=line))
        statements.append(Directive(name="long", args=[fallthrough_label], line=line))
        statements.append(Directive(name="long", args=[target_label], line=line))
        statements.append(Directive(name="section", args=[".text"], line=line))

        # 3. Compute condition index
        if mnemonic in ("je", "jz"):
            # index = scratch_cmp_eq
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="scratch_cmp_eq"),
                    Register("eax"),
                )
            )
        elif mnemonic in ("jne", "jnz"):
            # index = NOT(scratch_cmp_eq) = is_zero_lut[eq]
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="scratch_cmp_eq"),
                    Register("eax"),
                )
            )
            statements.append(
                mov(
                    "movzbl",
                    Memory(displacement="is_zero_lut", index=Register("eax"), scale=1),
                    Register("eax"),
                )
            )

        # 4. Load target address from dispatch table
        dispatch_lookup = Memory(
            displacement=table_label, index=Register("eax"), scale=4
        )
        statements.append(mov("movl", dispatch_lookup, Register("eax")))

        # 5. Store target address
        statements.append(
            mov("movl", Register("eax"), Memory(displacement="scratch_jcc_target"))
        )

        # 6. Restore temps
        self._emit_restore_temps(statements, line, None, save_map)

        # 7. Indirect jump
        statements.append(
            Instruction(
                mnemonic="jmp",
                operands=[LabelRef(name="*scratch_jcc_target")],
                line=line,
            )
        )

        # 8. Fallthrough label
        statements.append(Label(name=fallthrough_label, line=line))

        return statements

    def _translate_shl(
        self,
        instr: Instruction,
        update_flags: bool = True,
        save_temps: bool = True,
    ) -> List[Statement]:
        """Translate SHL $imm, dst to MOV-only sequence."""
        if len(instr.operands) != 2:
            raise TranslatorError(
                f"SHL requires 2 operands, got {len(instr.operands)}", instr.line
            )

        src, dst = (
            self._normalize_operand(instr.operands[0]),
            self._normalize_operand(instr.operands[1]),
        )

        # Source must be an immediate
        if not isinstance(src, Immediate):
            raise TranslatorError("SHL source must be an immediate value", instr.line)

        # Parse shift count
        try:
            shift_count = int(str(src.value), 0)
        except ValueError:
            raise TranslatorError(
                f"SHL shift count must be a number, got: {src.value}", instr.line
            )

        if shift_count < 0 or shift_count > 31:
            raise TranslatorError(
                f"SHL shift count must be 0-31, got: {shift_count}", instr.line
            )

        # Shift by 0 is a no-op
        if shift_count == 0:
            return []

        self._validate_32bit_operands(instr, [dst], "SHL")

        statements: List[Statement] = []
        line = instr.line

        def mov(mnemonic: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnemonic, src_op, dst_op)

        scratch_b = self._scratch_mem("scratch_b")
        scratch_r = self._scratch_mem("scratch_r")
        scratch_c = self._scratch_mem("scratch_c")

        dst_reg_name = dst.name if isinstance(dst, Register) else None

        save_map = (
            self._emit_save_temps(statements, line)
            if save_temps
            else self._temp_save_locations()
        )

        if isinstance(dst, Immediate):
            raise TranslatorError("SHL destination must be register or memory", line)
        self._emit_store_operand_long_to_scratch(
            statements, dst, "scratch_b", line, save_map
        )

        # 2. Perform shift_count iterations of shift-left-by-1
        for _iteration in range(shift_count):
            # Initialize carry to 0
            statements.append(mov("movb", Immediate(0), scratch_c))

            # Process bytes 0 to 3 (LSB to MSB)
            for byte_idx in range(4):
                b_byte = self._scratch_mem("scratch_b", byte_idx)
                r_byte = self._scratch_mem("scratch_r", byte_idx)

                # Load original byte into ecx
                statements.append(mov("movzbl", b_byte, Register("ecx")))

                # shifted = shl1_lut[byte]
                shl1_lookup = Memory(
                    displacement="shl1_lut", index=Register("ecx"), scale=1
                )
                statements.append(mov("movzbl", shl1_lookup, Register("eax")))

                # carry_out = shl1_carry_lut[byte]
                carry_lookup = Memory(
                    displacement="shl1_carry_lut", index=Register("ecx"), scale=1
                )
                statements.append(mov("movzbl", carry_lookup, Register("ecx")))

                # Load carry_in from scratch_c
                statements.append(mov("movzbl", scratch_c, Register("edx")))

                # result = add_lut[shifted][carry_in]
                # (shl1_lut always has bit 0 = 0, carry_in is 0 or 1, so no overflow)
                add_row_ptr = Memory(
                    displacement="add_row_ptrs", index=Register("eax"), scale=4
                )
                statements.append(mov("movl", add_row_ptr, Register("eax")))
                add_lookup = Memory(
                    base=Register("eax"), index=Register("edx"), scale=1
                )
                statements.append(mov("movzbl", add_lookup, Register("eax")))

                # Store result byte
                statements.append(mov("movb", Register("al"), r_byte))

                # Store carry_out for next byte
                statements.append(mov("movb", Register("cl"), scratch_c))

            # Copy scratch_r back to scratch_b for next iteration (if needed)
            if _iteration < shift_count - 1:
                self._emit_store_operand_long_to_scratch(
                    statements, scratch_r, "scratch_b", line, save_map
                )

        # 3. Update ZF (scratch_cmp_eq) from result
        if update_flags:
            self._emit_update_zf(statements, line)

        # 4. Store result to destination
        if isinstance(dst, Register):
            statements.append(mov("movl", scratch_r, dst))
        elif isinstance(dst, Memory):
            self._emit_store_long_to_memory(
                statements,
                scratch_r,
                dst,
                save_map["eax"],
                save_map["ecx"],
                save_map["edx"],
                line,
            )

        # 5. Restore temp registers (except destination if it's a temp reg)
        if save_temps:
            self._emit_restore_temps(statements, line, dst_reg_name, save_map)

        return statements

    def _translate_shr(
        self, instr: Instruction, update_flags: bool = True
    ) -> List[Statement]:
        """Translate SHR $imm, dst to MOV-only sequence."""
        if len(instr.operands) != 2:
            raise TranslatorError(
                f"SHR requires 2 operands, got {len(instr.operands)}", instr.line
            )

        src, dst = (
            self._normalize_operand(instr.operands[0]),
            self._normalize_operand(instr.operands[1]),
        )

        # Source must be an immediate
        if not isinstance(src, Immediate):
            raise TranslatorError("SHR source must be an immediate value", instr.line)

        # Parse shift count
        try:
            shift_count = int(str(src.value), 0)
        except ValueError:
            raise TranslatorError(
                f"SHR shift count must be a number, got: {src.value}", instr.line
            )

        if shift_count < 0 or shift_count > 31:
            raise TranslatorError(
                f"SHR shift count must be 0-31, got: {shift_count}", instr.line
            )

        # Shift by 0 is a no-op
        if shift_count == 0:
            return []

        self._validate_32bit_operands(instr, [dst], "SHR")

        statements: List[Statement] = []
        line = instr.line

        def mov(mnemonic: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnemonic, src_op, dst_op)

        scratch_b = self._scratch_mem("scratch_b")
        scratch_r = self._scratch_mem("scratch_r")
        scratch_c = self._scratch_mem("scratch_c")

        dst_reg_name = dst.name if isinstance(dst, Register) else None

        save_map = self._emit_save_temps(statements, line)

        if isinstance(dst, Immediate):
            raise TranslatorError("SHR destination must be register or memory", line)
        self._emit_store_operand_long_to_scratch(
            statements, dst, "scratch_b", line, save_map
        )

        # 2. Perform shift_count iterations of shift-right-by-1
        for _iteration in range(shift_count):
            # Initialize carry to 0
            statements.append(mov("movb", Immediate(0), scratch_c))

            # Process bytes 3 to 0 (MSB to LSB)
            for byte_idx in range(3, -1, -1):
                b_byte = self._scratch_mem("scratch_b", byte_idx)
                r_byte = self._scratch_mem("scratch_r", byte_idx)

                # Load original byte into ecx
                statements.append(mov("movzbl", b_byte, Register("ecx")))

                # shifted = shr1_lut[byte]
                shr1_lookup = Memory(
                    displacement="shr1_lut", index=Register("ecx"), scale=1
                )
                statements.append(mov("movzbl", shr1_lookup, Register("eax")))

                # carry_out = shr1_carry_lut[byte] (bit 0 of original)
                carry_lookup = Memory(
                    displacement="shr1_carry_lut", index=Register("ecx"), scale=1
                )
                statements.append(mov("movzbl", carry_lookup, Register("ecx")))

                # Load carry_in from scratch_c and convert to addend (0->0, 1->128)
                statements.append(mov("movzbl", scratch_c, Register("edx")))
                inject_lookup = Memory(
                    displacement="shr_carry_inject",
                    index=Register("edx"),
                    scale=1,
                )
                statements.append(mov("movzbl", inject_lookup, Register("edx")))

                # result = add_lut[shifted][addend]
                # (shr1_lut always has bit 7 = 0, addend is 0 or 128, no overflow)
                add_row_ptr = Memory(
                    displacement="add_row_ptrs", index=Register("eax"), scale=4
                )
                statements.append(mov("movl", add_row_ptr, Register("eax")))
                add_lookup = Memory(
                    base=Register("eax"), index=Register("edx"), scale=1
                )
                statements.append(mov("movzbl", add_lookup, Register("eax")))

                # Store result byte
                statements.append(mov("movb", Register("al"), r_byte))

                # Store carry_out for next byte
                statements.append(mov("movb", Register("cl"), scratch_c))

            # Copy scratch_r back to scratch_b for next iteration (if needed)
            if _iteration < shift_count - 1:
                self._emit_store_operand_long_to_scratch(
                    statements, scratch_r, "scratch_b", line, save_map
                )

        # 3. Update ZF (scratch_cmp_eq) from result
        if update_flags:
            self._emit_update_zf(statements, line)

        # 4. Store result to destination
        if isinstance(dst, Register):
            statements.append(mov("movl", scratch_r, dst))
        elif isinstance(dst, Memory):
            self._emit_store_long_to_memory(
                statements,
                scratch_r,
                dst,
                save_map["eax"],
                save_map["ecx"],
                save_map["edx"],
                line,
            )

        self._emit_restore_temps(statements, line, dst_reg_name, save_map)

        return statements

    def _translate_push(self, instr: Instruction) -> List[Statement]:
        """Translate PUSH src to MOV-only sequence."""
        if len(instr.operands) != 1:
            raise TranslatorError(
                f"PUSH requires 1 operand, got {len(instr.operands)}", instr.line
            )

        src = self._normalize_operand(instr.operands[0])

        self._validate_32bit_operands(instr, [src], "PUSH")

        statements: List[Statement] = []
        line = instr.line

        def mov(mnemonic: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnemonic, src_op, dst_op)

        scratch_t = Memory(displacement="scratch_t")
        scratch_t2 = Memory(displacement="scratch_t2")

        # Special case: pushl %esp - must save original ESP before decrement
        is_push_esp = isinstance(src, Register) and src.name == "esp"
        if is_push_esp:
            statements.append(mov("movl", Register("esp"), scratch_t))

        # 1. ESP -= 4 (create synthetic SUB and translate it)
        sub_instr = Instruction(
            mnemonic="subl",
            operands=[Immediate(4), Register("esp")],
            line=line,
        )
        statements.extend(self._translate_sub(sub_instr, update_flags=False))

        # 2. Store value to (%esp)
        esp_indirect = Memory(base=Register("esp"))

        if is_push_esp:
            # Store the saved original ESP value
            statements.append(mov("movl", scratch_t, Register("eax")))
            statements.append(mov("movl", Register("eax"), esp_indirect))
        elif isinstance(src, Register):
            # After _translate_sub, temp registers are restored properly
            statements.append(mov("movl", src, esp_indirect))
        elif isinstance(src, Immediate):
            statements.append(mov("movl", src, esp_indirect))
        elif isinstance(src, Memory):
            # Memory to memory needs intermediate register
            # Save eax (scratch_t2), use it, restore
            statements.append(mov("movl", Register("eax"), scratch_t2))
            statements.append(mov("movl", src, Register("eax")))
            statements.append(mov("movl", Register("eax"), esp_indirect))
            statements.append(mov("movl", scratch_t2, Register("eax")))
        else:
            raise TranslatorError(f"Unsupported operand type for PUSH", instr.line)

        return statements

    def _translate_pop(self, instr: Instruction) -> List[Statement]:
        """Translate POP dst to MOV-only sequence."""
        if len(instr.operands) != 1:
            raise TranslatorError(
                f"POP requires 1 operand, got {len(instr.operands)}", instr.line
            )

        dst = self._normalize_operand(instr.operands[0])

        # Only register and memory destinations are valid
        if not isinstance(dst, (Register, Memory)):
            raise TranslatorError(
                f"POP destination must be a register or memory operand, "
                f"got {type(dst).__name__}",
                instr.line,
            )

        self._validate_32bit_operands(instr, [dst], "POP")

        statements: List[Statement] = []
        line = instr.line

        def mov(mnemonic: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnemonic, src_op, dst_op)

        # Stack pointer indirect reference
        esp_indirect = Memory(base=Register("esp"))

        scratch_t2 = Memory(displacement="scratch_t2")

        # Special case: popl %esp => ESP = *(ESP), no +4 adjustment
        if isinstance(dst, Register) and dst.name == "esp":
            statements.append(mov("movl", esp_indirect, Register("esp")))
            return statements

        if isinstance(dst, Register):
            # 1. Load value from (%esp) to destination register
            statements.append(mov("movl", esp_indirect, dst))
        elif isinstance(dst, Memory):
            # Memory-to-memory: route through eax
            # Save eax (scratch_t2), load from stack into eax, store to memory dst, restore eax
            statements.append(mov("movl", Register("eax"), scratch_t2))
            statements.append(mov("movl", esp_indirect, Register("eax")))
            statements.append(mov("movl", Register("eax"), dst))
            statements.append(mov("movl", scratch_t2, Register("eax")))

        # 2. ESP += 4 (create synthetic ADD and translate it)
        add_instr = Instruction(
            mnemonic="addl",
            operands=[Immediate(4), Register("esp")],
            line=line,
        )
        statements.extend(self._translate_add(add_instr, update_flags=False))

        return statements

    def _translate_lea(self, instr: Instruction) -> List[Statement]:
        """Translate LEA (Load Effective Address) to MOV-only.

        LEA computes: dest = displacement + base + (index * scale)

        Handles all forms:
        - lea label, %reg              -> movl $label, %reg
        - lea (%base), %reg            -> movl %base, %reg
        - lea offset(%base), %reg      -> base + offset via ADD
        - lea (%base, %index, scale), %reg -> base + index*scale
        - lea offset(%base, %index, scale), %reg -> full computation
        """
        if len(instr.operands) != 2:
            raise TranslatorError(
                f"LEA requires exactly 2 operands, got {len(instr.operands)}",
                instr.line,
            )

        src, dst = instr.operands[0], instr.operands[1]

        # Destination must be a register
        if not isinstance(dst, Register):
            raise TranslatorError(
                f"LEA destination must be a register, got {type(dst).__name__}",
                instr.line,
            )

        # Handle LabelRef source: lea label, %reg -> movl $label, %reg
        if isinstance(src, LabelRef):
            return [
                Instruction(
                    mnemonic="movl",
                    operands=[Immediate(src.name, is_label=True), dst],
                    line=instr.line,
                )
            ]

        if not isinstance(src, Memory):
            raise TranslatorError(
                f"LEA source must be a label or memory reference, got {type(src).__name__}",
                instr.line,
            )

        # Simple case: displacement only (no base, no index)
        if src.base is None and src.index is None:
            if src.displacement is not None:
                return [
                    Instruction(
                        mnemonic="movl",
                        operands=[
                            Immediate(str(src.displacement), is_label=True),
                            dst,
                        ],
                        line=instr.line,
                    )
                ]
            # No displacement either - effectively lea 0, %reg
            return [
                Instruction(
                    mnemonic="movl",
                    operands=[Immediate(0), dst],
                    line=instr.line,
                )
            ]

        # Simple case: base only, no index, no displacement -> just copy base
        if src.index is None and src.displacement is None and src.base is not None:
            return [
                Instruction(
                    mnemonic="movl",
                    operands=[src.base, dst],
                    line=instr.line,
                )
            ]

        # Complex LEA: need to compute base + index*scale + offset
        return self._translate_lea_complex(instr, src, dst)

    def _translate_lea_complex(
        self, instr: Instruction, src: Memory, dst: Register
    ) -> List[Statement]:
        """Translate complex LEA (disp + base + index*scale) to MOV-only sequence."""
        statements: List[Statement] = []
        line = instr.line

        def mov(mnemonic: str, src_op: Operand, dst_op: Operand) -> Instruction:
            return self._mov_instr(line, mnemonic, src_op, dst_op)

        # Check if source operands use temp regs (need special handling)
        base_is_temp = src.base and src.base.name in self.TEMP_REGS
        index_is_temp = src.index and src.index.name in self.TEMP_REGS

        # Save base/index DIRECTLY from register FIRST, before any other saves.
        # Use dedicated scratch slots to avoid clobbering by nested translations.
        # This handles destination overlap cases like: lea (%eax, %ecx, 4), %eax
        lea_base_saved = Memory(displacement="lea_base_saved")
        lea_index_saved = Memory(displacement="lea_index_saved")

        if base_is_temp:
            assert src.base is not None  # Type guard
            statements.append(mov("movl", src.base, lea_base_saved))

        if index_is_temp:
            assert src.index is not None  # Type guard
            # If base and index are the same temp reg, re-use the saved base.
            if not (
                base_is_temp
                and src.base is not None
                and src.index.name == src.base.name
            ):
                statements.append(mov("movl", src.index, lea_index_saved))

        # 1. Save temp registers (for ADD/SHL sub-translations)
        save_map = self._emit_save_temps(statements, line)

        # 2. Initialize scratch_r with displacement
        if src.displacement is not None:
            disp = src.displacement
            if isinstance(disp, int) or (
                isinstance(disp, str) and disp.lstrip("-").isdigit()
            ):
                # Numeric displacement
                statements.append(
                    mov("movl", Immediate(str(disp)), Memory(displacement="scratch_r"))
                )
            else:
                # Label displacement - load label address
                statements.append(
                    mov(
                        "movl",
                        Immediate(str(disp), is_label=True),
                        Memory(displacement="scratch_r"),
                    )
                )
        else:
            statements.append(
                mov("movl", Immediate(0), Memory(displacement="scratch_r"))
            )

        # 3. Add base if present
        if src.base is not None:
            # Get base value (from saved location if it was a temp reg)
            if base_is_temp:
                base_src = Memory(displacement="lea_base_saved")
            else:
                base_src = src.base

            # Store base to scratch_a (avoid illegal mem-to-mem MOV)
            if isinstance(base_src, Memory):
                statements.append(mov("movl", base_src, Register("eax")))
                statements.append(
                    mov("movl", Register("eax"), Memory(displacement="scratch_a"))
                )
            else:
                statements.append(
                    mov("movl", base_src, Memory(displacement="scratch_a"))
                )

            # Add scratch_a to scratch_r using ADD translation
            # Create synthetic: addl scratch_a, scratch_r
            add_instr = Instruction(
                mnemonic="addl",
                operands=[
                    Memory(displacement="scratch_a"),
                    Memory(displacement="scratch_r"),
                ],
                line=instr.line,
            )
            statements.extend(
                self._translate_add(add_instr, update_flags=False, save_temps=False)
            )

        # 4. Add index * scale if index is present
        if src.index is not None:
            scale = src.scale if src.scale else 1

            scratch_r_saved = Memory(displacement="scratch_r_saved")

            # Get index value
            if index_is_temp:
                if (
                    base_is_temp
                    and src.base is not None
                    and src.index.name == src.base.name
                ):
                    index_src: Operand = Memory(displacement="lea_base_saved")
                else:
                    index_src = Memory(displacement="lea_index_saved")
            else:
                index_src = src.index

            # Store index to scratch_a (avoid illegal mem-to-mem MOV)
            if isinstance(index_src, Memory):
                statements.append(mov("movl", index_src, Register("eax")))
                statements.append(
                    mov("movl", Register("eax"), Memory(displacement="scratch_a"))
                )
            else:
                statements.append(
                    mov("movl", index_src, Memory(displacement="scratch_a"))
                )

            # Validate scale factor
            if scale not in (1, 2, 4, 8):
                raise TranslatorError(
                    f"LEA scale must be 1, 2, 4, or 8, got {scale}",
                    instr.line,
                )

            # Multiply by scale using shift left
            # scale=1: no shift, scale=2: shl 1, scale=4: shl 2, scale=8: shl 3
            if scale == 2:
                statements.append(
                    mov("movl", Memory(displacement="scratch_r"), Register("eax"))
                )
                statements.append(mov("movl", Register("eax"), scratch_r_saved))
                shl_instr = Instruction(
                    mnemonic="shll",
                    operands=[Immediate(1), Memory(displacement="scratch_a")],
                    line=instr.line,
                )
                statements.extend(
                    self._translate_shl(shl_instr, update_flags=False, save_temps=False)
                )
                statements.append(mov("movl", scratch_r_saved, Register("eax")))
                statements.append(
                    mov("movl", Register("eax"), Memory(displacement="scratch_r"))
                )
            elif scale == 4:
                statements.append(
                    mov("movl", Memory(displacement="scratch_r"), Register("eax"))
                )
                statements.append(mov("movl", Register("eax"), scratch_r_saved))
                shl_instr = Instruction(
                    mnemonic="shll",
                    operands=[Immediate(2), Memory(displacement="scratch_a")],
                    line=instr.line,
                )
                statements.extend(
                    self._translate_shl(shl_instr, update_flags=False, save_temps=False)
                )
                statements.append(mov("movl", scratch_r_saved, Register("eax")))
                statements.append(
                    mov("movl", Register("eax"), Memory(displacement="scratch_r"))
                )
            elif scale == 8:
                statements.append(
                    mov("movl", Memory(displacement="scratch_r"), Register("eax"))
                )
                statements.append(mov("movl", Register("eax"), scratch_r_saved))
                shl_instr = Instruction(
                    mnemonic="shll",
                    operands=[Immediate(3), Memory(displacement="scratch_a")],
                    line=instr.line,
                )
                statements.extend(
                    self._translate_shl(shl_instr, update_flags=False, save_temps=False)
                )
                statements.append(mov("movl", scratch_r_saved, Register("eax")))
                statements.append(
                    mov("movl", Register("eax"), Memory(displacement="scratch_r"))
                )
            # scale=1: no shift needed

            # Add scaled index to result
            add_instr = Instruction(
                mnemonic="addl",
                operands=[
                    Memory(displacement="scratch_a"),
                    Memory(displacement="scratch_r"),
                ],
                line=instr.line,
            )
            statements.extend(
                self._translate_add(add_instr, update_flags=False, save_temps=False)
            )

        # 5. Store result to destination
        statements.append(mov("movl", Memory(displacement="scratch_r"), dst))

        # 6. Restore temp registers (except destination)
        self._emit_restore_temps(statements, line, dst.name, save_map)

        return statements


def translate(program: Program, config: Optional[TranslatorConfig] = None) -> Program:
    translator = Translator(config)
    return translator.translate(program)
