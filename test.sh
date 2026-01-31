#!/bin/bash
# Integration test runner for the movfuscator.
# Translates each sample, compiles both original and translated,
# runs both, and compares exit codes + stdout.

set -euo pipefail

IN_DIR="samples/in"
IN_EXTRA_DIR="samples/in_extra"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'

pass=0
fail=0
skip=0

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}        MOVFUSCATOR INTEGRATION TEST RUNNER             ${NC}"
echo -e "${BLUE}=======================================================${NC}"

if [ ! -d "$IN_DIR" ]; then
    echo -e "${RED}Error: Directory '$IN_DIR' not found.${NC}"
    exit 1
fi

for in_source in "$IN_DIR"/*.S "$IN_DIR"/*.s "$IN_EXTRA_DIR"/*.S "$IN_EXTRA_DIR"/*.s; do
    [ -e "$in_source" ] || continue

    filename=$(basename "$in_source")
    name="${filename%.*}"

    echo -e "\n--- ${BLUE}$filename${NC} ---"

    # 1. Translate
    trans_source="$TMP_DIR/trans_${name}.s"
    trans_err=$(python "$SCRIPT_DIR/src/main.py" "$in_source" -o "$trans_source" 2>&1) || {
        echo -e "  ${RED}[TRANSLATE ERROR]${NC} $trans_err"
        fail=$((fail + 1))
        continue
    }

    # 2. Compile original
    exe_orig="$TMP_DIR/orig_${name}"
    compile_err=$(gcc -m32 "$in_source" -o "$exe_orig" -w -no-pie 2>&1) || {
        echo -e "  ${YELLOW}[SKIP]${NC} Original won't compile: $compile_err"
        skip=$((skip + 1))
        continue
    }

    # 3. Compile translated
    exe_trans="$TMP_DIR/trans_${name}"
    compile_err=$(gcc -m32 "$trans_source" -o "$exe_trans" -w -no-pie 2>&1) || {
        echo -e "  ${RED}[COMPILE ERROR]${NC} Translated won't compile:"
        echo "  $compile_err"
        fail=$((fail + 1))
        continue
    }

    # 4. Run original (with timeout)
    orig_out="$TMP_DIR/orig_out_${name}"
    orig_exit=0
    timeout 10 "$exe_orig" > "$orig_out" 2>&1 || orig_exit=$?

    # 5. Run translated (with timeout)
    trans_out="$TMP_DIR/trans_out_${name}"
    trans_exit=0
    timeout 10 "$exe_trans" > "$trans_out" 2>&1 || trans_exit=$?

    # 6. Compare
    orig_stdout=$(cat "$orig_out" | tr -d '\0')
    trans_stdout=$(cat "$trans_out" | tr -d '\0')

    failed=0

    if [ $orig_exit -ne $trans_exit ]; then
        echo -e "  ${RED}FAIL: Exit code mismatch${NC} (original=$orig_exit translated=$trans_exit)"
        failed=1
    fi

    if [ "$orig_stdout" != "$trans_stdout" ]; then
        echo -e "  ${RED}FAIL: Stdout mismatch${NC}"
        echo "    Original:   '$orig_stdout'"
        echo "    Translated: '$trans_stdout'"
        failed=1
    fi

    # 7. MOV-purity check: verify output contains only mov/jmp/int/call instructions
    non_mov=$(grep -E '^\s+[a-z]' "$trans_source" \
        | grep -vE '^\s*(mov[blwq]?|movzbl|movsbl|movzwl|movswl|movzbw|movsbw|movzwq|movswq|jmp|int|call|ret)(\s|$)' \
        | grep -vE '^\s*#' \
        | grep -vE '^\s*\.' \
        | head -5 || true)

    if [ -n "$non_mov" ]; then
        echo -e "  ${RED}FAIL: Non-MOV instructions found in output:${NC}"
        echo "$non_mov" | sed 's/^/    /'
        failed=1
    fi

    if [ $failed -eq 0 ]; then
        orig_lines=$(wc -l < "$in_source")
        trans_lines=$(wc -l < "$trans_source")
        echo -e "  ${GREEN}[PASS]${NC} exit=$orig_exit  (${orig_lines} -> ${trans_lines} lines)"
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
    fi
done

echo -e "\n${BLUE}=======================================================${NC}"
echo -e "  ${GREEN}PASS: $pass${NC}  ${RED}FAIL: $fail${NC}  ${YELLOW}SKIP: $skip${NC}"
echo -e "${BLUE}=======================================================${NC}"

[ $fail -eq 0 ]
