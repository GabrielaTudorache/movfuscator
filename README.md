# movfuscator

## Echipa

| Nume | Prenume | Grupa |
|------|---------|-------|
| Tudorache | Gabriela | 151 |
| Bejan | Monica | 151 |

## Descriere

**movfuscator** este un tool scris in Python care transforma cod assembly x86 pe 32 de biti (sintaxa AT&T) in cod semantic echivalent care foloseste **doar instructiuni MOV** (si variante precum `movl`, `movb`, `movzbl` etc.).

Proiectul se bazeaza pe demonstratia lui Stephen Dolan (2013) ca instructiunea MOV este Turing-completa: orice computatie poate fi exprimata prin citiri, cautari in tabele (lookup tables) si scrieri in memorie - toate operatii pe care MOV le poate realiza.

### Cum functioneaza

1. **Lexer** (`src/lexer.py`) - tokenizeaza codul assembly in sintaxa AT&T
2. **Parser** (`src/parser.py`) - construieste un AST (Abstract Syntax Tree) din tokeni
3. **Translator** (`src/translator/core.py`) - rescrie fiecare instructiune non-MOV intr-o secventa echivalenta de MOV-uri, folosind tabele de cautare (LUT) pre-calculate si memorie scratch
4. **OutputFormatter** (`src/translator/output.py`) - emite fisierul assembly final, incluzand doar tabelele LUT utilizate efectiv de program

Aritmetica se face byte cu byte (4 bytes pentru valori pe 32 de biti) cu propagare carry/borrow prin tabele de cautare 256x256. Salturile conditionale folosesc o tabela de dispatch cu 2 intrari si un `jmp *scratch_jcc_target` indirect. Fluxul de control foloseste in continuare `jmp` (iar `call`/`ret`/`int` sunt pasate neschimbate).

### Instructiuni suportate

| Categorie | Instructiuni |
|---|---|
| Mutare date | `mov`, `movl`, `movb`, `movw`, `movzbl`, `movsbl`, `movzwl`, `movswl`, `movzbw`, `movsbw`, `lea` |
| Aritmetica | `add`, `sub`, `inc`, `dec`, `mul`, `div` |
| Logica/shift | `xor`, `or`, `and`, `shl`, `sal`, `shr` |
| Comparatie | `cmp`, `test` |
| Flux de control | `jmp`, `je/jz`, `jne/jnz`, `jl`, `jle`, `jg`, `jge`, `jb`, `jbe`, `ja`, `jae`, `loop` |
| Stiva | `push`, `pop`, `leave` |
| Misc | `nop` |
| Passthrough | `int`, `call`, `ret` (nu pot fi inlocuite cu MOV) |

## Cod sursa, fisiere generate si fisiere in/out

### Cod sursa (Python)

```
src/
  main.py                CLI entry point
  lexer.py               Tokenizer pentru assembly AT&T
  parser.py              AST builder
  scratch.py             Layout-ul memoriei scratch (.data)
  lut_generator.py       Genereaza tabelele de cautare ca fisiere .s
  translator/
    core.py              Motorul de traducere (3700+ linii)
    output.py            Formatare output + includere selectiva LUT
    config.py            TranslatorConfig si TranslatorError
    __init__.py          API public translator
```

### Fisiere generate (lookup tables)

Directorul `lut/` contine 19 fisiere `.s` pre-generate de `lut_generator.py` (~6.9 MB total). Acestea sunt tabele de cautare 256x256 folosite pentru a implementa operatiile aritmetice si logice doar prin MOV:

```
lut/
  add_lut.s, sub_lut.s, carry_lut.s, borrow_lut.s       Aritmetica
  xor_lut.s, or_lut.s, and_lut.s                         Logica
  mul8_lo_lut.s, mul8_hi_lut.s                            Inmultire
  je_lut.s, jb_lut.s, jl_signed_lut.s                    Comparatie
  is_zero_lut.s, is_not_zero_lut.s                        Test zero
  shl1_lut.s, shl1_carry_lut.s, shr1_lut.s, shr1_carry_lut.s   Shift
  all_luts.s                                              Toate tabelele combinate
```

### Fisiere in/out (cod Assembly)

**Fisiere de intrare** - programe assembly x86-32 scrise manual, folosite ca teste:

```
samples/in/           8 programe de test principale
  2.S                 Factorial (10!) cu MUL + LOOP
  3.S                 Fibonacci cu ADD + LOOP
  4.S                 Tribonacci cu ADD + CMP + JE + LOOP
  5.S                 Cel mai lung sir de 1-bits cu TEST/SHR/INC/DEC
  6.S                 Urmatoarea putere a lui 2 cu CMP/JA/SHL
  7.S                 Suma patratelor cu MUL/ADD/DEC/CMP/JE
  8.S                 Max din array + numarare aparitii cu CMP/JG/JE/LOOP
  9.S                 Suma cifrelor cu DIV/ADD/CMP/JE + printf/fflush

samples/in_extra/     13 programe de test extinse
  01-13               Teste pentru: adresare memorie, LEA, push/pop,
                      salturi signed/unsigned, shift-uri, sume array,
                      factorial, GCD, operatii logice, indexare, stiva,
                      call/ret, instructiunea LOOP
```

**Fisiere de iesire** — generate de tool cu sufixul `.mov.s` (ex: `input.s` -> `input.mov.s`). Programele din `samples/in/` au output-urile pre-generate in `samples/out/`. Scriptul de test genereaza output-uri temporare separat.

```
samples/out/          Output-uri pre-generate (.mov.s) pentru samples/in/
```

## Utilizare

### Traducere

```bash
python src/main.py input.s -o output.s
```

### Compilare si rulare

```bash
gcc -m32 output.s -o program -no-pie
./program
```

### Optiuni CLI

```
python src/main.py INPUT [-o OUTPUT] [--no-luts] [--no-scratch] [--lut-path PATH]
```

| Optiune | Descriere |
|---|---|
| `INPUT` | Fisier assembly de intrare (obligatoriu) |
| `-o OUTPUT` | Fisier de iesire (implicit: `INPUT.mov.s`) |
| `--no-luts` | Omite datele LUT din output (pentru linking separat) |
| `--no-scratch` | Omite declaratiile de memorie scratch |
| `--lut-path PATH` | Calea catre directorul LUT (implicit: `lut/`) |

Output-ul implicit este un fisier assembly autonom: memorie scratch, tabele de cautare si codul tradus sunt toate incluse. Doar tabelele LUT efectiv referentiate de program sunt emise.

### Testare

```bash
bash test.sh
```

Scriptul de test traduce fiecare program sample, compileaza atat originalul cat si versiunea tradusa cu `gcc -m32`, ruleaza ambele (cu timeout de 10 secunde) si compara exit code-urile si stdout-ul. De asemenea, verifica puritatea MOV - output-ul nu contine decat instructiuni MOV/JMP/INT/CALL/RET.

## Ce merge si ce nu merge

### Ce merge

- Pipeline-ul complet de traducere: lexer -> parser -> translator -> output formatter
- Toate instructiunile de baza pe 32 de biti: ADD, SUB, INC, DEC, MUL, DIV, XOR, OR, AND, SHL, SAL, SHR, CMP, TEST
- Toate variantele MOV trec neschimbate (movl, movb, movw, movzbl, movsbl etc.)
- Salturi conditionale: JE/JZ, JNE/JNZ, JL, JLE, JG, JGE, JB, JBE, JA, JAE si toate alias-urile
- Instructiunea LOOP (decrementeaza ECX, salt conditionat)
- PUSH si POP (traduce aritmetica ESP prin LUT-uri SUB/ADD)
- LEA cu toate formele: displacement, base, index*scale si combinatii
- LEAVE (mov %ebp, %esp + pop %ebp)
- NOP (eliminat)
- INT, CALL, RET trec neschimbate
- Optimizare self-XOR: `xorl %reg, %reg` -> `movl $0, %reg`
- DIV cu semantica completa x86-32: dividend pe 64 de biti (EDX:EAX), detectie overflow cand EDX >= divisor, si impartire la zero - ambele genereaza `INT $0` (identic cu exceptia x86 #DE)
- Includere inteligenta a LUT-urilor: doar tabelele referentiate sunt incluse in output
- Toate cele 21 de programe sample trec testele de integrare

### Ce nu merge / Limitari

- **Doar 32 de biti.** Variantele pe 8-bit (`b`), 16-bit (`w`) si 64-bit (`q`) sunt respinse explicit cu eroare.
- **PUSH/POP cu operanzi `%esp`-relativi** (ex: `pushl 4(%esp)`) pot produce rezultate incorecte deoarece ESP se modifica in timpul operatiei.
- **Salturile conditionale** folosesc rezultate de comparatie din memorie scratch, nu flag-urile CPU. Instructiunile care nu seteaza flag-uri pe x86 real (LEA, PUSH, POP) nu actualizeaza starea de comparatie scratch.
- **SHL/SHR necesita contor imediat** (nu sunt suportate shift-uri cu registru, ex: `shll %cl, %eax`).
- **Instructiuni nesuportate:** NEG, NOT, IMUL, IDIV, SAR, ROL, ROR (prezente in parser dar fara handler in translator).
- **Fara virgula mobila.** Niciun suport SSE sau x87.

## Referinte

1. Stephen Dolan, *"mov is Turing-complete"*, Computer Laboratory, University of Cambridge, 2013. Disponibil la: [https://drwho.virtadpt.net/files/mov.pdf](https://drwho.virtadpt.net/files/mov.pdf)
   - Demonstratia teoretica pe care se bazeaza acest proiect. Dolan arata ca instructiunea x86 MOV este Turing-completa: orice computatie poate fi exprimata exclusiv prin citiri si scrieri in memorie, folosind tabele de cautare (lookup tables).
