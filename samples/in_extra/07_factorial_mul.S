.text
.global main

# Factorial via MUL in a loop.

main:
    movl $6, %ecx
    movl $1, %eax

fac_loop:
    cmpl $0, %ecx
    je fac_done
    mul %ecx
    decl %ecx
    jmp fac_loop

fac_done:
    movl %eax, %ebx
    movl $1, %eax
    int $0x80
