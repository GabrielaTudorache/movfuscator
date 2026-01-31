.text
.global main

# Uses LOOP instruction to sum 1..N.

main:
    movl $10, %ecx
    xorl %ebx, %ebx

sum_loop:
    addl %ecx, %ebx
    loop sum_loop

    movl $1, %eax
    int $0x80
