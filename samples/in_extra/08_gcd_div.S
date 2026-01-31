.text
.global main

# Euclidean GCD using DIV (unsigned), with explicit edx = 0.

main:
    movl $1071, %eax
    movl $462, %ebx

gcd_loop:
    cmpl $0, %ebx
    je gcd_done

    xorl %edx, %edx
    divl %ebx

    movl %ebx, %eax
    movl %edx, %ebx
    jmp gcd_loop

gcd_done:
    movl %eax, %ebx
    movl $1, %eax
    int $0x80
