.text
.global main
main:
    # Validate full x86 divl semantics for EDX:EAX / r/m32.
    # dividend = 0x00000001_00000002, divisor = 2
    # expected: EAX = 0x80000001, EDX = 0

    movl $0x2, %eax
    movl $0x00000001, %edx
    movl $2, %ebx
    divl %ebx

    cmpl $0x80000001, %eax
    jne fail
    cmpl $0, %edx
    jne fail

    movl $0, %ebx
    jmp exit

fail:
    movl $1, %ebx

exit:
    movl $1, %eax
    int $0x80
