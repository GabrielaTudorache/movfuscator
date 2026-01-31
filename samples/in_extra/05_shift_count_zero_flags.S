.text
.global main

# Shift by 0 should not change scratch-based flags.

main:
    movl $0, %eax
    cmpl $0, %eax

    shll $0, %eax
    jne fail

    shrl $0, %eax
    jne fail

    movl $0, %ebx
    jmp exit

fail:
    movl $1, %ebx

exit:
    movl $1, %eax
    int $0x80
