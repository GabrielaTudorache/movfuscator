.text
.global main

# Exercises signed/unsigned comparisons on boundary values.

main:
    # Signed: -1 < 1 should be true
    movl $-1, %eax
    movl $1, %ebx
    cmpl %ebx, %eax
    jl signed_ok
    movl $10, %ebx
    jmp exit

signed_ok:
    # Unsigned: 0xFFFFFFFF (4294967295) < 1 should be false
    movl $-1, %eax
    movl $1, %ebx
    cmpl %ebx, %eax
    jb fail

    movl $0, %ebx
    jmp exit

fail:
    movl $11, %ebx

exit:
    movl $1, %eax
    int $0x80
