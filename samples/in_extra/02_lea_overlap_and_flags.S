.text
.global main

# Verifies LEA does not clobber comparison state and handles dst overlap.

main:
    movl $0, %eax
    cmpl $0, %eax

    movl $arr, %ecx
    lea 4(%ecx, %eax, 4), %ecx

    jne fail
    movl $0, %ebx
    jmp exit

fail:
    movl $1, %ebx

exit:
    movl $1, %eax
    int $0x80

.data
arr: .long 0, 0
