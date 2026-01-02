.data
    x: .long 77

.text
.global main

main:
    movl x, %eax
    movl $1, %ebx

et_loop:
    cmpl %eax, %ebx
    ja et_exit          
    shll $1, %ebx       
    jmp et_loop

et_exit:
    movl $1, %eax
    xorl %ebx, %ebx
    int $0x80