.data
    n: .long 9
    v: .long 12, 15, 5, 15, 4, 1, 7, 15, 1
    maxim: .long 0
    ap: .long 0

.text
.global main

main:
    movl n, %ecx
    movl $v, %edi

et_loop:
    movl n, %ebx
    subl %ecx, %ebx         
    movl (%edi, %ebx, 4), %edx
    cmp maxim, %edx
    jg et_max_nou
    loop et_loop
    jmp cont_max

et_max_nou:
    movl %edx, maxim
    loop et_loop

cont_max:
    movl n, %ecx
    movl $v, %edi

loop_ap:
    movl n, %ebx
    subl %ecx, %ebx
    movl (%edi, %ebx, 4), %edx
    cmp maxim, %edx
    je et_egale
    loop loop_ap
    jmp et_exit

et_egale:
    incl ap
    loop loop_ap

et_exit:
    movl ap, %edx   
    movl $1, %eax
    xorl %ebx, %ebx
    int $0x80