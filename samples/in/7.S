.data
    n: .long 6
    s: .long 0

.text
.global main

main:
    movl n, %ecx
    xorl %eax, %eax
    xorl %ebx, %ebx    

et_loop:
    cmpl $0, %ecx
    je et_exit
    
    movl %ecx, %eax
    mull %eax           
    addl %eax, %ebx     
    
    decl %ecx
    jmp et_loop

et_exit:
    movl %ebx, s
    movl $1, %eax
    xorl %ebx, %ebx
    int $0x80