.data
    n: .long 10
    t1: .long 0
    t2: .long 1

.text
.global main

main:
    mov n, %ecx
    subl $2, %ecx

et_loop:
    movl t2, %eax
    movl t1, %ebx
    addl t2, %ebx
    movl %ebx, t2   
    movl %eax, t1  
      
    loop et_loop

et_exit:
    mov $1, %eax
    xor %ebx, %ebx
    int $0x80