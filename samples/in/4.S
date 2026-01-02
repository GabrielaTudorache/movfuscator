.data
    n: .long 10
    t0: .long 0
    t1: .long 5
    t2: .long 2

.text
.global main

main:
    mov n, %ecx
    subl $3, %ecx

et_loop:
    cmpl $0, %ecx
    je et_exit
    
    mov t0, %eax
    add t1, %eax
    add t2, %eax     
    
    mov t1, %edx
    mov %edx, t0     
    mov t2, %edx
    mov %edx, t1    
    mov %eax, t2    
    
    loop et_loop

et_exit:
    mov $1, %eax
    xor %ebx, %ebx
    int $0x80