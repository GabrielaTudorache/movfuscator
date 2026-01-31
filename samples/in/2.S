.data
    n: .long 10

.text
.global main

main:
    mov n, %ecx
    mov $1, %eax

et_loop:
    mul %ecx       
    loop et_loop    

exit:
    mov $1, %eax
    mov $0, %ebx
    int $0x80