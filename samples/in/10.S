.data
    v: .long 26, 12, 3, 56, 3, 18, 27, 35, 15
    n: .long 9
    max1: .long 0
    max2: .long 0
    formatAf: .asciz "%ld\n"

.text
.global main

main:
    movl n, %ecx
    lea v, %edi
    xorl %eax, %eax    

et_parcurgere:
    cmp $0, %ecx
    je et_afisare
    
    movl (%edi, %eax, 4), %edx  
    
    movl max1, %ebx
    cmp %ebx, %edx
    jle verifica_max2
    
    movl %ebx, max2     
    movl %edx, max1      
    jmp et_cont_parcurgere

verifica_max2:
    movl max2, %ebx
    cmp %ebx, %edx
    jle et_cont_parcurgere
    
    movl max1, %esi
    cmp %esi, %edx
    jge et_cont_parcurgere 

    movl %edx, max2

et_cont_parcurgere:
    incl %eax
    decl %ecx
    jmp et_parcurgere

et_afisare:
    movl max2, %edx      
    
    pushl max2
    pushl $formatAf
    call printf
    popl %ebx
    popl %ebx     

et_exit:
    movl $1, %eax
    xorl %ebx, %ebx
    int $0x80