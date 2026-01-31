.data
    v: .long 26, 12, 3, 56, 3, 18, 27, 35, 15
    n: .long 9
    frecv: .space 404   
    formatAfisare: .asciz "Numarul care apare cele mai multe ori: %d\n"

.text
.global main
.extern printf

main:
    movl n, %ecx
    lea v, %edi
    xorl %eax, %eax    

et_tablou_frecv:
    cmp $0, %ecx
    je et_parcurgere_prep
    
    movl (%edi, %eax, 4), %ebx  
    lea frecv, %esi
    movl (%esi, %ebx, 4), %edx  
    addl $1, %edx
    movl %edx, (%esi, %ebx, 4)   
    
    incl %eax
    decl %ecx
    jmp et_tablou_frecv

et_parcurgere_prep:
    movl $0, %eax       
    movl $0, %ebx      
    movl $0, %edx      
    lea frecv, %esi

et_cautare:
    cmp $100, %eax
    jg afisare
    
    movl (%esi, %eax, 4), %ecx  
    cmp %edx, %ecx
    jle et_cont_parcurgere
    
    movl %ecx, %edx     
    movl %eax, %ebx     

et_cont_parcurgere:
    incl %eax
    jmp et_cautare

afisare:    
    pushl %ebx
    pushl $formatAfisare
    call printf
    addl $8, %esp

et_exit:
    movl $1, %eax
    xorl %ebx, %ebx
    int $0x80