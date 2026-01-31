.data

.text
.global main

main:
    movl $0, %ecx
    movl $3, %eax       
    movl $5, %ebx       
    cmp %ebx, %eax      
    jge greater_or_equal 
    
    movl $1, %ecx
    jmp end

greater_or_equal:
    movl $2, %ecx

end:
    movl $1, %eax
    xor %ebx, %ebx
    int $0x80