.text
.global main

# Uses LEA to compute an element address and then loads/stores through it.

main:
    movl $arr, %esi
    movl $2, %ecx

    lea (%esi, %ecx, 4), %edi
    movl (%edi), %eax
    addl $10, %eax
    movl %eax, (%edi)

    movl (%esi, %ecx, 4), %ebx
    movl $1, %eax
    int $0x80

.data
arr: .long 1, 2, 3, 4
