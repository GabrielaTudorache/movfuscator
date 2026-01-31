.text
.global main

# Sum an integer array using indexed addressing.

main:
    movl $arr, %edi
    movl len, %ecx
    xorl %eax, %eax
    xorl %ebx, %ebx

loop_sum:
    cmpl $0, %ecx
    je done

    movl len, %edx
    subl %ecx, %edx
    movl (%edi, %edx, 4), %eax
    addl %eax, %ebx

    decl %ecx
    jmp loop_sum

done:
    movl $1, %eax
    int $0x80

.data
len: .long 8
arr: .long 3, 1, 4, 1, 5, 9, 2, 6
