.text
.global main

# Exercises memory writeback where address uses temp regs.
# Writes and updates arr[ecx] via (%eax,%ecx,4).

main:
    movl $arr, %eax
    movl $1, %ecx

    movl $5, (%eax, %ecx, 4)
    addl $2, (%eax, %ecx, 4)
    subl $1, (%eax, %ecx, 4)

    movl (%eax, %ecx, 4), %ebx
    movl $1, %eax
    int $0x80

.data
arr: .long 0, 0, 0, 0
