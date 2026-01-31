.text
.global main

# Mix of AND/OR/XOR/INC/DEC on registers and memory.

main:
    movl $0x12345678, %eax
    movl $0x0f0f0f0f, %ebx

    andl %ebx, %eax
    orl $0x00ff00ff, %eax
    xorl $0x11111111, %eax

    movl %eax, tmp
    incl tmp
    decl tmp
    xorl tmp, %eax

    movl %eax, %ebx
    movl $1, %eax
    int $0x80

.data
tmp: .long 0
