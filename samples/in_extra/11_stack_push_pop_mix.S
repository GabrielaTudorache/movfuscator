.text
.global main
.extern printf

# Stack usage with push/pop and a libc call.

main:
    movl $0, %ebx
    pushl $msg
    call printf
    addl $4, %esp

    movl $7, %eax
    pushl %eax
    popl %ebx

    movl $1, %eax
    int $0x80

.data
msg: .asciz "extra sample\n"
