.text
.global main

# Uses a local call/ret to exercise passthrough control flow.

main:
    movl $5, %eax
    call add_two
    movl %eax, %ebx
    movl $1, %eax
    int $0x80

add_two:
    addl $2, %eax
    ret
