.text
.global main

# Verifies pushl %esp stores the original ESP (x86 semantics)
# and that push/pop do not clobber comparison state.

main:
    movl %esp, %eax
    cmpl %esp, %eax

    pushl %esp
    popl %ebx

    subl %eax, %ebx

    jne fail
    movl $0, %ebx
    jmp exit

fail:
    movl $1, %ebx

exit:
    movl $1, %eax
    int $0x80
