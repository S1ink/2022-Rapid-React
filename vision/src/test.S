.global test_add6_asm
.global test_c3_add_asm


test_add6_asm:
	push {r4, r5, fp}
	add fp, sp, #0

	ldr r4, [sp, #(8+8)]
	ldr r5, [sp, #(4+8)]

	add r0, r0, r1
	add r0, r0, r2
	add r0, r0, r3
	add r0, r0, r4
	add r0, r0, r5

	sub sp, fp, #0
	pop {r4, r5, fp}
	bx lr