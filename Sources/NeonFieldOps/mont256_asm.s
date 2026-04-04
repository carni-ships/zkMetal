// ARM64 assembly: 4-limb CIOS Montgomery multiplication for BN254 Fr
//
// Two entry points:
//   void mont_mul_asm(uint64_t result[4], const uint64_t a[4], const uint64_t b[4])
//   void mont_mul_batch_asm(uint64_t *data, const uint64_t *multiplier, int n)
//     — data[i*4..i*4+3] *= multiplier for i in 0..n-1
//
// BN254 Fr modulus p = [0x43e1f593f0000001, 0x2833e84879b97091,
//                        0xb85045b68181585d, 0x30644e72e131a029]
// inv = -p^{-1} mod 2^64 = 0xc2e1f593efffffff

// ================================================================
// Macro: CIOS_ITER  ai_reg, accumulates a[i]*b into t[0..4]
//   then does Montgomery reduction (m*p + shift).
//   Uses: x7-x10 = b[0..3], x19-x22 = p[0..3], x23 = inv
//   Modifies: x11-x15 = t[0..4], x16 = m, x24-x26 = temp
// ================================================================

.macro CIOS_ITER ai
    // Multiply-accumulate: t += ai * b
    mul     x24, \ai, x7
    umulh   x25, \ai, x7
    adds    x11, x11, x24
    adc     x25, x25, xzr

    mul     x24, \ai, x8
    umulh   x26, \ai, x8
    adds    x24, x24, x25
    adc     x26, x26, xzr
    adds    x12, x12, x24
    adc     x25, x26, xzr

    mul     x24, \ai, x9
    umulh   x26, \ai, x9
    adds    x24, x24, x25
    adc     x26, x26, xzr
    adds    x13, x13, x24
    adc     x25, x26, xzr

    mul     x24, \ai, x10
    umulh   x26, \ai, x10
    adds    x24, x24, x25
    adc     x26, x26, xzr
    adds    x14, x14, x24
    adc     x26, x26, xzr
    add     x15, x15, x26

    // Montgomery reduce: m = t0 * inv; t = (t + m*p) >> 64
    mul     x16, x11, x23

    mul     x24, x16, x19
    umulh   x25, x16, x19
    adds    x24, x24, x11          // low word cancels to 0
    adc     x25, x25, xzr

    mul     x24, x16, x20
    umulh   x26, x16, x20
    adds    x24, x24, x25
    adc     x26, x26, xzr
    adds    x11, x24, x12          // new t0
    adc     x25, x26, xzr

    mul     x24, x16, x21
    umulh   x26, x16, x21
    adds    x24, x24, x25
    adc     x26, x26, xzr
    adds    x12, x24, x13          // new t1
    adc     x25, x26, xzr

    mul     x24, x16, x22
    umulh   x26, x16, x22
    adds    x24, x24, x25
    adc     x26, x26, xzr
    adds    x13, x24, x14          // new t2
    adc     x26, x26, xzr
    add     x14, x15, x26          // new t3
    mov     x15, xzr
.endm

// Macro: FINAL_SUB — conditional subtraction if t >= p
// Note: uses x24-x26 and x0 as temps (avoids clobbering x27 which is used as loop counter)
.macro FINAL_SUB
    subs    x24, x11, x19
    sbcs    x25, x12, x20
    sbcs    x26, x13, x21
    sbcs    x0, x14, x22
    csel    x11, x24, x11, cs
    csel    x12, x25, x12, cs
    csel    x13, x26, x13, cs
    csel    x14, x0, x14, cs
.endm

// Macro: LOAD_CONSTANTS — load p[0..3] and inv into x19-x23
.macro LOAD_CONSTANTS
    movz    x19, #0x0001
    movk    x19, #0xf000, lsl #16
    movk    x19, #0xf593, lsl #32
    movk    x19, #0x43e1, lsl #48

    movz    x20, #0x7091
    movk    x20, #0x79b9, lsl #16
    movk    x20, #0xe848, lsl #32
    movk    x20, #0x2833, lsl #48

    movz    x21, #0x585d
    movk    x21, #0x8181, lsl #16
    movk    x21, #0x45b6, lsl #32
    movk    x21, #0xb850, lsl #48

    movz    x22, #0xa029
    movk    x22, #0xe131, lsl #16
    movk    x22, #0x4e72, lsl #32
    movk    x22, #0x3064, lsl #48

    movz    x23, #0xffff
    movk    x23, #0xefff, lsl #16
    movk    x23, #0xf593, lsl #32
    movk    x23, #0xc2e1, lsl #48
.endm

// ================================================================
// Single multiplication: mont_mul_asm(result, a, b)
// ================================================================
.globl _mont_mul_asm
.p2align 4
_mont_mul_asm:
    stp     x19, x20, [sp, #-96]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #48]
    stp     x27, x28, [sp, #64]
    str     x29, [sp, #80]

    mov     x28, x0                // save result pointer

    ldp     x3, x4, [x1]
    ldp     x5, x6, [x1, #16]
    ldp     x7, x8, [x2]
    ldp     x9, x10, [x2, #16]

    LOAD_CONSTANTS

    // Init accumulator
    mov     x11, xzr
    mov     x12, xzr
    mov     x13, xzr
    mov     x14, xzr
    mov     x15, xzr

    CIOS_ITER x3
    CIOS_ITER x4
    CIOS_ITER x5
    CIOS_ITER x6
    FINAL_SUB

    stp     x11, x12, [x28]
    stp     x13, x14, [x28, #16]

    ldr     x29, [sp, #80]
    ldp     x27, x28, [sp, #64]
    ldp     x25, x26, [sp, #48]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #96
    ret

// ================================================================
// Batch multiplication: mont_mul_batch_asm(data, multiplier, n)
//   data[i] *= multiplier for i in 0..n-1
//   x0 = data pointer (array of n elements, each 4 uint64_t)
//   x1 = multiplier pointer (4 uint64_t)
//   x2 = n (count)
// ================================================================
.globl _mont_mul_batch_asm
.p2align 4
_mont_mul_batch_asm:
    cbz     x2, Lbatch_done

    stp     x19, x20, [sp, #-96]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #48]
    stp     x27, x28, [sp, #64]
    stp     x29, x30, [sp, #80]

    mov     x28, x0                // data pointer
    mov     x29, x2                // count

    // Load multiplier (stays in registers for all iterations)
    ldp     x7, x8, [x1]
    ldp     x9, x10, [x1, #16]

    LOAD_CONSTANTS

Lbatch_loop:
    // Load data[i]
    ldp     x3, x4, [x28]
    ldp     x5, x6, [x28, #16]

    // Init accumulator
    mov     x11, xzr
    mov     x12, xzr
    mov     x13, xzr
    mov     x14, xzr
    mov     x15, xzr

    CIOS_ITER x3
    CIOS_ITER x4
    CIOS_ITER x5
    CIOS_ITER x6
    FINAL_SUB

    // Store result back
    stp     x11, x12, [x28]
    stp     x13, x14, [x28, #16]

    add     x28, x28, #32          // advance to next element
    subs    x29, x29, #1
    b.ne    Lbatch_loop

    ldp     x29, x30, [sp, #80]
    ldp     x27, x28, [sp, #64]
    ldp     x25, x26, [sp, #48]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #96

Lbatch_done:
    ret

// ================================================================
// Batch a*b: mont_mul_pair_batch_asm(result, a, b, n)
//   result[i] = a[i] * b[i] for i in 0..n-1
//   x0 = result pointer
//   x1 = a pointer
//   x2 = b pointer
//   x3 = n
// ================================================================
.globl _mont_mul_pair_batch_asm
.p2align 4
_mont_mul_pair_batch_asm:
    cbz     x3, Lpair_done

    stp     x19, x20, [sp, #-96]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #48]
    stp     x27, x28, [sp, #64]
    stp     x29, x30, [sp, #80]

    mov     x28, x0                // result pointer
    mov     x29, x1                // a pointer
    mov     x30, x2                // b pointer
    mov     x27, x3                // count

    LOAD_CONSTANTS

Lpair_loop:
    // Load a[i] into x3-x6
    ldp     x3, x4, [x29]
    ldp     x5, x6, [x29, #16]

    // Load b[i] into x7-x10
    ldp     x7, x8, [x30]
    ldp     x9, x10, [x30, #16]

    mov     x11, xzr
    mov     x12, xzr
    mov     x13, xzr
    mov     x14, xzr
    mov     x15, xzr

    CIOS_ITER x3
    CIOS_ITER x4
    CIOS_ITER x5
    CIOS_ITER x6
    FINAL_SUB

    stp     x11, x12, [x28]
    stp     x13, x14, [x28, #16]

    add     x28, x28, #32
    add     x29, x29, #32
    add     x30, x30, #32
    subs    x27, x27, #1
    b.ne    Lpair_loop

    ldp     x29, x30, [sp, #80]
    ldp     x27, x28, [sp, #64]
    ldp     x25, x26, [sp, #48]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #96

Lpair_done:
    ret
