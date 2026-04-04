// Keccak-256 GPU kernel — batch hashing
// Each thread computes one Keccak-256 hash of a fixed-size input.
// Implements the Keccak-f[1600] permutation with 24 rounds.
//
// Optimization: bit-interleaved representation.
// Each 64-bit lane is split into even bits and odd bits, each in a 32-bit word.
// This halves register width to 32 bits (native on Apple GPU) and simplifies rotations.

#include <metal_stdlib>
using namespace metal;

// Bit-interleaved round constants (even bits, odd bits)
// Pre-split from the standard 64-bit round constants.
constant uint2 KECCAK_RC_IL[24] = {
    uint2(0x00000001u, 0x00000000u), // 0x0000000000000001
    uint2(0x00000000u, 0x00000089u), // 0x0000000000008082
    uint2(0x00000000u, 0x8000008bu), // 0x800000000000808a
    uint2(0x00000000u, 0x80008080u), // 0x8000000080008000
    uint2(0x00000001u, 0x0000008bu), // 0x000000000000808b
    uint2(0x00000001u, 0x00008000u), // 0x0000000080000001
    uint2(0x00000001u, 0x80008088u), // 0x8000000080008081
    uint2(0x00000001u, 0x80000082u), // 0x8000000000008009
    uint2(0x00000000u, 0x0000000bu), // 0x000000000000008a
    uint2(0x00000000u, 0x0000000au), // 0x0000000000000088
    uint2(0x00000001u, 0x00008082u), // 0x0000000080008009
    uint2(0x00000000u, 0x00008003u), // 0x000000008000000a
    uint2(0x00000001u, 0x0000808bu), // 0x000000008000808b
    uint2(0x00000001u, 0x8000000bu), // 0x800000000000008b
    uint2(0x00000001u, 0x8000008au), // 0x8000000000008089
    uint2(0x00000001u, 0x80000081u), // 0x8000000000008003
    uint2(0x00000000u, 0x80000081u), // 0x8000000000008002
    uint2(0x00000000u, 0x80000008u), // 0x8000000000000080
    uint2(0x00000000u, 0x00000083u), // 0x000000000000800a
    uint2(0x00000000u, 0x80008003u), // 0x800000008000000a
    uint2(0x00000001u, 0x80008088u), // 0x8000000080008081
    uint2(0x00000000u, 0x80000088u), // 0x8000000000008080
    uint2(0x00000001u, 0x00008000u), // 0x0000000080000001
    uint2(0x00000000u, 0x80008082u), // 0x8000000080008008
};

// Standard round constants (kept for the non-interleaved path used by Merkle)
constant ulong KECCAK_RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808aUL, 0x8000000080008000UL,
    0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008aUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL,
    0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL,
    0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL,
};

// --- Bit-interleaving helpers (O(log n) bit-manipulation, no loops) ---
// Separate even and odd bits of a 32-bit word into two 16-bit halves packed in low bits.
// Returns (even_bits_in_low16, odd_bits_in_low16) packed as: even in low 16, odd in bits 16-31.
inline uint deinterleave_uint(uint x) {
    // Extract even bits (0,2,4,...,30) into contiguous low 16 bits
    uint e = x & 0x55555555u;       // keep even bits
    e = (e | (e >> 1)) & 0x33333333u;
    e = (e | (e >> 2)) & 0x0f0f0f0fu;
    e = (e | (e >> 4)) & 0x00ff00ffu;
    e = (e | (e >> 8)) & 0x0000ffffu;
    // Extract odd bits (1,3,5,...,31)
    uint o = (x >> 1) & 0x55555555u;
    o = (o | (o >> 1)) & 0x33333333u;
    o = (o | (o >> 2)) & 0x0f0f0f0fu;
    o = (o | (o >> 4)) & 0x00ff00ffu;
    o = (o | (o >> 8)) & 0x0000ffffu;
    return e | (o << 16);
}

// Interleave two 16-bit values (in low 16 bits each) into even/odd bits of a 32-bit word
inline uint interleave_uint(uint even16, uint odd16) {
    uint e = even16 & 0x0000ffffu;
    e = (e | (e << 8)) & 0x00ff00ffu;
    e = (e | (e << 4)) & 0x0f0f0f0fu;
    e = (e | (e << 2)) & 0x33333333u;
    e = (e | (e << 1)) & 0x55555555u;
    uint o = odd16 & 0x0000ffffu;
    o = (o | (o << 8)) & 0x00ff00ffu;
    o = (o | (o << 4)) & 0x0f0f0f0fu;
    o = (o | (o << 2)) & 0x33333333u;
    o = (o | (o << 1)) & 0x55555555u;
    return e | (o << 1);
}

// Convert standard 64-bit to interleaved (even bits, odd bits) representation
inline uint2 to_interleaved(ulong x) {
    uint lo = (uint)x;
    uint hi = (uint)(x >> 32);
    uint dlo = deinterleave_uint(lo); // low16=even_lo, high16=odd_lo
    uint dhi = deinterleave_uint(hi); // low16=even_hi, high16=odd_hi
    uint even = (dlo & 0xffffu) | ((dhi & 0xffffu) << 16);
    uint odd  = (dlo >> 16)     | ((dhi >> 16) << 16);
    return uint2(even, odd);
}

// Convert interleaved back to standard 64-bit
inline ulong from_interleaved(uint2 v) {
    uint even = v.x, odd = v.y;
    uint lo = interleave_uint(even & 0xffffu, odd & 0xffffu);
    uint hi = interleave_uint(even >> 16, odd >> 16);
    return ((ulong)hi << 32) | (ulong)lo;
}

// Rotation of interleaved lane by n bits — macro for guaranteed compile-time constant folding.
// Even n: rotate both halves by n/2
// Odd n:  swap halves + rotate by n/2 and (n+1)/2
#define ROTL_IL_EVEN(v, h) uint2(((v).x << (h)) | ((v).x >> (32-(h))), ((v).y << (h)) | ((v).y >> (32-(h))))
#define ROTL_IL_ODD(v, h)  uint2(((v).y << ((h)+1)) | ((v).y >> (32-((h)+1))), ((v).x << (h)) | ((v).x >> (32-(h))))
// Special cases for edge values
#define ROTL_IL_1(v)  uint2(((v).y << 1) | ((v).y >> 31), (v).x)  // n=1: h=0, h+1=1
#define ROTL_IL_63(v) uint2((v).y, ((v).x << 31) | ((v).x >> 1))  // n=63: h=31, h+1=32=identity

// Dispatch macro: selects even/odd path at compile time
#define ROTL_IL(v, n) ( \
    (n) == 0  ? (v) : \
    (n) == 1  ? ROTL_IL_1(v) : \
    (n) == 63 ? ROTL_IL_63(v) : \
    ((n) & 1) ? ROTL_IL_ODD(v, (n)>>1) : \
                ROTL_IL_EVEN(v, (n)>>1) \
)

// Keccak-f[1600] in bit-interleaved form: state is 25 x uint2
// Uses named scalar variables to help the register allocator avoid array spills.
void keccak_f1600_il(thread uint2 state[25]) {
    // Unpack into named locals — helps register allocator
    uint2 s00=state[0],  s01=state[1],  s02=state[2],  s03=state[3],  s04=state[4];
    uint2 s05=state[5],  s06=state[6],  s07=state[7],  s08=state[8],  s09=state[9];
    uint2 s10=state[10], s11=state[11], s12=state[12], s13=state[13], s14=state[14];
    uint2 s15=state[15], s16=state[16], s17=state[17], s18=state[18], s19=state[19];
    uint2 s20=state[20], s21=state[21], s22=state[22], s23=state[23], s24=state[24];

    for (uint round = 0; round < 24; round++) {
        // Theta
        uint2 C0 = s00 ^ s05 ^ s10 ^ s15 ^ s20;
        uint2 C1 = s01 ^ s06 ^ s11 ^ s16 ^ s21;
        uint2 C2 = s02 ^ s07 ^ s12 ^ s17 ^ s22;
        uint2 C3 = s03 ^ s08 ^ s13 ^ s18 ^ s23;
        uint2 C4 = s04 ^ s09 ^ s14 ^ s19 ^ s24;

        uint2 D0 = C4 ^ ROTL_IL(C1, 1);
        uint2 D1 = C0 ^ ROTL_IL(C2, 1);
        uint2 D2 = C1 ^ ROTL_IL(C3, 1);
        uint2 D3 = C2 ^ ROTL_IL(C4, 1);
        uint2 D4 = C3 ^ ROTL_IL(C0, 1);

        s00 ^= D0; s05 ^= D0; s10 ^= D0; s15 ^= D0; s20 ^= D0;
        s01 ^= D1; s06 ^= D1; s11 ^= D1; s16 ^= D1; s21 ^= D1;
        s02 ^= D2; s07 ^= D2; s12 ^= D2; s17 ^= D2; s22 ^= D2;
        s03 ^= D3; s08 ^= D3; s13 ^= D3; s18 ^= D3; s23 ^= D3;
        s04 ^= D4; s09 ^= D4; s14 ^= D4; s19 ^= D4; s24 ^= D4;

        // Rho + Pi in-place using Pi's 24-cycle
        uint2 temp = s01;
        s01 = ROTL_IL(s06, 44); s06 = ROTL_IL(s09, 20); s09 = ROTL_IL(s22, 61);
        s22 = ROTL_IL(s14, 39); s14 = ROTL_IL(s20, 18); s20 = ROTL_IL(s02, 62);
        s02 = ROTL_IL(s12, 43); s12 = ROTL_IL(s13, 25); s13 = ROTL_IL(s19,  8);
        s19 = ROTL_IL(s23, 56); s23 = ROTL_IL(s15, 41); s15 = ROTL_IL(s04, 27);
        s04 = ROTL_IL(s24, 14); s24 = ROTL_IL(s21,  2); s21 = ROTL_IL(s08, 55);
        s08 = ROTL_IL(s16, 45); s16 = ROTL_IL(s05, 36); s05 = ROTL_IL(s03, 28);
        s03 = ROTL_IL(s18, 21); s18 = ROTL_IL(s17, 15); s17 = ROTL_IL(s11, 10);
        s11 = ROTL_IL(s07,  6); s07 = ROTL_IL(s10,  3); s10 = ROTL_IL(temp,  1);

        // Chi — row 0
        { uint2 t0=s00,t1=s01,t2=s02,t3=s03,t4=s04;
          s00=t0^(~t1&t2); s01=t1^(~t2&t3); s02=t2^(~t3&t4); s03=t3^(~t4&t0); s04=t4^(~t0&t1); }
        // Chi — row 1
        { uint2 t0=s05,t1=s06,t2=s07,t3=s08,t4=s09;
          s05=t0^(~t1&t2); s06=t1^(~t2&t3); s07=t2^(~t3&t4); s08=t3^(~t4&t0); s09=t4^(~t0&t1); }
        // Chi — row 2
        { uint2 t0=s10,t1=s11,t2=s12,t3=s13,t4=s14;
          s10=t0^(~t1&t2); s11=t1^(~t2&t3); s12=t2^(~t3&t4); s13=t3^(~t4&t0); s14=t4^(~t0&t1); }
        // Chi — row 3
        { uint2 t0=s15,t1=s16,t2=s17,t3=s18,t4=s19;
          s15=t0^(~t1&t2); s16=t1^(~t2&t3); s17=t2^(~t3&t4); s18=t3^(~t4&t0); s19=t4^(~t0&t1); }
        // Chi — row 4
        { uint2 t0=s20,t1=s21,t2=s22,t3=s23,t4=s24;
          s20=t0^(~t1&t2); s21=t1^(~t2&t3); s22=t2^(~t3&t4); s23=t3^(~t4&t0); s24=t4^(~t0&t1); }

        // Iota
        s00 ^= KECCAK_RC_IL[round];
    }

    // Pack back to array
    state[0]=s00; state[1]=s01; state[2]=s02; state[3]=s03; state[4]=s04;
    state[5]=s05; state[6]=s06; state[7]=s07; state[8]=s08; state[9]=s09;
    state[10]=s10; state[11]=s11; state[12]=s12; state[13]=s13; state[14]=s14;
    state[15]=s15; state[16]=s16; state[17]=s17; state[18]=s18; state[19]=s19;
    state[20]=s20; state[21]=s21; state[22]=s22; state[23]=s23; state[24]=s24;
}

ulong rotl64(ulong x, uint n) {
    return (x << n) | (x >> (64 - n));
}

// Standard (non-interleaved) Keccak-f[1600] — used by Merkle fused kernel
void keccak_f1600(thread ulong state[25]) {
    for (uint round = 0; round < 24; round++) {
        // Theta
        ulong C0 = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
        ulong C1 = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
        ulong C2 = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
        ulong C3 = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
        ulong C4 = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];

        ulong D0 = C4 ^ rotl64(C1, 1);
        ulong D1 = C0 ^ rotl64(C2, 1);
        ulong D2 = C1 ^ rotl64(C3, 1);
        ulong D3 = C2 ^ rotl64(C4, 1);
        ulong D4 = C3 ^ rotl64(C0, 1);

        state[0]  ^= D0; state[5]  ^= D0; state[10] ^= D0; state[15] ^= D0; state[20] ^= D0;
        state[1]  ^= D1; state[6]  ^= D1; state[11] ^= D1; state[16] ^= D1; state[21] ^= D1;
        state[2]  ^= D2; state[7]  ^= D2; state[12] ^= D2; state[17] ^= D2; state[22] ^= D2;
        state[3]  ^= D3; state[8]  ^= D3; state[13] ^= D3; state[18] ^= D3; state[23] ^= D3;
        state[4]  ^= D4; state[9]  ^= D4; state[14] ^= D4; state[19] ^= D4; state[24] ^= D4;

        // Rho + Pi in-place
        ulong temp = state[1];
        state[1]  = rotl64(state[6],  44);
        state[6]  = rotl64(state[9],  20);
        state[9]  = rotl64(state[22], 61);
        state[22] = rotl64(state[14], 39);
        state[14] = rotl64(state[20], 18);
        state[20] = rotl64(state[2],  62);
        state[2]  = rotl64(state[12], 43);
        state[12] = rotl64(state[13], 25);
        state[13] = rotl64(state[19],  8);
        state[19] = rotl64(state[23], 56);
        state[23] = rotl64(state[15], 41);
        state[15] = rotl64(state[4],  27);
        state[4]  = rotl64(state[24], 14);
        state[24] = rotl64(state[21],  2);
        state[21] = rotl64(state[8],  55);
        state[8]  = rotl64(state[16], 45);
        state[16] = rotl64(state[5],  36);
        state[5]  = rotl64(state[3],  28);
        state[3]  = rotl64(state[18], 21);
        state[18] = rotl64(state[17], 15);
        state[17] = rotl64(state[11], 10);
        state[11] = rotl64(state[7],   6);
        state[7]  = rotl64(state[10],  3);
        state[10] = rotl64(temp,       1);

        // Chi
        #pragma unroll
        for (uint y = 0; y < 5; y++) {
            ulong t0 = state[y*5], t1 = state[y*5+1], t2 = state[y*5+2],
                  t3 = state[y*5+3], t4 = state[y*5+4];
            state[y*5]   = t0 ^ (~t1 & t2);
            state[y*5+1] = t1 ^ (~t2 & t3);
            state[y*5+2] = t2 ^ (~t3 & t4);
            state[y*5+3] = t3 ^ (~t4 & t0);
            state[y*5+4] = t4 ^ (~t0 & t1);
        }

        // Iota
        state[0] ^= KECCAK_RC[round];
    }
}

// Keccak-256 hash of a 64-byte input (two 32-byte leaves for Merkle)
// rate = 1088 bits = 136 bytes for Keccak-256
// Absorb 64 bytes, pad, squeeze 32 bytes
// Uses bit-interleaved representation for lower register pressure on 32-bit ALUs.
kernel void keccak256_hash_64(
    device const uchar* input      [[buffer(0)]],
    device uchar* output           [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    // Initialize interleaved state to zero
    uint2 state[25];
    for (uint i = 0; i < 25; i++) state[i] = uint2(0, 0);

    // Absorb 64 bytes of input (8 lanes of 8 bytes each)
    device const ulong* in64 = (device const ulong*)(input + gid * 64);
    for (uint i = 0; i < 8; i++) {
        state[i] = to_interleaved(in64[i]);
    }

    // Keccak padding: 0x01 at byte 64 (lane 8, byte 0), 0x80 at byte 135 (lane 16, byte 7)
    // Pre-computed: to_interleaved(0x01) = uint2(1,0), to_interleaved(0x80...0) = uint2(0, 0x80000000)
    state[8] ^= uint2(0x00000001u, 0x00000000u);
    state[16] ^= uint2(0x00000000u, 0x80000000u);

    // Permute
    keccak_f1600_il(state);

    // Squeeze 32 bytes (4 lanes) — de-interleave
    device ulong* out64 = (device ulong*)(output + gid * 32);
    for (uint i = 0; i < 4; i++) {
        out64[i] = from_interleaved(state[i]);
    }
}

// Bit-interleaved hash of two 32-byte inputs for Merkle (shared memory, uint2 format)
void keccak256_hash_pair_il(threadgroup uint2* left, threadgroup uint2* right,
                             threadgroup uint2* out) {
    uint2 state[25];
    for (uint i = 0; i < 25; i++) state[i] = uint2(0, 0);
    for (uint i = 0; i < 4; i++) state[i] = left[i];
    for (uint i = 0; i < 4; i++) state[4 + i] = right[i];
    state[8] ^= uint2(0x00000001u, 0x00000000u);
    state[16] ^= uint2(0x00000000u, 0x80000000u);
    keccak_f1600_il(state);
    for (uint i = 0; i < 4; i++) out[i] = state[i];
}

// Fused multi-level Keccak Merkle tree: each threadgroup processes a subtree
// Shared memory: 1024 * 4 uint2 = 32KB max (bit-interleaved format)
// Subtree size determined by num_levels: subtree_size = 1 << num_levels (max 1024)

kernel void keccak256_merkle_fused(
    device const uchar* leaves    [[buffer(0)]],
    device uchar* roots           [[buffer(1)]],
    constant uint& num_levels     [[buffer(2)]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    threadgroup uint2 shared_data[1024 * 4];

    uint subtree_size = 1u << num_levels;
    uint leaf_base = tgid * subtree_size;
    device const ulong* leaves64 = (device const ulong*)leaves;

    // Load leaves and convert to bit-interleaved format (generic stride loop)
    for (uint i = tid; i < subtree_size; i += tg_size) {
        for (uint j = 0; j < 4; j++) {
            shared_data[i * 4 + j] = to_interleaved(leaves64[(leaf_base + i) * 4 + j]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active = subtree_size;
    for (uint level = 0; level < num_levels; level++) {
        uint pairs = active >> 1;
        if (tid < pairs) {
            keccak256_hash_pair_il(&shared_data[tid * 2 * 4],
                                   &shared_data[(tid * 2 + 1) * 4],
                                   &shared_data[tid * 4]);
        }
        active = pairs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        device ulong* out64 = (device ulong*)(roots + tgid * 32);
        for (uint i = 0; i < 4; i++) out64[i] = from_interleaved(shared_data[i]);
    }
}

// Keccak-256 hash of a 32-byte input (single field element or leaf)
kernel void keccak256_hash_32(
    device const uchar* input      [[buffer(0)]],
    device uchar* output           [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint2 state[25];
    for (uint i = 0; i < 25; i++) state[i] = uint2(0, 0);

    device const ulong* in64 = (device const ulong*)(input + gid * 32);
    for (uint i = 0; i < 4; i++) {
        state[i] = to_interleaved(in64[i]);
    }

    state[4] ^= uint2(0x00000001u, 0x00000000u);
    state[16] ^= uint2(0x00000000u, 0x80000000u);

    keccak_f1600_il(state);

    device ulong* out64 = (device ulong*)(output + gid * 32);
    for (uint i = 0; i < 4; i++) {
        out64[i] = from_interleaved(state[i]);
    }
}

// Keccak-256 hash of a 4-byte M31 value → 32-byte digest
// Equivalent to keccak256(le_bytes(uint32)) with standard padding
kernel void keccak256_hash_m31(
    device const uint* input       [[buffer(0)]],
    device uchar* output           [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint2 state[25];
    for (uint i = 0; i < 25; i++) state[i] = uint2(0, 0);

    // 4 bytes of M31 data at offset 0, padding 0x01 at byte 4
    ulong lane0 = ulong(input[gid]) | (ulong(0x01) << 32);
    state[0] = to_interleaved(lane0);
    // Final padding bit at byte 135 (lane 16, byte 7)
    state[16] ^= uint2(0x00000000u, 0x80000000u);

    keccak_f1600_il(state);

    device ulong* out64p = (device ulong*)(output + gid * 32);
    for (uint i = 0; i < 4; i++) {
        out64p[i] = from_interleaved(state[i]);
    }
}
