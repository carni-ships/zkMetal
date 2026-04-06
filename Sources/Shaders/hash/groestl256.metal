// Groestl-256 GPU kernel — batch hashing on Metal
// Each thread computes one Groestl-256 hash of a 64-byte input (single message block).
// Uses AES S-box in constant memory, operates on 8x8 byte state matrix.

#include <metal_stdlib>
using namespace metal;

// AES S-box in constant memory (shared across all threads)
constant uchar GROESTL_SBOX[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
};

// GF(2^8) multiply by 2 with reduction polynomial 0x1b
inline uchar gf_mul2(uchar x) {
    return (x << 1) ^ ((x >> 7) * 0x1b);
}

// GF(2^8) multiply by arbitrary value via repeated doubling
inline uchar gf_mul(uchar a, uchar b) {
    uchar r = 0;
    for (uint i = 0; i < 8; i++) {
        if (b & 1) r ^= a;
        a = gf_mul2(a);
        b >>= 1;
    }
    return r;
}

// MixBytes circulant matrix row: [2, 2, 3, 4, 5, 3, 5, 7]
constant uchar MIX_ROW[8] = {2, 2, 3, 4, 5, 3, 5, 7};

// State is stored as state[col * 8 + row], 8 columns x 8 rows = 64 bytes
// Column-major: byte index i -> col = i/8, row = i%8

// SubBytes: apply AES S-box to all 64 bytes
inline void groestl_sub_bytes(thread uchar state[64]) {
    for (uint i = 0; i < 64; i++) {
        state[i] = GROESTL_SBOX[state[i]];
    }
}

// ShiftBytes for P permutation: row r is shifted left by r positions
inline void groestl_shift_bytes_P(thread uchar state[64]) {
    for (uint row = 1; row < 8; row++) {
        uchar tmp[8];
        for (uint col = 0; col < 8; col++) {
            tmp[col] = state[((col + row) % 8) * 8 + row];
        }
        for (uint col = 0; col < 8; col++) {
            state[col * 8 + row] = tmp[col];
        }
    }
}

// ShiftBytes for Q permutation: shift amounts [1,3,5,7,0,2,4,6]
inline void groestl_shift_bytes_Q(thread uchar state[64]) {
    const uint shifts[8] = {1, 3, 5, 7, 0, 2, 4, 6};
    for (uint row = 0; row < 8; row++) {
        uint shift = shifts[row];
        if (shift == 0) continue;
        uchar tmp[8];
        for (uint col = 0; col < 8; col++) {
            tmp[col] = state[((col + shift) % 8) * 8 + row];
        }
        for (uint col = 0; col < 8; col++) {
            state[col * 8 + row] = tmp[col];
        }
    }
}

// MixBytes: multiply each column by the circulant MDS matrix
inline void groestl_mix_bytes(thread uchar state[64]) {
    for (uint col = 0; col < 8; col++) {
        uchar newcol[8];
        for (uint row = 0; row < 8; row++) {
            uchar acc = 0;
            for (uint k = 0; k < 8; k++) {
                // Matrix element at (row, k) = MIX_ROW[(k - row + 8) % 8]
                acc ^= gf_mul(MIX_ROW[(k + 8 - row) % 8], state[col * 8 + k]);
            }
            newcol[row] = acc;
        }
        for (uint row = 0; row < 8; row++) {
            state[col * 8 + row] = newcol[row];
        }
    }
}

// P permutation (10 rounds)
inline void groestl_P(thread uchar state[64]) {
    for (uint round = 0; round < 10; round++) {
        // AddRoundConstant for P
        for (uint col = 0; col < 8; col++) {
            state[col * 8 + 0] ^= uchar((col << 4) ^ round);
        }
        groestl_sub_bytes(state);
        groestl_shift_bytes_P(state);
        groestl_mix_bytes(state);
    }
}

// Q permutation (10 rounds)
inline void groestl_Q(thread uchar state[64]) {
    for (uint round = 0; round < 10; round++) {
        // AddRoundConstant for Q
        for (uint col = 0; col < 8; col++) {
            for (uint row = 0; row < 7; row++) {
                state[col * 8 + row] ^= 0xff;
            }
            state[col * 8 + 7] ^= uchar(0xff ^ ((col << 4) ^ round));
        }
        groestl_sub_bytes(state);
        groestl_shift_bytes_Q(state);
        groestl_mix_bytes(state);
    }
}

// Groestl-256 hash of a single 64-byte message block
// Handles padding internally: one block of data + padding block
kernel void groestl256_hash_batch(
    device const uchar* input    [[buffer(0)]],
    device uchar* output         [[buffer(1)]],
    constant uint& count         [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    // IV: 256-bit output length encoded in last 2 bytes
    uchar h[64];
    for (uint i = 0; i < 64; i++) h[i] = 0;
    h[62] = 0x01;
    h[63] = 0x00;

    // Load 64-byte message block
    uchar m[64];
    for (uint i = 0; i < 64; i++) {
        m[i] = input[gid * 64 + i];
    }

    // Compression: h' = P(h XOR m) XOR Q(m) XOR h
    uchar hxm[64];
    for (uint i = 0; i < 64; i++) hxm[i] = h[i] ^ m[i];

    uchar qm[64];
    for (uint i = 0; i < 64; i++) qm[i] = m[i];

    groestl_P(hxm);
    groestl_Q(qm);

    for (uint i = 0; i < 64; i++) {
        h[i] = hxm[i] ^ qm[i] ^ h[i];
    }

    // Padding block: 0x80, zeros, then 64-bit block count = 2 (IV + data block)
    uchar pad[64];
    for (uint i = 0; i < 64; i++) pad[i] = 0;
    pad[0] = 0x80;
    // Block count = 2 (big-endian, last 8 bytes)
    pad[63] = 2;

    // Second compression
    uchar hxp[64];
    for (uint i = 0; i < 64; i++) hxp[i] = h[i] ^ pad[i];

    uchar qp[64];
    for (uint i = 0; i < 64; i++) qp[i] = pad[i];

    groestl_P(hxp);
    groestl_Q(qp);

    for (uint i = 0; i < 64; i++) {
        h[i] = hxp[i] ^ qp[i] ^ h[i];
    }

    // Output transformation: Omega(h) = P(h) XOR h, then take last 32 bytes
    uchar ph[64];
    for (uint i = 0; i < 64; i++) ph[i] = h[i];
    groestl_P(ph);

    for (uint i = 0; i < 32; i++) {
        output[gid * 32 + i] = ph[32 + i] ^ h[32 + i];
    }
}
