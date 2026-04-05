// BN254 scalar field Fr arithmetic for WebGPU
//
// r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// Field elements as array<u32, 8> in Montgomery form (little-endian).
// TWO_ADICITY = 28 (r-1 = 2^28 * t, t odd)
//
// WGSL has no native u64, so all wide multiplies use 16-bit half-limb decomposition.

// BN254 scalar field modulus r (little-endian 32-bit limbs)
const FR_P = array<u32, 8>(
    0xf0000001u, 0x43e1f593u, 0x79b97091u, 0x2833e848u,
    0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u
);

// Montgomery parameter: R^2 mod r
const FR_R2 = array<u32, 8>(
    0xae216da7u, 0x1bb8e645u, 0xe35c59e3u, 0x53fe3ab1u,
    0x53bb8085u, 0x8c49833du, 0x7f4e44a5u, 0x0216d0b1u
);

// Montgomery inverse: -r^(-1) mod 2^32
const FR_INV: u32 = 0xefffffffu;

// R mod r (Montgomery form of 1)
const FR_ONE = array<u32, 8>(
    0x4ffffffbu, 0xac96341cu, 0x9f60cd29u, 0x36fc7695u,
    0x7879462eu, 0x666ea36fu, 0x9a07df2fu, 0x0e0a77c1u
);

// --- Emulated 64-bit multiply ---
// Returns (lo, hi) of a * b where a, b are u32.
// Splits each operand into 16-bit halves and cross-multiplies.
fn mul_wide(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xffffu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xffffu;
    let b_hi = b >> 16u;

    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;

    let mid = p1 + (p0 >> 16u);
    let mid2 = (mid & 0xffffu) + p2;

    let lo = ((mid2 & 0xffffu) << 16u) | (p0 & 0xffffu);
    let hi = p3 + (mid >> 16u) + (mid2 >> 16u);
    return vec2<u32>(lo, hi);
}

// Add with carry: returns (sum, carry)
fn adc(a: u32, b: u32, carry_in: u32) -> vec2<u32> {
    let sum_lo = a + b;
    let c1 = select(0u, 1u, sum_lo < a);
    let sum = sum_lo + carry_in;
    let c2 = select(0u, 1u, sum < sum_lo);
    return vec2<u32>(sum, c1 + c2);
}

// Subtract with borrow: returns (diff, borrow)
fn sbb(a: u32, b: u32, borrow_in: u32) -> vec2<u32> {
    let diff1 = a - b;
    let b1 = select(0u, 1u, a < b);
    let diff = diff1 - borrow_in;
    let b2 = select(0u, 1u, diff1 < borrow_in);
    return vec2<u32>(diff, b1 + b2);
}

// --- 256-bit Field Operations ---

fn fr_zero() -> array<u32, 8> {
    return array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}

fn fr_one() -> array<u32, 8> {
    return FR_ONE;
}

fn fr_is_zero(a: array<u32, 8>) -> bool {
    return (a[0] | a[1] | a[2] | a[3] | a[4] | a[5] | a[6] | a[7]) == 0u;
}

// a >= b (unsigned 256-bit comparison)
fn fr_gte(a: array<u32, 8>, b: array<u32, 8>) -> bool {
    for (var i = 7i; i >= 0i; i--) {
        if (a[i] > b[i]) { return true; }
        if (a[i] < b[i]) { return false; }
    }
    return true; // equal
}

// Raw addition: a + b, returns (result, carry)
fn fr_add_raw(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 9> {
    var r: array<u32, 9>;
    var carry = 0u;
    for (var i = 0u; i < 8u; i++) {
        let s = adc(a[i], b[i], carry);
        r[i] = s.x;
        carry = s.y;
    }
    r[8] = carry;
    return r;
}

// Raw subtraction: a - b, returns (result, borrow in last element)
fn fr_sub_raw(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 9> {
    var r: array<u32, 9>;
    var borrow = 0u;
    for (var i = 0u; i < 8u; i++) {
        let d = sbb(a[i], b[i], borrow);
        r[i] = d.x;
        borrow = d.y;
    }
    r[8] = borrow;
    return r;
}

// Modular addition: (a + b) mod r
fn fr_add(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    let s = fr_add_raw(a, b);
    var r: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) { r[i] = s[i]; }

    if (s[8] != 0u || fr_gte(r, FR_P)) {
        let d = fr_sub_raw(r, FR_P);
        for (var i = 0u; i < 8u; i++) { r[i] = d[i]; }
    }
    return r;
}

// Lazy addition: no modular reduction. Result may be in [0, 2^256).
fn fr_add_lazy(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    let s = fr_add_raw(a, b);
    var r: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) { r[i] = s[i]; }
    return r;
}

// Reduce value to [0, p)
fn fr_reduce(a: array<u32, 8>) -> array<u32, 8> {
    var r = a;
    if (fr_gte(a, FR_P)) {
        let d = fr_sub_raw(a, FR_P);
        for (var i = 0u; i < 8u; i++) { r[i] = d[i]; }
    }
    return r;
}

// Modular subtraction: (a - b) mod r
fn fr_sub(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    let d = fr_sub_raw(a, b);
    var r: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) { r[i] = d[i]; }

    if (d[8] != 0u) {
        let s = fr_add_raw(r, FR_P);
        for (var i = 0u; i < 8u; i++) { r[i] = s[i]; }
    }
    return r;
}

// Specialized doubling: left-shift by 1
fn fr_double(a: array<u32, 8>) -> array<u32, 8> {
    var r: array<u32, 8>;
    var carry = 0u;
    for (var i = 0u; i < 8u; i++) {
        let doubled = (a[i] << 1u) | carry;
        carry = a[i] >> 31u;
        r[i] = doubled;
    }
    if (carry != 0u || fr_gte(r, FR_P)) {
        let d = fr_sub_raw(r, FR_P);
        for (var i = 0u; i < 8u; i++) { r[i] = d[i]; }
    }
    return r;
}

// --- Montgomery CIOS Multiplication ---
// Without native u64, we emulate the u32*u32->u64 products using mul_wide.
// This is the core performance-critical function.
//
// CIOS algorithm: for each limb i of a, multiply-accumulate a[i]*b[j]
// into accumulator t[], then reduce with Montgomery factor m.
fn fr_mul(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var t: array<u32, 10>;
    for (var k = 0u; k < 10u; k++) { t[k] = 0u; }

    for (var i = 0u; i < 8u; i++) {
        // Multiply-accumulate: t += a[i] * b
        var carry = 0u;
        for (var j = 0u; j < 8u; j++) {
            let prod = mul_wide(a[i], b[j]);
            // t[j] + prod.lo + carry
            let s1 = adc(t[j], prod.x, carry);
            // carry from addition + prod.hi
            let s2 = adc(s1.y, prod.y, 0u);
            t[j] = s1.x;
            carry = s2.x;
        }
        let ext1 = adc(t[8], carry, 0u);
        t[8] = ext1.x;
        t[9] = ext1.y;

        // Montgomery reduction step
        let m = t[0] * FR_INV;
        let mp0 = mul_wide(m, FR_P[0]);
        let red0 = adc(t[0], mp0.x, 0u);
        carry = adc(red0.y, mp0.y, 0u).x;

        for (var j = 1u; j < 8u; j++) {
            let mp = mul_wide(m, FR_P[j]);
            let s1 = adc(t[j], mp.x, carry);
            let s2 = adc(s1.y, mp.y, 0u);
            t[j - 1u] = s1.x;
            carry = s2.x;
        }
        let ext2 = adc(t[8], carry, 0u);
        t[7] = ext2.x;
        t[8] = t[9] + ext2.y;
        t[9] = 0u;
    }

    var r: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) { r[i] = t[i]; }
    if (t[8] != 0u || fr_gte(r, FR_P)) {
        let d = fr_sub_raw(r, FR_P);
        for (var i = 0u; i < 8u; i++) { r[i] = d[i]; }
    }
    return r;
}

// Montgomery squaring (reuses mul for correctness; can be specialized later)
fn fr_sqr(a: array<u32, 8>) -> array<u32, 8> {
    return fr_mul(a, a);
}

// Modular exponentiation by squaring: a^exp mod r
fn fr_pow(base_in: array<u32, 8>, exp: array<u32, 8>) -> array<u32, 8> {
    var result = fr_one();
    var b = base_in;
    for (var i = 0u; i < 8u; i++) {
        var word = exp[i];
        for (var bit = 0u; bit < 32u; bit++) {
            if ((word & 1u) != 0u) {
                result = fr_mul(result, b);
            }
            b = fr_sqr(b);
            word >>= 1u;
        }
    }
    return result;
}

// Modular inverse via Fermat's little theorem: a^(r-2) mod r
fn fr_inv(a: array<u32, 8>) -> array<u32, 8> {
    let exp = array<u32, 8>(
        0xefffffffu, 0x43e1f593u, 0x79b97091u, 0x2833e848u,
        0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u
    );
    return fr_pow(a, exp);
}
