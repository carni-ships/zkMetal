// NTT/iNTT GPU kernels for BN254 scalar field — WebGPU/WGSL
// Cooley-Tukey radix-2 DIT forward transform
// Gentleman-Sande radix-2 DIF inverse transform
// Multi-pass: one dispatch per butterfly stage

// ---- Inline BN254 Fr arithmetic (WGSL has no #include) ----
// In production, a preprocessor would merge bn254_fr.wgsl here.
// For clarity, we duplicate the essential functions.

const FR_P = array<u32, 8>(
    0xf0000001u, 0x43e1f593u, 0x79b97091u, 0x2833e848u,
    0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u
);
const FR_INV: u32 = 0xefffffffu;
const FR_ONE = array<u32, 8>(
    0x4ffffffbu, 0xac96341cu, 0x9f60cd29u, 0x36fc7695u,
    0x7879462eu, 0x666ea36fu, 0x9a07df2fu, 0x0e0a77c1u
);

fn mul_wide(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xffffu; let a_hi = a >> 16u;
    let b_lo = b & 0xffffu; let b_hi = b >> 16u;
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

fn adc(a: u32, b: u32, c: u32) -> vec2<u32> {
    let s1 = a + b; let c1 = select(0u, 1u, s1 < a);
    let s2 = s1 + c; let c2 = select(0u, 1u, s2 < s1);
    return vec2<u32>(s2, c1 + c2);
}

fn sbb(a: u32, b: u32, borrow: u32) -> vec2<u32> {
    let d1 = a - b; let b1 = select(0u, 1u, a < b);
    let d2 = d1 - borrow; let b2 = select(0u, 1u, d1 < borrow);
    return vec2<u32>(d2, b1 + b2);
}

fn fr_gte(a: array<u32, 8>, b: array<u32, 8>) -> bool {
    for (var i = 7i; i >= 0i; i--) {
        if (a[i] > b[i]) { return true; }
        if (a[i] < b[i]) { return false; }
    }
    return true;
}

fn fr_add_raw(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 9> {
    var r: array<u32, 9>; var carry = 0u;
    for (var i = 0u; i < 8u; i++) {
        let s = adc(a[i], b[i], carry); r[i] = s.x; carry = s.y;
    }
    r[8] = carry; return r;
}

fn fr_sub_raw(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 9> {
    var r: array<u32, 9>; var bw = 0u;
    for (var i = 0u; i < 8u; i++) {
        let d = sbb(a[i], b[i], bw); r[i] = d.x; bw = d.y;
    }
    r[8] = bw; return r;
}

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

fn fr_mul(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var t: array<u32, 10>;
    for (var k = 0u; k < 10u; k++) { t[k] = 0u; }
    for (var i = 0u; i < 8u; i++) {
        var carry = 0u;
        for (var j = 0u; j < 8u; j++) {
            let prod = mul_wide(a[i], b[j]);
            let s1 = adc(t[j], prod.x, carry);
            let s2 = adc(s1.y, prod.y, 0u);
            t[j] = s1.x; carry = s2.x;
        }
        let ext1 = adc(t[8], carry, 0u);
        t[8] = ext1.x; t[9] = ext1.y;

        let m = t[0] * FR_INV;
        let mp0 = mul_wide(m, FR_P[0]);
        let red0 = adc(t[0], mp0.x, 0u);
        carry = adc(red0.y, mp0.y, 0u).x;
        for (var j = 1u; j < 8u; j++) {
            let mp = mul_wide(m, FR_P[j]);
            let s1 = adc(t[j], mp.x, carry);
            let s2 = adc(s1.y, mp.y, 0u);
            t[j - 1u] = s1.x; carry = s2.x;
        }
        let ext2 = adc(t[8], carry, 0u);
        t[7] = ext2.x; t[8] = t[9] + ext2.y; t[9] = 0u;
    }
    var r: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) { r[i] = t[i]; }
    if (t[8] != 0u || fr_gte(r, FR_P)) {
        let d = fr_sub_raw(r, FR_P);
        for (var i = 0u; i < 8u; i++) { r[i] = d[i]; }
    }
    return r;
}

// ---- NTT Kernel Data Layout ----
// data: array of N field elements, each 8 x u32 = 32 bytes
// twiddles: array of N/2 precomputed twiddle factors in Montgomery form

struct NttParams {
    n: u32,
    stage: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read> twiddles: array<u32>;
@group(0) @binding(2) var<uniform> params: NttParams;

fn load_fr(buf: ptr<storage, array<u32>, read_write>, idx: u32) -> array<u32, 8> {
    var r: array<u32, 8>;
    let base = idx * 8u;
    for (var i = 0u; i < 8u; i++) { r[i] = (*buf)[base + i]; }
    return r;
}

fn store_fr(buf: ptr<storage, array<u32>, read_write>, idx: u32, val: array<u32, 8>) {
    let base = idx * 8u;
    for (var i = 0u; i < 8u; i++) { (*buf)[base + i] = val[i]; }
}

fn load_fr_ro(idx: u32) -> array<u32, 8> {
    var r: array<u32, 8>;
    let base = idx * 8u;
    for (var i = 0u; i < 8u; i++) { r[i] = twiddles[base + i]; }
    return r;
}

// --- NTT Butterfly Kernel (Cooley-Tukey DIT, one stage per dispatch) ---
// Each thread processes one butterfly pair.
// stage: current stage index (0 = stride 1, 1 = stride 2, ...)
//   a' = a + w*b
//   b' = a - w*b

@compute @workgroup_size(256)
fn ntt_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_block = 1u << params.stage;
    let block_size = half_block << 1u;
    let num_butterflies = params.n >> 1u;

    if (gid.x >= num_butterflies) { return; }

    let block_idx = gid.x / half_block;
    let local_idx = gid.x % half_block;
    let i = block_idx * block_size + local_idx;
    let j = i + half_block;

    // Twiddle factor index: for DIT stage s, twiddle[local_idx * (n / block_size)]
    let twiddle_idx = local_idx * (params.n / block_size);

    let a = load_fr(&data, i);
    let b = load_fr(&data, j);
    let w = load_fr_ro(twiddle_idx);
    let wb = fr_mul(w, b);

    store_fr(&data, i, fr_add(a, wb));
    store_fr(&data, j, fr_sub(a, wb));
}

// --- iNTT Butterfly Kernel (Gentleman-Sande DIF, one stage per dispatch) ---
//   a' = a + b
//   b' = (a - b) * w_inv

@compute @workgroup_size(256)
fn intt_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_block = 1u << params.stage;
    let block_size = half_block << 1u;
    let num_butterflies = params.n >> 1u;

    if (gid.x >= num_butterflies) { return; }

    let block_idx = gid.x / half_block;
    let local_idx = gid.x % half_block;
    let i = block_idx * block_size + local_idx;
    let j = i + half_block;

    let twiddle_idx = local_idx * (params.n / block_size);

    let a = load_fr(&data, i);
    let b = load_fr(&data, j);

    let sum = fr_add(a, b);
    let diff = fr_sub(a, b);
    let w = load_fr_ro(twiddle_idx);

    store_fr(&data, i, sum);
    store_fr(&data, j, fr_mul(diff, w));
}
