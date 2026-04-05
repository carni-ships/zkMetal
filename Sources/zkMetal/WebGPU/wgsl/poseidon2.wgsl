// Poseidon2 GPU kernel for BN254 Fr, t=3 — WebGPU/WGSL
// Each thread computes one independent Poseidon2 permutation.
// d=5 (x^5 S-box), rounds_f=8, rounds_p=56, total=64 rounds

// ---- Inline BN254 Fr arithmetic ----

const FR_P = array<u32, 8>(
    0xf0000001u, 0x43e1f593u, 0x79b97091u, 0x2833e848u,
    0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u
);
const FR_INV: u32 = 0xefffffffu;

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

fn fr_add_lazy(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    let s = fr_add_raw(a, b);
    var r: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) { r[i] = s[i]; }
    return r;
}

fn fr_reduce(a: array<u32, 8>) -> array<u32, 8> {
    var r = a;
    if (fr_gte(a, FR_P)) {
        let d = fr_sub_raw(a, FR_P);
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

fn fr_zero() -> array<u32, 8> {
    return array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}

// ---- Poseidon2 Permutation ----

// S-box: x -> x^5
fn p2_sbox(x: array<u32, 8>) -> array<u32, 8> {
    let x2 = fr_mul(x, x);
    let x4 = fr_mul(x2, x2);
    return fr_mul(x4, x);
}

// External linear layer: circulant [2,1,1] for t=3
// M_E * [a,b,c] = [a+(a+b+c), b+(a+b+c), c+(a+b+c)]
struct State3 {
    s0: array<u32, 8>,
    s1: array<u32, 8>,
    s2: array<u32, 8>,
}

fn p2_external_layer(st: State3) -> State3 {
    let sum = fr_reduce(fr_add_lazy(fr_add_lazy(st.s0, st.s1), st.s2));
    return State3(
        fr_add_lazy(st.s0, sum),
        fr_add_lazy(st.s1, sum),
        fr_add_lazy(st.s2, sum),
    );
}

// Internal linear layer: M_I = [[2,1,1],[1,2,1],[1,1,3]]
fn p2_internal_layer(st: State3) -> State3 {
    let sum = fr_add(fr_add(st.s0, st.s1), st.s2);
    return State3(
        fr_add(st.s0, sum),
        fr_add(st.s1, sum),
        fr_add(fr_add(st.s2, sum), st.s2),  // a+b+3c
    );
}

// ---- Kernel Bindings ----

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read> rc: array<u32>;   // round constants: 64*3 = 192 Fr elements
@group(0) @binding(3) var<uniform> count: u32;              // number of permutations

fn load_fr_from(buf: ptr<storage, array<u32>, read>, idx: u32) -> array<u32, 8> {
    var r: array<u32, 8>;
    let base = idx * 8u;
    for (var i = 0u; i < 8u; i++) { r[i] = (*buf)[base + i]; }
    return r;
}

fn store_fr_to(buf: ptr<storage, array<u32>, read_write>, idx: u32, val: array<u32, 8>) {
    let base = idx * 8u;
    for (var i = 0u; i < 8u; i++) { (*buf)[base + i] = val[i]; }
}

fn load_rc(idx: u32) -> array<u32, 8> {
    var r: array<u32, 8>;
    let base = idx * 8u;
    for (var i = 0u; i < 8u; i++) { r[i] = rc[base + i]; }
    return r;
}

// Poseidon2 permutation: each thread computes one independent hash
@compute @workgroup_size(256)
fn poseidon2_permute(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= count) { return; }

    let base_idx = gid.x * 3u;
    var st = State3(
        load_fr_from(&input, base_idx),
        load_fr_from(&input, base_idx + 1u),
        load_fr_from(&input, base_idx + 2u),
    );

    // Initial external linear layer
    st = p2_external_layer(st);

    // First half of full rounds (rounds 0..3)
    for (var r = 0u; r < 4u; r++) {
        let rc_base = r * 3u;
        st.s0 = fr_add_lazy(st.s0, load_rc(rc_base));
        st.s1 = fr_add_lazy(st.s1, load_rc(rc_base + 1u));
        st.s2 = fr_add_lazy(st.s2, load_rc(rc_base + 2u));
        st.s0 = p2_sbox(st.s0);
        st.s1 = p2_sbox(st.s1);
        st.s2 = p2_sbox(st.s2);
        st = p2_external_layer(st);
    }

    // Reduce before partial rounds
    st.s0 = fr_reduce(st.s0);
    st.s1 = fr_reduce(st.s1);
    st.s2 = fr_reduce(st.s2);

    // Partial rounds (rounds 4..59) — only s0 gets RC and S-box
    for (var r = 4u; r < 60u; r++) {
        st.s0 = fr_add_lazy(st.s0, load_rc(r * 3u));
        st.s0 = p2_sbox(st.s0);
        st = p2_internal_layer(st);
    }

    // Second half of full rounds (rounds 60..63)
    for (var r = 60u; r < 64u; r++) {
        let rc_base = r * 3u;
        st.s0 = fr_add_lazy(st.s0, load_rc(rc_base));
        st.s1 = fr_add_lazy(st.s1, load_rc(rc_base + 1u));
        st.s2 = fr_add_lazy(st.s2, load_rc(rc_base + 2u));
        st.s0 = p2_sbox(st.s0);
        st.s1 = p2_sbox(st.s1);
        st.s2 = p2_sbox(st.s2);
        st = p2_external_layer(st);
    }

    // Store reduced output
    store_fr_to(&output, base_idx, fr_reduce(st.s0));
    store_fr_to(&output, base_idx + 1u, fr_reduce(st.s1));
    store_fr_to(&output, base_idx + 2u, fr_reduce(st.s2));
}

// 2-to-1 compression: hash pairs of field elements
// state = [a, b, 0], output = permute(state)[0]
@compute @workgroup_size(256)
fn poseidon2_hash_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= count) { return; }

    var st = State3(
        load_fr_from(&input, gid.x * 2u),
        load_fr_from(&input, gid.x * 2u + 1u),
        fr_zero(),
    );

    st = p2_external_layer(st);

    for (var r = 0u; r < 4u; r++) {
        let rc_base = r * 3u;
        st.s0 = fr_add_lazy(st.s0, load_rc(rc_base));
        st.s1 = fr_add_lazy(st.s1, load_rc(rc_base + 1u));
        st.s2 = fr_add_lazy(st.s2, load_rc(rc_base + 2u));
        st.s0 = p2_sbox(st.s0); st.s1 = p2_sbox(st.s1); st.s2 = p2_sbox(st.s2);
        st = p2_external_layer(st);
    }

    st.s0 = fr_reduce(st.s0); st.s1 = fr_reduce(st.s1); st.s2 = fr_reduce(st.s2);

    for (var r = 4u; r < 60u; r++) {
        st.s0 = fr_add_lazy(st.s0, load_rc(r * 3u));
        st.s0 = p2_sbox(st.s0);
        st = p2_internal_layer(st);
    }

    for (var r = 60u; r < 64u; r++) {
        let rc_base = r * 3u;
        st.s0 = fr_add_lazy(st.s0, load_rc(rc_base));
        st.s1 = fr_add_lazy(st.s1, load_rc(rc_base + 1u));
        st.s2 = fr_add_lazy(st.s2, load_rc(rc_base + 2u));
        st.s0 = p2_sbox(st.s0); st.s1 = p2_sbox(st.s1); st.s2 = p2_sbox(st.s2);
        st = p2_external_layer(st);
    }

    store_fr_to(&output, gid.x, fr_reduce(st.s0));
}
