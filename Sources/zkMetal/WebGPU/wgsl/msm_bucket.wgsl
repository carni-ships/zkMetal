// MSM GPU kernels: Pippenger's bucket method — WebGPU/WGSL
// Phase 1: Signed-digit scalar extraction
// Phase 2: Bucket accumulation (serial per bucket)
// Phase 3: Weighted bucket sum per segment
// Phase 4: Segment combination per window
//
// Note: WGSL lacks SIMD shuffle and atomics with memory_order_relaxed,
// so cooperative reduce and GPU counting sort are handled differently
// than Metal. This port focuses on the serial bucket accumulation path.

// ---- Inline BN254 Fp arithmetic (base field for curve points) ----

const FP_P = array<u32, 8>(
    0xd87cfd47u, 0x3c208c16u, 0x6871ca8du, 0x97816a91u,
    0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u
);
const FP_INV: u32 = 0xd20d4127u;
const FP_ONE = array<u32, 8>(
    0xd35d438du, 0x0a78eb28u, 0x7178d00bu, 0x8c767a54u,
    0x71536f1cu, 0x7c1b7d10u, 0x85acf861u, 0x14c4d2e2u
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

fn fp_gte(a: array<u32, 8>, b: array<u32, 8>) -> bool {
    for (var i = 7i; i >= 0i; i--) {
        if (a[i] > b[i]) { return true; }
        if (a[i] < b[i]) { return false; }
    }
    return true;
}

fn fp_add_raw(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 9> {
    var r: array<u32, 9>; var carry = 0u;
    for (var i = 0u; i < 8u; i++) {
        let s = adc(a[i], b[i], carry); r[i] = s.x; carry = s.y;
    }
    r[8] = carry; return r;
}

fn fp_sub_raw(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 9> {
    var r: array<u32, 9>; var bw = 0u;
    for (var i = 0u; i < 8u; i++) {
        let d = sbb(a[i], b[i], bw); r[i] = d.x; bw = d.y;
    }
    r[8] = bw; return r;
}

fn fp_add(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    let s = fp_add_raw(a, b);
    var r: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) { r[i] = s[i]; }
    if (s[8] != 0u || fp_gte(r, FP_P)) {
        let d = fp_sub_raw(r, FP_P);
        for (var i = 0u; i < 8u; i++) { r[i] = d[i]; }
    }
    return r;
}

fn fp_sub(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    let d = fp_sub_raw(a, b);
    var r: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) { r[i] = d[i]; }
    if (d[8] != 0u) {
        let s = fp_add_raw(r, FP_P);
        for (var i = 0u; i < 8u; i++) { r[i] = s[i]; }
    }
    return r;
}

fn fp_double(a: array<u32, 8>) -> array<u32, 8> {
    return fp_add(a, a);
}

fn fp_mul(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
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
        let m = t[0] * FP_INV;
        let mp0 = mul_wide(m, FP_P[0]);
        let red0 = adc(t[0], mp0.x, 0u);
        carry = adc(red0.y, mp0.y, 0u).x;
        for (var j = 1u; j < 8u; j++) {
            let mp = mul_wide(m, FP_P[j]);
            let s1 = adc(t[j], mp.x, carry);
            let s2 = adc(s1.y, mp.y, 0u);
            t[j - 1u] = s1.x; carry = s2.x;
        }
        let ext2 = adc(t[8], carry, 0u);
        t[7] = ext2.x; t[8] = t[9] + ext2.y; t[9] = 0u;
    }
    var r: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) { r[i] = t[i]; }
    if (t[8] != 0u || fp_gte(r, FP_P)) {
        let d = fp_sub_raw(r, FP_P);
        for (var i = 0u; i < 8u; i++) { r[i] = d[i]; }
    }
    return r;
}

fn fp_zero() -> array<u32, 8> {
    return array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}

fn fp_one() -> array<u32, 8> {
    return FP_ONE;
}

fn fp_is_zero(a: array<u32, 8>) -> bool {
    return (a[0] | a[1] | a[2] | a[3] | a[4] | a[5] | a[6] | a[7]) == 0u;
}

fn fp_neg(a: array<u32, 8>) -> array<u32, 8> {
    if (fp_is_zero(a)) { return a; }
    return fp_sub(fp_zero(), a);
}

// ---- Elliptic Curve: BN254 G1 Jacobian Projective ----
// Point = (x: Fp, y: Fp, z: Fp), 24 u32 limbs total
// Affine = (x: Fp, y: Fp), 16 u32 limbs total

struct PointProj {
    x: array<u32, 8>,
    y: array<u32, 8>,
    z: array<u32, 8>,
}

struct PointAff {
    x: array<u32, 8>,
    y: array<u32, 8>,
}

fn point_identity() -> PointProj {
    return PointProj(fp_one(), fp_one(), fp_zero());
}

fn point_is_identity(p: PointProj) -> bool {
    return fp_is_zero(p.z);
}

fn point_from_affine(a: PointAff) -> PointProj {
    return PointProj(a.x, a.y, fp_one());
}

// Point doubling: 4M + 6S + 7add (a=0 for BN254)
fn point_double(p: PointProj) -> PointProj {
    if (point_is_identity(p)) { return p; }

    let a = fp_mul(p.x, p.x);
    let b = fp_mul(p.y, p.y);
    let c = fp_mul(b, b);

    let xb = fp_add(p.x, b);
    let d = fp_double(fp_sub(fp_mul(xb, xb), fp_add(a, c)));
    let e = fp_add(fp_double(a), a); // 3*X^2
    let f = fp_mul(e, e);

    let rx = fp_sub(f, fp_double(d));
    let ry = fp_sub(fp_mul(e, fp_sub(d, rx)), fp_double(fp_double(fp_double(c))));
    let yz = fp_add(p.y, p.z);
    let rz = fp_sub(fp_mul(yz, yz), fp_add(b, fp_mul(p.z, p.z)));
    return PointProj(rx, ry, rz);
}

// Mixed addition: projective + affine
fn point_add_mixed(p: PointProj, q: PointAff) -> PointProj {
    if (point_is_identity(p)) { return point_from_affine(q); }

    let z1z1 = fp_mul(p.z, p.z);
    let u2 = fp_mul(q.x, z1z1);
    let s2 = fp_mul(q.y, fp_mul(p.z, z1z1));
    let h = fp_sub(u2, p.x);
    let hh = fp_mul(h, h);
    let i = fp_double(fp_double(hh));
    let j = fp_mul(h, i);
    let rr = fp_double(fp_sub(s2, p.y));
    let v = fp_mul(p.x, i);

    let rx = fp_sub(fp_sub(fp_mul(rr, rr), j), fp_double(v));
    let ry = fp_sub(fp_mul(rr, fp_sub(v, rx)), fp_double(fp_mul(p.y, j)));
    let z1h = fp_add(p.z, h);
    let rz = fp_sub(fp_mul(z1h, z1h), fp_add(z1z1, hh));
    return PointProj(rx, ry, rz);
}

// Full projective addition
fn point_add(p: PointProj, q: PointProj) -> PointProj {
    if (point_is_identity(p)) { return q; }
    if (point_is_identity(q)) { return p; }

    let z1z1 = fp_mul(p.z, p.z);
    let z2z2 = fp_mul(q.z, q.z);
    let u1 = fp_mul(p.x, z2z2);
    let u2 = fp_mul(q.x, z1z1);
    let s1 = fp_mul(p.y, fp_mul(q.z, z2z2));
    let s2 = fp_mul(q.y, fp_mul(p.z, z1z1));
    let h = fp_sub(u2, u1);
    let i = fp_double(h);
    let ii = fp_mul(i, i);
    let j = fp_mul(h, ii);
    let rr = fp_double(fp_sub(s2, s1));
    let v = fp_mul(u1, ii);

    let rx = fp_sub(fp_sub(fp_mul(rr, rr), j), fp_double(v));
    let ry = fp_sub(fp_mul(rr, fp_sub(v, rx)), fp_double(fp_mul(s1, j)));
    let z12 = fp_add(p.z, q.z);
    let rz = fp_mul(fp_sub(fp_mul(z12, z12), fp_add(z1z1, z2z2)), h);
    return PointProj(rx, ry, rz);
}

// ---- Kernel Bindings ----

struct MsmParams {
    n_points: u32,
    window_bits: u32,
    n_buckets: u32,
}

// --- Signed-digit scalar extraction ---
// scalars: n_points * 8 u32 (256-bit scalars)
// digits: n_windows * n_points u32 (extracted digits, sign in MSB)

@group(0) @binding(0) var<storage, read> scalars: array<u32>;
@group(0) @binding(1) var<storage, read_write> digits: array<u32>;
@group(0) @binding(2) var<uniform> extract_params: vec3<u32>; // (n_points, window_bits, n_windows)

@compute @workgroup_size(256)
fn signed_digit_extract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_points = extract_params.x;
    let window_bits = extract_params.y;
    let n_windows = extract_params.z;

    if (gid.x >= n_points) { return; }

    let sp_base = gid.x * 8u;
    let mask = (1u << window_bits) - 1u;
    let half_bk = 1u << (window_bits - 1u);
    let full_bk = 1u << window_bits;
    var carry = 0u;

    for (var w = 0u; w < n_windows; w++) {
        let bit_off = w * window_bits;
        let limb_idx = bit_off / 32u;
        let bit_pos = bit_off % 32u;

        var idx = 0u;
        if (limb_idx < 8u) {
            idx = scalars[sp_base + limb_idx] >> bit_pos;
            if (bit_pos + window_bits > 32u && limb_idx + 1u < 8u) {
                idx |= scalars[sp_base + limb_idx + 1u] << (32u - bit_pos);
            }
            idx &= mask;
        }

        var digit = idx + carry;
        carry = 0u;
        if (digit > half_bk) {
            digit = full_bk - digit;
            carry = 1u;
            digits[w * n_points + gid.x] = digit | 0x80000000u;
        } else {
            digits[w * n_points + gid.x] = digit;
        }
    }
}

// --- Bucket sum: weighted sum per segment ---
// buckets: n_windows * n_buckets * 24 u32 (PointProj)
// segment_results: n_segments * n_windows * 24 u32

// Second bind group for bucket sum phase
@group(1) @binding(0) var<storage, read> buckets: array<u32>;
@group(1) @binding(1) var<storage, read_write> segment_results: array<u32>;
@group(1) @binding(2) var<uniform> msm_params: MsmParams;
@group(1) @binding(3) var<uniform> seg_params: vec2<u32>; // (n_segments, n_windows)

fn load_point(buf: ptr<storage, array<u32>, read>, idx: u32) -> PointProj {
    let base = idx * 24u;
    var p: PointProj;
    for (var i = 0u; i < 8u; i++) { p.x[i] = (*buf)[base + i]; }
    for (var i = 0u; i < 8u; i++) { p.y[i] = (*buf)[base + 8u + i]; }
    for (var i = 0u; i < 8u; i++) { p.z[i] = (*buf)[base + 16u + i]; }
    return p;
}

fn store_point(buf: ptr<storage, array<u32>, read_write>, idx: u32, p: PointProj) {
    let base = idx * 24u;
    for (var i = 0u; i < 8u; i++) { (*buf)[base + i] = p.x[i]; }
    for (var i = 0u; i < 8u; i++) { (*buf)[base + 8u + i] = p.y[i]; }
    for (var i = 0u; i < 8u; i++) { (*buf)[base + 16u + i] = p.z[i]; }
}

@compute @workgroup_size(256)
fn msm_bucket_sum_direct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_segments = seg_params.x;
    let n_windows = seg_params.y;
    let total = n_segments * n_windows;
    if (gid.x >= total) { return; }

    let window_idx = gid.x / n_segments;
    let seg_idx = gid.x % n_segments;
    let n_buckets = msm_params.n_buckets;
    let seg_size = (n_buckets + n_segments - 1u) / n_segments;
    let bucket_base = window_idx * n_buckets;

    let hi_s = i32(n_buckets) - i32(seg_idx * seg_size);
    let lo_raw_s = i32((seg_idx + 1u) * seg_size);
    var lo_s = select(i32(n_buckets) - lo_raw_s, 1i, lo_raw_s >= i32(n_buckets));
    if (lo_s < 1i) { lo_s = 1i; }
    if (hi_s <= lo_s) {
        store_point(&segment_results, gid.x, point_identity());
        return;
    }

    var running = point_identity();
    var sum = point_identity();
    let hi = u32(hi_s);
    let lo = u32(lo_s);

    for (var i = hi - 1u; i >= lo; i--) {
        let bucket = load_point(&buckets, bucket_base + i);
        if (!point_is_identity(bucket)) {
            if (point_is_identity(running)) {
                running = bucket;
            } else {
                running = point_add(running, bucket);
            }
        }
        if (!point_is_identity(running)) {
            if (point_is_identity(sum)) {
                sum = running;
            } else {
                sum = point_add(sum, running);
            }
        }
        if (i == lo) { break; }
    }

    // Weight the running total by (lo - 1)
    var weight = lo - 1u;
    if (weight > 0u && !point_is_identity(running)) {
        var weighted = point_identity();
        var base_pt = running;
        var k = weight;
        while (k > 0u) {
            if ((k & 1u) != 0u) {
                if (point_is_identity(weighted)) {
                    weighted = base_pt;
                } else {
                    weighted = point_add(weighted, base_pt);
                }
            }
            base_pt = point_double(base_pt);
            k >>= 1u;
        }
        if (point_is_identity(sum)) {
            sum = weighted;
        } else {
            sum = point_add(sum, weighted);
        }
    }

    store_point(&segment_results, gid.x, sum);
}
