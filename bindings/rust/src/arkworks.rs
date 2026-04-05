//! Arkworks compatibility layer for zkMetal GPU kernels.
//!
//! Provides zero-copy (where possible) conversions between arkworks types
//! (`ark_bn254::Fr`, `ark_bn254::G1Affine`, `ark_bn254::G1Projective`) and
//! zkMetal's FFI types, plus high-level wrappers for GPU MSM and NTT.
//!
//! Enabled by the `arkworks` feature flag.
//!
//! # Example
//!
//! ```rust,no_run
//! use ark_bn254::{Fr, G1Affine};
//! use zkmetal_sys::arkworks::{ArkMSM, ArkNTT};
//!
//! let points: Vec<G1Affine> = /* ... */;
//! let scalars: Vec<Fr> = /* ... */;
//! let result: ark_bn254::G1Projective = ArkMSM::msm(&points, &scalars).unwrap();
//! ```

use ark_bn254::{Fq, Fr as ArkFr, G1Affine as ArkG1Affine, G1Projective as ArkG1Projective};
use ark_ff::{BigInteger, BigInteger256, Field, PrimeField};

use crate::bn254::Fr;
use crate::msm::{G1Affine, G1Projective};

// ============================================================================
// Fr conversions
// ============================================================================

impl From<ArkFr> for Fr {
    /// Convert `ark_bn254::Fr` to `zkmetal::bn254::Fr`.
    ///
    /// Both are 4 x u64 limbs in little-endian Montgomery form, so this is
    /// a direct limb copy with no arithmetic.
    fn from(ark_fr: ArkFr) -> Self {
        let bigint: BigInteger256 = ark_fr.into();
        Fr(bigint.0)
    }
}

impl From<Fr> for ArkFr {
    /// Convert `zkmetal::bn254::Fr` to `ark_bn254::Fr`.
    ///
    /// Wraps the raw Montgomery limbs back into an arkworks field element.
    fn from(zk_fr: Fr) -> Self {
        ArkFr::from_bigint(BigInteger256::new(zk_fr.0))
            .expect("zkMetal Fr limbs should be valid BN254 Fr")
    }
}

// ============================================================================
// G1Affine conversions
// ============================================================================

impl From<ArkG1Affine> for G1Affine {
    /// Convert `ark_bn254::G1Affine` to `zkmetal::msm::G1Affine`.
    ///
    /// Arkworks stores affine coordinates as `(Fq, Fq)` with `Fq` being
    /// `[u64; 4]` Montgomery limbs. zkMetal expects 32-byte little-endian
    /// Montgomery coordinates, which is the same memory layout.
    fn from(ark_pt: ArkG1Affine) -> Self {
        if ark_pt.infinity {
            // Point at infinity: encode as (0, 0).
            return G1Affine {
                x: [0u8; 32],
                y: [0u8; 32],
            };
        }

        let x_bigint: BigInteger256 = ark_pt.x.into();
        let y_bigint: BigInteger256 = ark_pt.y.into();

        G1Affine {
            x: limbs_to_le_bytes(&x_bigint.0),
            y: limbs_to_le_bytes(&y_bigint.0),
        }
    }
}

impl From<G1Affine> for ArkG1Affine {
    /// Convert `zkmetal::msm::G1Affine` to `ark_bn254::G1Affine`.
    fn from(zk_pt: G1Affine) -> Self {
        let x_limbs = le_bytes_to_limbs(&zk_pt.x);
        let y_limbs = le_bytes_to_limbs(&zk_pt.y);

        // Check for point at infinity (both coordinates zero).
        if x_limbs == [0u64; 4] && y_limbs == [0u64; 4] {
            return ArkG1Affine::identity();
        }

        let x = Fq::from_bigint(BigInteger256::new(x_limbs))
            .expect("zkMetal x coordinate should be valid BN254 Fq");
        let y = Fq::from_bigint(BigInteger256::new(y_limbs))
            .expect("zkMetal y coordinate should be valid BN254 Fq");

        ArkG1Affine::new(x, y, false)
    }
}

// ============================================================================
// G1Projective conversions
// ============================================================================

impl From<ArkG1Projective> for G1Projective {
    /// Convert `ark_bn254::G1Projective` to `zkmetal::msm::G1Projective`.
    fn from(ark_pt: ArkG1Projective) -> Self {
        let x_bigint: BigInteger256 = ark_pt.x.into();
        let y_bigint: BigInteger256 = ark_pt.y.into();
        let z_bigint: BigInteger256 = ark_pt.z.into();

        G1Projective {
            x: limbs_to_le_bytes(&x_bigint.0),
            y: limbs_to_le_bytes(&y_bigint.0),
            z: limbs_to_le_bytes(&z_bigint.0),
        }
    }
}

impl From<G1Projective> for ArkG1Projective {
    /// Convert `zkmetal::msm::G1Projective` to `ark_bn254::G1Projective`.
    fn from(zk_pt: G1Projective) -> Self {
        let x_limbs = le_bytes_to_limbs(&zk_pt.x);
        let y_limbs = le_bytes_to_limbs(&zk_pt.y);
        let z_limbs = le_bytes_to_limbs(&zk_pt.z);

        let x = Fq::from_bigint(BigInteger256::new(x_limbs))
            .expect("zkMetal X coordinate should be valid BN254 Fq");
        let y = Fq::from_bigint(BigInteger256::new(y_limbs))
            .expect("zkMetal Y coordinate should be valid BN254 Fq");
        let z = Fq::from_bigint(BigInteger256::new(z_limbs))
            .expect("zkMetal Z coordinate should be valid BN254 Fq");

        ArkG1Projective::new(x, y, z)
    }
}

// ============================================================================
// Byte <-> limb helpers
// ============================================================================

/// Convert 4 x u64 Montgomery limbs to 32-byte little-endian representation.
#[inline]
fn limbs_to_le_bytes(limbs: &[u64; 4]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (i, limb) in limbs.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    bytes
}

/// Convert 32-byte little-endian representation to 4 x u64 Montgomery limbs.
#[inline]
fn le_bytes_to_limbs(bytes: &[u8; 32]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for i in 0..4 {
        limbs[i] = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
    }
    limbs
}

// ============================================================================
// ArkMSM -- GPU MSM wrapper for arkworks types
// ============================================================================

/// GPU-accelerated Multi-Scalar Multiplication for arkworks BN254 types.
///
/// Converts arkworks points and scalars to zkMetal's byte-level representation,
/// dispatches to the Metal GPU, and converts the result back.
pub struct ArkMSM;

impl ArkMSM {
    /// Compute MSM: result = sum(scalars[i] * points[i]) on the Metal GPU.
    ///
    /// Uses the lazy singleton GPU engine (`_auto` API) so no explicit
    /// engine management is needed.
    ///
    /// # Panics
    ///
    /// Panics if `points.len() != scalars.len()`.
    #[cfg(feature = "gpu")]
    pub fn msm(
        points: &[ArkG1Affine],
        scalars: &[ArkFr],
    ) -> crate::Result<ArkG1Projective> {
        assert_eq!(
            points.len(),
            scalars.len(),
            "MSM requires equal number of points and scalars"
        );
        let n = points.len();
        if n == 0 {
            return Ok(ArkG1Projective::default());
        }

        // Convert points: arkworks G1Affine -> 64 bytes each (x || y, LE Montgomery).
        let mut point_bytes = vec![0u8; n * 64];
        for (i, pt) in points.iter().enumerate() {
            let zk_pt: G1Affine = (*pt).into();
            point_bytes[i * 64..i * 64 + 32].copy_from_slice(&zk_pt.x);
            point_bytes[i * 64 + 32..i * 64 + 64].copy_from_slice(&zk_pt.y);
        }

        // Convert scalars: arkworks Fr -> 32 bytes each (LE integer form, NOT Montgomery).
        // zkMetal GPU MSM expects scalars in standard (non-Montgomery) integer form.
        let mut scalar_bytes = vec![0u8; n * 32];
        for (i, s) in scalars.iter().enumerate() {
            let bigint: BigInteger256 = s.into_bigint();
            let bytes = limbs_to_le_bytes(&bigint.0);
            scalar_bytes[i * 32..(i + 1) * 32].copy_from_slice(&bytes);
        }

        let (rx, ry, rz) = crate::bn254_msm_auto(&point_bytes, &scalar_bytes, n as u32)?;

        let result = G1Projective {
            x: rx,
            y: ry,
            z: rz,
        };
        Ok(result.into())
    }

    /// Compute MSM using an explicit [`MsmEngine`](crate::MsmEngine).
    ///
    /// Useful when performing many MSMs to avoid singleton contention.
    #[cfg(feature = "gpu")]
    pub fn msm_with_engine(
        engine: &crate::MsmEngine,
        points: &[ArkG1Affine],
        scalars: &[ArkFr],
    ) -> crate::Result<ArkG1Projective> {
        assert_eq!(
            points.len(),
            scalars.len(),
            "MSM requires equal number of points and scalars"
        );
        let n = points.len();
        if n == 0 {
            return Ok(ArkG1Projective::default());
        }

        let mut point_bytes = vec![0u8; n * 64];
        for (i, pt) in points.iter().enumerate() {
            let zk_pt: G1Affine = (*pt).into();
            point_bytes[i * 64..i * 64 + 32].copy_from_slice(&zk_pt.x);
            point_bytes[i * 64 + 32..i * 64 + 64].copy_from_slice(&zk_pt.y);
        }

        let mut scalar_bytes = vec![0u8; n * 32];
        for (i, s) in scalars.iter().enumerate() {
            let bigint: BigInteger256 = s.into_bigint();
            let bytes = limbs_to_le_bytes(&bigint.0);
            scalar_bytes[i * 32..(i + 1) * 32].copy_from_slice(&bytes);
        }

        let result = engine.msm(&point_bytes, &scalar_bytes, n as u32)?;
        Ok(result.into())
    }
}

// ============================================================================
// ArkNTT -- GPU NTT wrapper for arkworks types
// ============================================================================

/// GPU-accelerated Number Theoretic Transform for arkworks BN254 Fr elements.
///
/// Converts between arkworks `Fr` slices and zkMetal's byte-level GPU NTT,
/// performing in-place forward and inverse transforms.
pub struct ArkNTT;

impl ArkNTT {
    /// Forward NTT in-place on a slice of arkworks Fr elements.
    ///
    /// `data.len()` must be a power of two.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not a power of two.
    #[cfg(feature = "gpu")]
    pub fn ntt(data: &mut [ArkFr]) -> crate::Result<()> {
        let n = data.len();
        assert!(n.is_power_of_two(), "NTT requires power-of-two length");
        let log_n = n.trailing_zeros();

        // Convert to byte representation: each Fr -> 32 bytes LE Montgomery.
        // arkworks Fr internal repr IS Montgomery, same as zkMetal.
        let mut bytes = ark_fr_slice_to_bytes(data);

        crate::bn254_ntt_auto(&mut bytes, log_n)?;

        // Convert back: 32 bytes LE Montgomery -> arkworks Fr.
        bytes_to_ark_fr_slice(&bytes, data);
        Ok(())
    }

    /// Inverse NTT in-place on a slice of arkworks Fr elements.
    ///
    /// `data.len()` must be a power of two.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not a power of two.
    #[cfg(feature = "gpu")]
    pub fn intt(data: &mut [ArkFr]) -> crate::Result<()> {
        let n = data.len();
        assert!(n.is_power_of_two(), "INTT requires power-of-two length");
        let log_n = n.trailing_zeros();

        let mut bytes = ark_fr_slice_to_bytes(data);

        crate::bn254_intt_auto(&mut bytes, log_n)?;

        bytes_to_ark_fr_slice(&bytes, data);
        Ok(())
    }

    /// Forward NTT using an explicit [`NttEngine`](crate::NttEngine).
    #[cfg(feature = "gpu")]
    pub fn ntt_with_engine(
        engine: &crate::NttEngine,
        data: &mut [ArkFr],
    ) -> crate::Result<()> {
        let n = data.len();
        assert!(n.is_power_of_two(), "NTT requires power-of-two length");
        let log_n = n.trailing_zeros();

        let mut bytes = ark_fr_slice_to_bytes(data);
        engine.ntt(&mut bytes, log_n)?;
        bytes_to_ark_fr_slice(&bytes, data);
        Ok(())
    }

    /// Inverse NTT using an explicit [`NttEngine`](crate::NttEngine).
    #[cfg(feature = "gpu")]
    pub fn intt_with_engine(
        engine: &crate::NttEngine,
        data: &mut [ArkFr],
    ) -> crate::Result<()> {
        let n = data.len();
        assert!(n.is_power_of_two(), "INTT requires power-of-two length");
        let log_n = n.trailing_zeros();

        let mut bytes = ark_fr_slice_to_bytes(data);
        engine.intt(&mut bytes, log_n)?;
        bytes_to_ark_fr_slice(&bytes, data);
        Ok(())
    }
}

// ============================================================================
// Batch conversion helpers
// ============================================================================

/// Convert a slice of arkworks Fr to a flat byte buffer (32 bytes per element).
///
/// The output is in little-endian Montgomery form, matching zkMetal's GPU format.
fn ark_fr_slice_to_bytes(data: &[ArkFr]) -> Vec<u8> {
    let mut bytes = vec![0u8; data.len() * 32];
    for (i, fr) in data.iter().enumerate() {
        // `.into()` on an arkworks Fp gives the Montgomery limbs as BigInteger256.
        let bigint: BigInteger256 = (*fr).into();
        let elem_bytes = limbs_to_le_bytes(&bigint.0);
        bytes[i * 32..(i + 1) * 32].copy_from_slice(&elem_bytes);
    }
    bytes
}

/// Convert a flat byte buffer back into a slice of arkworks Fr.
///
/// Reads 32 bytes per element in little-endian Montgomery form.
fn bytes_to_ark_fr_slice(bytes: &[u8], out: &mut [ArkFr]) {
    assert_eq!(bytes.len(), out.len() * 32);
    for (i, fr) in out.iter_mut().enumerate() {
        let limbs = le_bytes_to_limbs(
            bytes[i * 32..(i + 1) * 32]
                .try_into()
                .expect("slice is exactly 32 bytes"),
        );
        *fr = ArkFr::from_bigint(BigInteger256::new(limbs))
            .expect("bytes should be valid BN254 Fr");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fr_roundtrip() {
        // Test that Fr conversion is lossless.
        let ark_fr = ArkFr::from(42u64);
        let zk_fr: Fr = ark_fr.into();
        let back: ArkFr = zk_fr.into();
        assert_eq!(ark_fr, back);
    }

    #[test]
    fn test_fr_zero_roundtrip() {
        let ark_fr = ArkFr::from(0u64);
        let zk_fr: Fr = ark_fr.into();
        let back: ArkFr = zk_fr.into();
        assert_eq!(ark_fr, back);
    }

    #[test]
    fn test_g1affine_identity_roundtrip() {
        let ark_pt = ArkG1Affine::identity();
        let zk_pt: G1Affine = ark_pt.into();
        let back: ArkG1Affine = zk_pt.into();
        assert_eq!(ark_pt, back);
    }

    #[test]
    fn test_g1affine_generator_roundtrip() {
        use ark_ec::AffineCurve;
        let gen = ArkG1Affine::prime_subgroup_generator();
        let zk_pt: G1Affine = gen.into();
        let back: ArkG1Affine = zk_pt.into();
        assert_eq!(gen, back);
    }

    #[test]
    fn test_g1projective_generator_roundtrip() {
        use ark_ec::ProjectiveCurve;
        let gen = ArkG1Projective::prime_subgroup_generator();
        let zk_pt: G1Projective = gen.into();
        let back: ArkG1Projective = zk_pt.into();
        // Projective equality is up to scalar multiple of Z; compare via affine.
        use ark_ec::AffineCurve;
        assert_eq!(gen.into_affine(), back.into_affine());
    }
}
