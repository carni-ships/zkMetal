// Parallel CPU Pippenger MSM for Pallas and Vesta curves.
// Calls C implementations in pasta_ops.c via NeonFieldOps.
// Provides massive speedup over naive Swift scalar-mul loops (100-1000x).

import Foundation
import NeonFieldOps

/// CPU Pippenger MSM for Pallas curve (points in Pallas Fp, scalars in Vesta Fp / Pallas Fr).
/// Points: affine PallasPointAffine (8xUInt32 x, 8xUInt32 y in Montgomery form).
/// Scalars: VestaFp values (converted to integer form internally).
/// Returns: PallasPointProjective result.
public func pallasCpuMSM(points: [PallasPointAffine], scalars: [VestaFp]) -> PallasPointProjective {
    let n = points.count
    precondition(n == scalars.count)
    if n == 0 { return pallasPointIdentity() }

    // Convert affine points to flat uint64_t array (n x 8 uint64_t)
    var flatPoints = [UInt64](repeating: 0, count: n * 8)
    for i in 0..<n {
        let x64 = points[i].x.to64()
        let y64 = points[i].y.to64()
        flatPoints[i * 8 + 0] = x64[0]
        flatPoints[i * 8 + 1] = x64[1]
        flatPoints[i * 8 + 2] = x64[2]
        flatPoints[i * 8 + 3] = x64[3]
        flatPoints[i * 8 + 4] = y64[0]
        flatPoints[i * 8 + 5] = y64[1]
        flatPoints[i * 8 + 6] = y64[2]
        flatPoints[i * 8 + 7] = y64[3]
    }

    // Convert scalars to flat uint32_t array (n x 8 uint32_t, integer form)
    // Scalars are VestaFp (Montgomery form) -- need to convert to integer form
    var flatScalars = [UInt32](repeating: 0, count: n * 8)
    for i in 0..<n {
        let intVal = vestaToInt(scalars[i])  // [UInt64] x 4, integer form
        flatScalars[i * 8 + 0] = UInt32(intVal[0] & 0xFFFFFFFF)
        flatScalars[i * 8 + 1] = UInt32(intVal[0] >> 32)
        flatScalars[i * 8 + 2] = UInt32(intVal[1] & 0xFFFFFFFF)
        flatScalars[i * 8 + 3] = UInt32(intVal[1] >> 32)
        flatScalars[i * 8 + 4] = UInt32(intVal[2] & 0xFFFFFFFF)
        flatScalars[i * 8 + 5] = UInt32(intVal[2] >> 32)
        flatScalars[i * 8 + 6] = UInt32(intVal[3] & 0xFFFFFFFF)
        flatScalars[i * 8 + 7] = UInt32(intVal[3] >> 32)
    }

    // Call C Pippenger MSM
    var result = [UInt64](repeating: 0, count: 12)
    pallas_pippenger_msm(&flatPoints, &flatScalars, Int32(n), &result)

    // Convert result back to PallasPointProjective
    return PallasPointProjective(
        x: PallasFp.from64([result[0], result[1], result[2], result[3]]),
        y: PallasFp.from64([result[4], result[5], result[6], result[7]]),
        z: PallasFp.from64([result[8], result[9], result[10], result[11]])
    )
}

/// CPU Pippenger MSM for Vesta curve (points in Vesta Fp, scalars in Pallas Fp / Vesta Fr).
public func vestaCpuMSM(points: [VestaPointAffine], scalars: [PallasFp]) -> VestaPointProjective {
    let n = points.count
    precondition(n == scalars.count)
    if n == 0 { return vestaPointIdentity() }

    // Convert affine points to flat uint64_t array (n x 8 uint64_t)
    var flatPoints = [UInt64](repeating: 0, count: n * 8)
    for i in 0..<n {
        let x64 = points[i].x.to64()
        let y64 = points[i].y.to64()
        flatPoints[i * 8 + 0] = x64[0]
        flatPoints[i * 8 + 1] = x64[1]
        flatPoints[i * 8 + 2] = x64[2]
        flatPoints[i * 8 + 3] = x64[3]
        flatPoints[i * 8 + 4] = y64[0]
        flatPoints[i * 8 + 5] = y64[1]
        flatPoints[i * 8 + 6] = y64[2]
        flatPoints[i * 8 + 7] = y64[3]
    }

    // Convert scalars to flat uint32_t array (n x 8 uint32_t, integer form)
    var flatScalars = [UInt32](repeating: 0, count: n * 8)
    for i in 0..<n {
        let intVal = pallasToInt(scalars[i])
        flatScalars[i * 8 + 0] = UInt32(intVal[0] & 0xFFFFFFFF)
        flatScalars[i * 8 + 1] = UInt32(intVal[0] >> 32)
        flatScalars[i * 8 + 2] = UInt32(intVal[1] & 0xFFFFFFFF)
        flatScalars[i * 8 + 3] = UInt32(intVal[1] >> 32)
        flatScalars[i * 8 + 4] = UInt32(intVal[2] & 0xFFFFFFFF)
        flatScalars[i * 8 + 5] = UInt32(intVal[2] >> 32)
        flatScalars[i * 8 + 6] = UInt32(intVal[3] & 0xFFFFFFFF)
        flatScalars[i * 8 + 7] = UInt32(intVal[3] >> 32)
    }

    // Call C Pippenger MSM
    var result = [UInt64](repeating: 0, count: 12)
    vesta_pippenger_msm(&flatPoints, &flatScalars, Int32(n), &result)

    // Convert result back to VestaPointProjective
    return VestaPointProjective(
        x: VestaFp.from64([result[0], result[1], result[2], result[3]]),
        y: VestaFp.from64([result[4], result[5], result[6], result[7]]),
        z: VestaFp.from64([result[8], result[9], result[10], result[11]])
    )
}

/// Variant taking pre-extracted UInt32 scalar limbs (avoids Montgomery->int conversion).
public func pallasCpuMSM(points: [PallasPointAffine], scalarLimbs: [[UInt32]]) -> PallasPointProjective {
    let n = points.count
    precondition(n == scalarLimbs.count)
    if n == 0 { return pallasPointIdentity() }

    var flatPoints = [UInt64](repeating: 0, count: n * 8)
    for i in 0..<n {
        let x64 = points[i].x.to64()
        let y64 = points[i].y.to64()
        for j in 0..<4 { flatPoints[i * 8 + j] = x64[j] }
        for j in 0..<4 { flatPoints[i * 8 + 4 + j] = y64[j] }
    }

    var flatScalars = [UInt32](repeating: 0, count: n * 8)
    for i in 0..<n {
        for j in 0..<8 { flatScalars[i * 8 + j] = scalarLimbs[i][j] }
    }

    var result = [UInt64](repeating: 0, count: 12)
    pallas_pippenger_msm(&flatPoints, &flatScalars, Int32(n), &result)

    return PallasPointProjective(
        x: PallasFp.from64([result[0], result[1], result[2], result[3]]),
        y: PallasFp.from64([result[4], result[5], result[6], result[7]]),
        z: PallasFp.from64([result[8], result[9], result[10], result[11]])
    )
}

/// Variant taking pre-extracted UInt32 scalar limbs for Vesta.
public func vestaCpuMSM(points: [VestaPointAffine], scalarLimbs: [[UInt32]]) -> VestaPointProjective {
    let n = points.count
    precondition(n == scalarLimbs.count)
    if n == 0 { return vestaPointIdentity() }

    var flatPoints = [UInt64](repeating: 0, count: n * 8)
    for i in 0..<n {
        let x64 = points[i].x.to64()
        let y64 = points[i].y.to64()
        for j in 0..<4 { flatPoints[i * 8 + j] = x64[j] }
        for j in 0..<4 { flatPoints[i * 8 + 4 + j] = y64[j] }
    }

    var flatScalars = [UInt32](repeating: 0, count: n * 8)
    for i in 0..<n {
        for j in 0..<8 { flatScalars[i * 8 + j] = scalarLimbs[i][j] }
    }

    var result = [UInt64](repeating: 0, count: 12)
    vesta_pippenger_msm(&flatPoints, &flatScalars, Int32(n), &result)

    return VestaPointProjective(
        x: VestaFp.from64([result[0], result[1], result[2], result[3]]),
        y: VestaFp.from64([result[4], result[5], result[6], result[7]]),
        z: VestaFp.from64([result[8], result[9], result[10], result[11]])
    )
}
