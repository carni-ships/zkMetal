// zkMetal FFI — C-callable wrappers around zkMetal GPU engines
// Uses @_cdecl to export Swift functions with C linkage.

import Foundation
import Metal
import zkMetal

// MARK: - Constants

private let ZKMETAL_SUCCESS: Int32 = 0
private let ZKMETAL_ERR_NO_GPU: Int32 = -1
private let ZKMETAL_ERR_INVALID_INPUT: Int32 = -2
private let ZKMETAL_ERR_GPU_ERROR: Int32 = -3
private let ZKMETAL_ERR_ALLOC_FAILED: Int32 = -4

// Static version string — allocated once, never freed
private let versionBytes: [CChar] = Array("0.1.0".utf8CString)

// MARK: - Engine Lifecycle

@_cdecl("zkmetal_msm_engine_create")
public func zkmetal_msm_engine_create(_ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32 {
    do {
        let engine = try MetalMSM()
        let retained = Unmanaged.passRetained(engine).toOpaque()
        out.pointee = retained
        return ZKMETAL_SUCCESS
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

@_cdecl("zkmetal_msm_engine_destroy")
public func zkmetal_msm_engine_destroy(_ engine: UnsafeMutableRawPointer?) {
    guard let engine = engine else { return }
    Unmanaged<MetalMSM>.fromOpaque(engine).release()
}

@_cdecl("zkmetal_ntt_engine_create")
public func zkmetal_ntt_engine_create(_ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32 {
    do {
        let engine = try NTTEngine()
        let retained = Unmanaged.passRetained(engine).toOpaque()
        out.pointee = retained
        return ZKMETAL_SUCCESS
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

@_cdecl("zkmetal_ntt_engine_destroy")
public func zkmetal_ntt_engine_destroy(_ engine: UnsafeMutableRawPointer?) {
    guard let engine = engine else { return }
    Unmanaged<NTTEngine>.fromOpaque(engine).release()
}

@_cdecl("zkmetal_poseidon2_engine_create")
public func zkmetal_poseidon2_engine_create(_ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32 {
    do {
        let engine = try Poseidon2Engine()
        let retained = Unmanaged.passRetained(engine).toOpaque()
        out.pointee = retained
        return ZKMETAL_SUCCESS
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

@_cdecl("zkmetal_poseidon2_engine_destroy")
public func zkmetal_poseidon2_engine_destroy(_ engine: UnsafeMutableRawPointer?) {
    guard let engine = engine else { return }
    Unmanaged<Poseidon2Engine>.fromOpaque(engine).release()
}

// MARK: - MSM

@_cdecl("zkmetal_bn254_msm")
public func zkmetal_bn254_msm(
    _ engine: UnsafeMutableRawPointer,
    _ pointsPtr: UnsafePointer<UInt8>,
    _ scalarsPtr: UnsafePointer<UInt8>,
    _ nPoints: UInt32,
    _ resultX: UnsafeMutablePointer<UInt8>,
    _ resultY: UnsafeMutablePointer<UInt8>,
    _ resultZ: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let msm = Unmanaged<MetalMSM>.fromOpaque(engine).takeUnretainedValue()
    let n = Int(nPoints)

    if n == 0 {
        return ZKMETAL_ERR_INVALID_INPUT
    }

    // Reinterpret points bytes as PointAffine array (each is 2x Fp = 64 bytes)
    let pointSize = MemoryLayout<PointAffine>.stride
    let scalarLimbs = 8  // 8x UInt32 per scalar

    // Copy points from raw bytes
    var points = [PointAffine](repeating: PointAffine(x: Fp(v: (0,0,0,0,0,0,0,0)), y: Fp(v: (0,0,0,0,0,0,0,0))), count: n)
    _ = points.withUnsafeMutableBytes { dst in
        memcpy(dst.baseAddress!, pointsPtr, n * pointSize)
    }

    // Copy scalars: each scalar is 8x UInt32 = 32 bytes
    var scalars = [[UInt32]](repeating: [UInt32](repeating: 0, count: scalarLimbs), count: n)
    scalarsPtr.withMemoryRebound(to: UInt32.self, capacity: n * scalarLimbs) { scalarSrc in
        for i in 0..<n {
            for j in 0..<scalarLimbs {
                scalars[i][j] = scalarSrc[i * scalarLimbs + j]
            }
        }
    }

    do {
        let result = try msm.msm(points: points, scalars: scalars)

        // Copy result projective coordinates out
        _ = withUnsafeBytes(of: result.x) { src in
            memcpy(resultX, src.baseAddress!, 32)
        }
        _ = withUnsafeBytes(of: result.y) { src in
            memcpy(resultY, src.baseAddress!, 32)
        }
        _ = withUnsafeBytes(of: result.z) { src in
            memcpy(resultZ, src.baseAddress!, 32)
        }

        return ZKMETAL_SUCCESS
    } catch MSMError.invalidInput {
        return ZKMETAL_ERR_INVALID_INPUT
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

// MARK: - NTT

@_cdecl("zkmetal_bn254_ntt")
public func zkmetal_bn254_ntt(
    _ engine: UnsafeMutableRawPointer,
    _ dataPtr: UnsafeMutablePointer<UInt8>,
    _ logN: UInt32
) -> Int32 {
    let ntt = Unmanaged<NTTEngine>.fromOpaque(engine).takeUnretainedValue()
    let n = 1 << Int(logN)
    let elemSize = MemoryLayout<Fr>.stride  // 32 bytes

    // Copy data in
    var data = [Fr](repeating: Fr(v: (0,0,0,0,0,0,0,0)), count: n)
    _ = data.withUnsafeMutableBytes { dst in
        memcpy(dst.baseAddress!, dataPtr, n * elemSize)
    }

    do {
        let result = try ntt.ntt(data)
        _ = result.withUnsafeBytes { src in
            memcpy(dataPtr, src.baseAddress!, n * elemSize)
        }
        return ZKMETAL_SUCCESS
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

@_cdecl("zkmetal_bn254_intt")
public func zkmetal_bn254_intt(
    _ engine: UnsafeMutableRawPointer,
    _ dataPtr: UnsafeMutablePointer<UInt8>,
    _ logN: UInt32
) -> Int32 {
    let ntt = Unmanaged<NTTEngine>.fromOpaque(engine).takeUnretainedValue()
    let n = 1 << Int(logN)
    let elemSize = MemoryLayout<Fr>.stride

    var data = [Fr](repeating: Fr(v: (0,0,0,0,0,0,0,0)), count: n)
    _ = data.withUnsafeMutableBytes { dst in
        memcpy(dst.baseAddress!, dataPtr, n * elemSize)
    }

    do {
        let result = try ntt.intt(data)
        _ = result.withUnsafeBytes { src in
            memcpy(dataPtr, src.baseAddress!, n * elemSize)
        }
        return ZKMETAL_SUCCESS
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

// MARK: - Poseidon2

@_cdecl("zkmetal_bn254_poseidon2_hash_pairs")
public func zkmetal_bn254_poseidon2_hash_pairs(
    _ engine: UnsafeMutableRawPointer,
    _ inputPtr: UnsafePointer<UInt8>,
    _ nPairs: UInt32,
    _ outputPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let p2 = Unmanaged<Poseidon2Engine>.fromOpaque(engine).takeUnretainedValue()
    let n = Int(nPairs)
    let elemSize = MemoryLayout<Fr>.stride

    if n == 0 {
        return ZKMETAL_ERR_INVALID_INPUT
    }

    // Copy 2*n elements in
    var input = [Fr](repeating: Fr(v: (0,0,0,0,0,0,0,0)), count: 2 * n)
    _ = input.withUnsafeMutableBytes { dst in
        memcpy(dst.baseAddress!, inputPtr, 2 * n * elemSize)
    }

    do {
        let result = try p2.hashPairs(input)
        _ = result.withUnsafeBytes { src in
            memcpy(outputPtr, src.baseAddress!, n * elemSize)
        }
        return ZKMETAL_SUCCESS
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

// MARK: - Utility

@_cdecl("zkmetal_set_shader_dir")
public func zkmetal_set_shader_dir(_ path: UnsafePointer<CChar>?) {
    if let path = path {
        let dir = String(cString: path)
        setenv("ZKMETAL_SHADER_DIR", dir, 1)
    } else {
        unsetenv("ZKMETAL_SHADER_DIR")
    }
}

@_cdecl("zkmetal_gpu_available")
public func zkmetal_gpu_available() -> Int32 {
    return MTLCreateSystemDefaultDevice() != nil ? 1 : 0
}

@_cdecl("zkmetal_version")
public func zkmetal_version() -> UnsafePointer<CChar> {
    return versionBytes.withUnsafeBufferPointer { $0.baseAddress! }
}
