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
private let versionBytes: [CChar] = Array("0.2.0".utf8CString)

// MARK: - Lazy Singleton Engines

/// Thread-safe lazy singletons for convenience (one-shot) API.
/// Engines are created on first use and persist for the process lifetime.
private class LazyEngines {
    static let shared = LazyEngines()

    private var _msm: MetalMSM?
    private var _ntt: NTTEngine?
    private var _poseidon2: Poseidon2Engine?
    private var _keccak: Keccak256Engine?
    private var _fri: FRIEngine?
    private let lock = NSLock()

    func msm() throws -> MetalMSM {
        lock.lock()
        defer { lock.unlock() }
        if let e = _msm { return e }
        let e = try MetalMSM()
        _msm = e
        return e
    }

    func ntt() throws -> NTTEngine {
        lock.lock()
        defer { lock.unlock() }
        if let e = _ntt { return e }
        let e = try NTTEngine()
        _ntt = e
        return e
    }

    func poseidon2() throws -> Poseidon2Engine {
        lock.lock()
        defer { lock.unlock() }
        if let e = _poseidon2 { return e }
        let e = try Poseidon2Engine()
        _poseidon2 = e
        return e
    }

    func keccak() throws -> Keccak256Engine {
        lock.lock()
        defer { lock.unlock() }
        if let e = _keccak { return e }
        let e = try Keccak256Engine()
        _keccak = e
        return e
    }

    func fri() throws -> FRIEngine {
        lock.lock()
        defer { lock.unlock() }
        if let e = _fri { return e }
        let e = try FRIEngine()
        _fri = e
        return e
    }
}

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

@_cdecl("zkmetal_keccak_engine_create")
public func zkmetal_keccak_engine_create(_ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32 {
    do {
        let engine = try Keccak256Engine()
        let retained = Unmanaged.passRetained(engine).toOpaque()
        out.pointee = retained
        return ZKMETAL_SUCCESS
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

@_cdecl("zkmetal_keccak_engine_destroy")
public func zkmetal_keccak_engine_destroy(_ engine: UnsafeMutableRawPointer?) {
    guard let engine = engine else { return }
    Unmanaged<Keccak256Engine>.fromOpaque(engine).release()
}

@_cdecl("zkmetal_fri_engine_create")
public func zkmetal_fri_engine_create(_ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32 {
    do {
        let engine = try FRIEngine()
        let retained = Unmanaged.passRetained(engine).toOpaque()
        out.pointee = retained
        return ZKMETAL_SUCCESS
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

@_cdecl("zkmetal_fri_engine_destroy")
public func zkmetal_fri_engine_destroy(_ engine: UnsafeMutableRawPointer?) {
    guard let engine = engine else { return }
    Unmanaged<FRIEngine>.fromOpaque(engine).release()
}

// MARK: - MSM (engine-based)

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
    return _msm_impl(msm, pointsPtr, scalarsPtr, nPoints, resultX, resultY, resultZ)
}

// MARK: - MSM (convenience — lazy singleton)

@_cdecl("zkmetal_bn254_msm_auto")
public func zkmetal_bn254_msm_auto(
    _ pointsPtr: UnsafePointer<UInt8>,
    _ scalarsPtr: UnsafePointer<UInt8>,
    _ nPoints: UInt32,
    _ resultX: UnsafeMutablePointer<UInt8>,
    _ resultY: UnsafeMutablePointer<UInt8>,
    _ resultZ: UnsafeMutablePointer<UInt8>
) -> Int32 {
    do {
        let msm = try LazyEngines.shared.msm()
        return _msm_impl(msm, pointsPtr, scalarsPtr, nPoints, resultX, resultY, resultZ)
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

private func _msm_impl(
    _ msm: MetalMSM,
    _ pointsPtr: UnsafePointer<UInt8>,
    _ scalarsPtr: UnsafePointer<UInt8>,
    _ nPoints: UInt32,
    _ resultX: UnsafeMutablePointer<UInt8>,
    _ resultY: UnsafeMutablePointer<UInt8>,
    _ resultZ: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let n = Int(nPoints)
    if n == 0 { return ZKMETAL_ERR_INVALID_INPUT }

    let pointSize = MemoryLayout<PointAffine>.stride
    let scalarLimbs = 8

    var points = [PointAffine](repeating: PointAffine(x: Fp(v: (0,0,0,0,0,0,0,0)), y: Fp(v: (0,0,0,0,0,0,0,0))), count: n)
    _ = points.withUnsafeMutableBytes { dst in
        memcpy(dst.baseAddress!, pointsPtr, n * pointSize)
    }

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
        _ = withUnsafeBytes(of: result.x) { src in memcpy(resultX, src.baseAddress!, 32) }
        _ = withUnsafeBytes(of: result.y) { src in memcpy(resultY, src.baseAddress!, 32) }
        _ = withUnsafeBytes(of: result.z) { src in memcpy(resultZ, src.baseAddress!, 32) }
        return ZKMETAL_SUCCESS
    } catch MSMError.invalidInput {
        return ZKMETAL_ERR_INVALID_INPUT
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

// MARK: - NTT (engine-based)

@_cdecl("zkmetal_bn254_ntt")
public func zkmetal_bn254_ntt(
    _ engine: UnsafeMutableRawPointer,
    _ dataPtr: UnsafeMutablePointer<UInt8>,
    _ logN: UInt32
) -> Int32 {
    let ntt = Unmanaged<NTTEngine>.fromOpaque(engine).takeUnretainedValue()
    return _ntt_impl(ntt, dataPtr, logN, inverse: false)
}

@_cdecl("zkmetal_bn254_intt")
public func zkmetal_bn254_intt(
    _ engine: UnsafeMutableRawPointer,
    _ dataPtr: UnsafeMutablePointer<UInt8>,
    _ logN: UInt32
) -> Int32 {
    let ntt = Unmanaged<NTTEngine>.fromOpaque(engine).takeUnretainedValue()
    return _ntt_impl(ntt, dataPtr, logN, inverse: true)
}

// MARK: - NTT (convenience — lazy singleton)

@_cdecl("zkmetal_bn254_ntt_auto")
public func zkmetal_bn254_ntt_auto(
    _ dataPtr: UnsafeMutablePointer<UInt8>,
    _ logN: UInt32
) -> Int32 {
    do {
        let ntt = try LazyEngines.shared.ntt()
        return _ntt_impl(ntt, dataPtr, logN, inverse: false)
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

@_cdecl("zkmetal_bn254_intt_auto")
public func zkmetal_bn254_intt_auto(
    _ dataPtr: UnsafeMutablePointer<UInt8>,
    _ logN: UInt32
) -> Int32 {
    do {
        let ntt = try LazyEngines.shared.ntt()
        return _ntt_impl(ntt, dataPtr, logN, inverse: true)
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

private func _ntt_impl(
    _ ntt: NTTEngine,
    _ dataPtr: UnsafeMutablePointer<UInt8>,
    _ logN: UInt32,
    inverse: Bool
) -> Int32 {
    let n = 1 << Int(logN)
    let elemSize = MemoryLayout<Fr>.stride

    var data = [Fr](repeating: Fr(v: (0,0,0,0,0,0,0,0)), count: n)
    _ = data.withUnsafeMutableBytes { dst in
        memcpy(dst.baseAddress!, dataPtr, n * elemSize)
    }

    do {
        let result = inverse ? try ntt.intt(data) : try ntt.ntt(data)
        _ = result.withUnsafeBytes { src in
            memcpy(dataPtr, src.baseAddress!, n * elemSize)
        }
        return ZKMETAL_SUCCESS
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

// MARK: - Poseidon2 (engine-based)

@_cdecl("zkmetal_bn254_poseidon2_hash_pairs")
public func zkmetal_bn254_poseidon2_hash_pairs(
    _ engine: UnsafeMutableRawPointer,
    _ inputPtr: UnsafePointer<UInt8>,
    _ nPairs: UInt32,
    _ outputPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let p2 = Unmanaged<Poseidon2Engine>.fromOpaque(engine).takeUnretainedValue()
    return _poseidon2_impl(p2, inputPtr, nPairs, outputPtr)
}

// MARK: - Poseidon2 (convenience — lazy singleton)

@_cdecl("zkmetal_bn254_poseidon2_hash_pairs_auto")
public func zkmetal_bn254_poseidon2_hash_pairs_auto(
    _ inputPtr: UnsafePointer<UInt8>,
    _ nPairs: UInt32,
    _ outputPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    do {
        let p2 = try LazyEngines.shared.poseidon2()
        return _poseidon2_impl(p2, inputPtr, nPairs, outputPtr)
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

private func _poseidon2_impl(
    _ p2: Poseidon2Engine,
    _ inputPtr: UnsafePointer<UInt8>,
    _ nPairs: UInt32,
    _ outputPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let n = Int(nPairs)
    let elemSize = MemoryLayout<Fr>.stride

    if n == 0 { return ZKMETAL_ERR_INVALID_INPUT }

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

// MARK: - Keccak-256 (engine-based)

@_cdecl("zkmetal_keccak256_hash")
public func zkmetal_keccak256_hash(
    _ engine: UnsafeMutableRawPointer,
    _ inputPtr: UnsafePointer<UInt8>,
    _ nInputs: UInt32,
    _ outputPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let keccak = Unmanaged<Keccak256Engine>.fromOpaque(engine).takeUnretainedValue()
    return _keccak_impl(keccak, inputPtr, nInputs, outputPtr)
}

// MARK: - Keccak-256 (convenience — lazy singleton)

@_cdecl("zkmetal_keccak256_hash_auto")
public func zkmetal_keccak256_hash_auto(
    _ inputPtr: UnsafePointer<UInt8>,
    _ nInputs: UInt32,
    _ outputPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    do {
        let keccak = try LazyEngines.shared.keccak()
        return _keccak_impl(keccak, inputPtr, nInputs, outputPtr)
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

private func _keccak_impl(
    _ keccak: Keccak256Engine,
    _ inputPtr: UnsafePointer<UInt8>,
    _ nInputs: UInt32,
    _ outputPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let n = Int(nInputs)
    if n == 0 { return ZKMETAL_ERR_INVALID_INPUT }

    // Each input is 64 bytes, each output is 32 bytes
    let input = Array(UnsafeBufferPointer(start: inputPtr, count: n * 64))

    do {
        let result = try keccak.hash64(input)
        result.withUnsafeBytes { src in
            memcpy(outputPtr, src.baseAddress!, n * 32)
        }
        return ZKMETAL_SUCCESS
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

// MARK: - FRI Fold (engine-based)

@_cdecl("zkmetal_fri_fold")
public func zkmetal_fri_fold(
    _ engine: UnsafeMutableRawPointer,
    _ evalsPtr: UnsafePointer<UInt8>,
    _ logN: UInt32,
    _ betaPtr: UnsafePointer<UInt8>,
    _ resultPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let fri = Unmanaged<FRIEngine>.fromOpaque(engine).takeUnretainedValue()
    return _fri_fold_impl(fri, evalsPtr, logN, betaPtr, resultPtr)
}

// MARK: - FRI Fold (convenience — lazy singleton)

@_cdecl("zkmetal_fri_fold_auto")
public func zkmetal_fri_fold_auto(
    _ evalsPtr: UnsafePointer<UInt8>,
    _ logN: UInt32,
    _ betaPtr: UnsafePointer<UInt8>,
    _ resultPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    do {
        let fri = try LazyEngines.shared.fri()
        return _fri_fold_impl(fri, evalsPtr, logN, betaPtr, resultPtr)
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

private func _fri_fold_impl(
    _ fri: FRIEngine,
    _ evalsPtr: UnsafePointer<UInt8>,
    _ logN: UInt32,
    _ betaPtr: UnsafePointer<UInt8>,
    _ resultPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let n = 1 << Int(logN)
    let half = n / 2
    let elemSize = MemoryLayout<Fr>.stride

    if logN < 1 { return ZKMETAL_ERR_INVALID_INPUT }

    // Read evaluations
    var evals = [Fr](repeating: Fr(v: (0,0,0,0,0,0,0,0)), count: n)
    _ = evals.withUnsafeMutableBytes { dst in
        memcpy(dst.baseAddress!, evalsPtr, n * elemSize)
    }

    // Read beta challenge
    var beta = Fr(v: (0,0,0,0,0,0,0,0))
    _ = withUnsafeMutableBytes(of: &beta) { dst in
        memcpy(dst.baseAddress!, betaPtr, elemSize)
    }

    do {
        let result = try fri.fold(evals: evals, beta: beta)
        _ = result.withUnsafeBytes { src in
            memcpy(resultPtr, src.baseAddress!, half * elemSize)
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
