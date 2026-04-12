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
    private var _pallasMsm: PallasMSM?
    private var _vestaMsm: VestaMSM?
    private var _ntt: NTTEngine?
    private var _poseidon2: Poseidon2Engine?
    private var _keccak: Keccak256Engine?
    private var _fri: FRIEngine?
    private var _pairing: BN254PairingEngine?
    private let lock = NSLock()

    func pairing() throws -> BN254PairingEngine {
        lock.lock()
        defer { lock.unlock() }
        if let e = _pairing { return e }
        let e = try BN254PairingEngine()
        _pairing = e
        return e
    }

    func msm() throws -> MetalMSM {
        lock.lock()
        defer { lock.unlock() }
        if let e = _msm { return e }
        let e = try MetalMSM()
        _msm = e
        return e
    }

    func pallasMsm() throws -> PallasMSM {
        lock.lock()
        defer { lock.unlock() }
        if let e = _pallasMsm { return e }
        let e = try PallasMSM()
        _pallasMsm = e
        return e
    }

    func vestaMsm() throws -> VestaMSM {
        lock.lock()
        defer { lock.unlock() }
        if let e = _vestaMsm { return e }
        let e = try VestaMSM()
        _vestaMsm = e
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

@_cdecl("zkmetal_pallas_msm_engine_create")
public func zkmetal_pallas_msm_engine_create(_ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32 {
    do {
        let engine = try PallasMSM()
        let retained = Unmanaged.passRetained(engine).toOpaque()
        out.pointee = retained
        return ZKMETAL_SUCCESS
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

@_cdecl("zkmetal_pallas_msm_engine_destroy")
public func zkmetal_pallas_msm_engine_destroy(_ engine: UnsafeMutableRawPointer?) {
    guard let engine = engine else { return }
    Unmanaged<PallasMSM>.fromOpaque(engine).release()
}

@_cdecl("zkmetal_vesta_msm_engine_create")
public func zkmetal_vesta_msm_engine_create(_ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32 {
    do {
        let engine = try VestaMSM()
        let retained = Unmanaged.passRetained(engine).toOpaque()
        out.pointee = retained
        return ZKMETAL_SUCCESS
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

@_cdecl("zkmetal_vesta_msm_engine_destroy")
public func zkmetal_vesta_msm_engine_destroy(_ engine: UnsafeMutableRawPointer?) {
    guard let engine = engine else { return }
    Unmanaged<VestaMSM>.fromOpaque(engine).release()
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

@_cdecl("zkmetal_pairing_engine_create")
public func zkmetal_pairing_engine_create(_ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32 {
    do {
        let engine = try BN254PairingEngine()
        let retained = Unmanaged.passRetained(engine).toOpaque()
        out.pointee = retained
        return ZKMETAL_SUCCESS
    } catch PairingError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

@_cdecl("zkmetal_pairing_engine_destroy")
public func zkmetal_pairing_engine_destroy(_ engine: UnsafeMutableRawPointer?) {
    guard let engine = engine else { return }
    Unmanaged<BN254PairingEngine>.fromOpaque(engine).release()
}

// MARK: - Pasta NTT Engine Lifecycle (CPU kernels — no GPU engine needed)

@_cdecl("zkmetal_pasta_ntt_engine_create")
public func zkmetal_pasta_ntt_engine_create(_ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32 {
    // Pasta NTT uses CPU kernels from pasta_ntt.c; no GPU engine needed.
    // Return a dummy non-null pointer to satisfy the C API contract.
    let dummy = UnsafeMutableRawPointer(bitPattern: 0x1)!
    out.pointee = dummy
    return ZKMETAL_SUCCESS
}

@_cdecl("zkmetal_pasta_ntt_engine_destroy")
public func zkmetal_pasta_ntt_engine_destroy(_ engine: UnsafeMutableRawPointer?) {
    // No-op: CPU kernels have no mutable engine state to clean up.
}

// MARK: - Pasta Poseidon Engine Lifecycle (CPU kernels — no GPU engine needed)

@_cdecl("zkmetal_pasta_poseidon_engine_create")
public func zkmetal_pasta_poseidon_engine_create(_ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32 {
    // Pasta Poseidon uses CPU kernels from pasta_poseidon.c; no GPU engine needed.
    let dummy = UnsafeMutableRawPointer(bitPattern: 0x2)!
    out.pointee = dummy
    return ZKMETAL_SUCCESS
}

@_cdecl("zkmetal_pasta_poseidon_engine_destroy")
public func zkmetal_pasta_poseidon_engine_destroy(_ engine: UnsafeMutableRawPointer?) {
    // No-op: CPU kernels have no mutable engine state to clean up.
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

// MARK: - Pallas MSM (engine-based)

@_cdecl("zkmetal_pallas_msm")
public func zkmetal_pallas_msm(
    _ engine: UnsafeMutableRawPointer,
    _ pointsPtr: UnsafePointer<UInt8>,
    _ scalarsPtr: UnsafePointer<UInt8>,
    _ nPoints: UInt32,
    _ resultX: UnsafeMutablePointer<UInt8>,
    _ resultY: UnsafeMutablePointer<UInt8>,
    _ resultZ: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let msm = Unmanaged<PallasMSM>.fromOpaque(engine).takeUnretainedValue()
    return _pallas_msm_impl(msm, pointsPtr, scalarsPtr, nPoints, resultX, resultY, resultZ)
}

// MARK: - Pallas MSM (convenience — lazy singleton)

@_cdecl("zkmetal_pallas_msm_auto")
public func zkmetal_pallas_msm_auto(
    _ pointsPtr: UnsafePointer<UInt8>,
    _ scalarsPtr: UnsafePointer<UInt8>,
    _ nPoints: UInt32,
    _ resultX: UnsafeMutablePointer<UInt8>,
    _ resultY: UnsafeMutablePointer<UInt8>,
    _ resultZ: UnsafeMutablePointer<UInt8>
) -> Int32 {
    do {
        let msm = try LazyEngines.shared.pallasMsm()
        return _pallas_msm_impl(msm, pointsPtr, scalarsPtr, nPoints, resultX, resultY, resultZ)
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

private func _pallas_msm_impl(
    _ msm: PallasMSM,
    _ pointsPtr: UnsafePointer<UInt8>,
    _ scalarsPtr: UnsafePointer<UInt8>,
    _ nPoints: UInt32,
    _ resultX: UnsafeMutablePointer<UInt8>,
    _ resultY: UnsafeMutablePointer<UInt8>,
    _ resultZ: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let n = Int(nPoints)
    if n == 0 { return ZKMETAL_ERR_INVALID_INPUT }

    let pointSize = MemoryLayout<PallasPointAffine>.stride
    let scalarLimbs = 8

    var points = [PallasPointAffine](repeating: PallasPointAffine(x: PallasFp(v: (0,0,0,0,0,0,0,0)), y: PallasFp(v: (0,0,0,0,0,0,0,0))), count: n)
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

// MARK: - Vesta MSM (engine-based)

@_cdecl("zkmetal_vesta_msm")
public func zkmetal_vesta_msm(
    _ engine: UnsafeMutableRawPointer,
    _ pointsPtr: UnsafePointer<UInt8>,
    _ scalarsPtr: UnsafePointer<UInt8>,
    _ nPoints: UInt32,
    _ resultX: UnsafeMutablePointer<UInt8>,
    _ resultY: UnsafeMutablePointer<UInt8>,
    _ resultZ: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let msm = Unmanaged<VestaMSM>.fromOpaque(engine).takeUnretainedValue()
    return _vesta_msm_impl(msm, pointsPtr, scalarsPtr, nPoints, resultX, resultY, resultZ)
}

// MARK: - Vesta MSM (convenience — lazy singleton)

@_cdecl("zkmetal_vesta_msm_auto")
public func zkmetal_vesta_msm_auto(
    _ pointsPtr: UnsafePointer<UInt8>,
    _ scalarsPtr: UnsafePointer<UInt8>,
    _ nPoints: UInt32,
    _ resultX: UnsafeMutablePointer<UInt8>,
    _ resultY: UnsafeMutablePointer<UInt8>,
    _ resultZ: UnsafeMutablePointer<UInt8>
) -> Int32 {
    do {
        let msm = try LazyEngines.shared.vestaMsm()
        return _vesta_msm_impl(msm, pointsPtr, scalarsPtr, nPoints, resultX, resultY, resultZ)
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

private func _vesta_msm_impl(
    _ msm: VestaMSM,
    _ pointsPtr: UnsafePointer<UInt8>,
    _ scalarsPtr: UnsafePointer<UInt8>,
    _ nPoints: UInt32,
    _ resultX: UnsafeMutablePointer<UInt8>,
    _ resultY: UnsafeMutablePointer<UInt8>,
    _ resultZ: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let n = Int(nPoints)
    if n == 0 { return ZKMETAL_ERR_INVALID_INPUT }

    let pointSize = MemoryLayout<VestaPointAffine>.stride
    let scalarLimbs = 8

    var points = [VestaPointAffine](repeating: VestaPointAffine(x: VestaFp(v: (0,0,0,0,0,0,0,0)), y: VestaFp(v: (0,0,0,0,0,0,0,0))), count: n)
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

// MARK: - Pasta NTT (CPU kernels — direct C FFI)

// External C kernels from pasta_ntt.c
@_silgen_name("pallas_fr_ntt")
private func _c_pallas_fr_ntt(_ data: UnsafeMutablePointer<UInt64>, _ logN: Int32)

@_silgen_name("pallas_fr_intt")
private func _c_pallas_fr_intt(_ data: UnsafeMutablePointer<UInt64>, _ logN: Int32)

@_silgen_name("vesta_fr_ntt")
private func _c_vesta_fr_ntt(_ data: UnsafeMutablePointer<UInt64>, _ logN: Int32)

@_silgen_name("vesta_fr_intt")
private func _c_vesta_fr_intt(_ data: UnsafeMutablePointer<UInt64>, _ logN: Int32)

@_cdecl("zkmetal_pallas_ntt_auto")
public func zkmetal_pallas_ntt_auto(
    _ dataPtr: UnsafeMutablePointer<UInt8>,
    _ logN: UInt32
) -> Int32 {
    if logN < 1 { return ZKMETAL_ERR_INVALID_INPUT }
    dataPtr.withMemoryRebound(to: UInt64.self, capacity: (1 << logN) * 4) { ptr in
        _c_pallas_fr_ntt(ptr, Int32(logN))
    }
    return ZKMETAL_SUCCESS
}

@_cdecl("zkmetal_pallas_intt_auto")
public func zkmetal_pallas_intt_auto(
    _ dataPtr: UnsafeMutablePointer<UInt8>,
    _ logN: UInt32
) -> Int32 {
    if logN < 1 { return ZKMETAL_ERR_INVALID_INPUT }
    dataPtr.withMemoryRebound(to: UInt64.self, capacity: (1 << logN) * 4) { ptr in
        _c_pallas_fr_intt(ptr, Int32(logN))
    }
    return ZKMETAL_SUCCESS
}

@_cdecl("zkmetal_vesta_ntt_auto")
public func zkmetal_vesta_ntt_auto(
    _ dataPtr: UnsafeMutablePointer<UInt8>,
    _ logN: UInt32
) -> Int32 {
    if logN < 1 { return ZKMETAL_ERR_INVALID_INPUT }
    dataPtr.withMemoryRebound(to: UInt64.self, capacity: (1 << logN) * 4) { ptr in
        _c_vesta_fr_ntt(ptr, Int32(logN))
    }
    return ZKMETAL_SUCCESS
}

@_cdecl("zkmetal_vesta_intt_auto")
public func zkmetal_vesta_intt_auto(
    _ dataPtr: UnsafeMutablePointer<UInt8>,
    _ logN: UInt32
) -> Int32 {
    if logN < 1 { return ZKMETAL_ERR_INVALID_INPUT }
    dataPtr.withMemoryRebound(to: UInt64.self, capacity: (1 << logN) * 4) { ptr in
        _c_vesta_fr_intt(ptr, Int32(logN))
    }
    return ZKMETAL_SUCCESS
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

// MARK: - Pasta Poseidon Permutation (CPU kernels — direct C FFI)

// External C kernels from pasta_poseidon.c
@_silgen_name("pallas_poseidon_permutation_cpu")
private func _c_pallas_poseidon_permutation_cpu(_ state: UnsafePointer<UInt64>, _ result: UnsafeMutablePointer<UInt64>)

@_silgen_name("vesta_poseidon_permutation_cpu")
private func _c_vesta_poseidon_permutation_cpu(_ state: UnsafePointer<UInt64>, _ result: UnsafeMutablePointer<UInt64>)

@_cdecl("zkmetal_pallas_poseidon_permutation_auto")
public func zkmetal_pallas_poseidon_permutation_auto(
    _ statePtr: UnsafePointer<UInt8>,
    _ resultPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    // state: 12 * 8 = 96 bytes (3 field elements × 4 uint64_t each)
    statePtr.withMemoryRebound(to: UInt64.self, capacity: 12) { stateSrc in
        resultPtr.withMemoryRebound(to: UInt64.self, capacity: 12) { resultDst in
            _c_pallas_poseidon_permutation_cpu(stateSrc, resultDst)
        }
    }
    return ZKMETAL_SUCCESS
}

@_cdecl("zkmetal_vesta_poseidon_permutation_auto")
public func zkmetal_vesta_poseidon_permutation_auto(
    _ statePtr: UnsafePointer<UInt8>,
    _ resultPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    // state: 12 * 8 = 96 bytes (3 field elements × 4 uint64_t each)
    statePtr.withMemoryRebound(to: UInt64.self, capacity: 12) { stateSrc in
        resultPtr.withMemoryRebound(to: UInt64.self, capacity: 12) { resultDst in
            _c_vesta_poseidon_permutation_cpu(stateSrc, resultDst)
        }
    }
    return ZKMETAL_SUCCESS
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

// MARK: - Batch Pairing (engine-based)

@_cdecl("zkmetal_bn254_batch_pairing")
public func zkmetal_bn254_batch_pairing(
    _ engine: UnsafeMutableRawPointer,
    _ g1Ptr: UnsafePointer<UInt8>,
    _ g2Ptr: UnsafePointer<UInt8>,
    _ nPairs: UInt32,
    _ resultPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let pairing = Unmanaged<BN254PairingEngine>.fromOpaque(engine).takeUnretainedValue()
    return _batch_pairing_impl(pairing, g1Ptr, g2Ptr, nPairs, resultPtr)
}

@_cdecl("zkmetal_bn254_pairing_check")
public func zkmetal_bn254_pairing_check(
    _ engine: UnsafeMutableRawPointer,
    _ g1Ptr: UnsafePointer<UInt8>,
    _ g2Ptr: UnsafePointer<UInt8>,
    _ nPairs: UInt32
) -> Int32 {
    let pairing = Unmanaged<BN254PairingEngine>.fromOpaque(engine).takeUnretainedValue()
    return _pairing_check_impl(pairing, g1Ptr, g2Ptr, nPairs)
}

// MARK: - Batch Pairing (convenience — lazy singleton)

@_cdecl("zkmetal_bn254_batch_pairing_auto")
public func zkmetal_bn254_batch_pairing_auto(
    _ g1Ptr: UnsafePointer<UInt8>,
    _ g2Ptr: UnsafePointer<UInt8>,
    _ nPairs: UInt32,
    _ resultPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    do {
        let pairing = try LazyEngines.shared.pairing()
        return _batch_pairing_impl(pairing, g1Ptr, g2Ptr, nPairs, resultPtr)
    } catch PairingError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

@_cdecl("zkmetal_bn254_pairing_check_auto")
public func zkmetal_bn254_pairing_check_auto(
    _ g1Ptr: UnsafePointer<UInt8>,
    _ g2Ptr: UnsafePointer<UInt8>,
    _ nPairs: UInt32
) -> Int32 {
    do {
        let pairing = try LazyEngines.shared.pairing()
        return _pairing_check_impl(pairing, g1Ptr, g2Ptr, nPairs)
    } catch PairingError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

private func _decode_pairing_pairs(
    _ g1Ptr: UnsafePointer<UInt8>,
    _ g2Ptr: UnsafePointer<UInt8>,
    _ nPairs: UInt32
) -> [(PointAffine, G2AffinePoint)] {
    let n = Int(nPairs)
    var pairs = [(PointAffine, G2AffinePoint)]()
    pairs.reserveCapacity(n)

    let pointSize = MemoryLayout<PointAffine>.stride  // 64 bytes
    // G2 affine: 4 Fp elements = 128 bytes (x.c0, x.c1, y.c0, y.c1)

    for i in 0..<n {
        // Read G1 point (64 bytes = x,y each 32 bytes)
        var g1 = PointAffine(x: Fp(v: (0,0,0,0,0,0,0,0)), y: Fp(v: (0,0,0,0,0,0,0,0)))
        withUnsafeMutableBytes(of: &g1) { dst in
            memcpy(dst.baseAddress!, g1Ptr.advanced(by: i * pointSize), pointSize)
        }

        // Read G2 point (128 bytes = x0,x1,y0,y1 each 32 bytes)
        let g2Base = g2Ptr.advanced(by: i * 128)
        func readFp(_ offset: Int) -> Fp {
            var fp = Fp(v: (0,0,0,0,0,0,0,0))
            withUnsafeMutableBytes(of: &fp) { dst in
                memcpy(dst.baseAddress!, g2Base.advanced(by: offset), 32)
            }
            return fp
        }
        let g2 = G2AffinePoint(
            x: Fp2(c0: readFp(0), c1: readFp(32)),
            y: Fp2(c0: readFp(64), c1: readFp(96))
        )

        pairs.append((g1, g2))
    }
    return pairs
}

private func _batch_pairing_impl(
    _ pairing: BN254PairingEngine,
    _ g1Ptr: UnsafePointer<UInt8>,
    _ g2Ptr: UnsafePointer<UInt8>,
    _ nPairs: UInt32,
    _ resultPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let n = Int(nPairs)
    if n == 0 { return ZKMETAL_ERR_INVALID_INPUT }

    let pairs = _decode_pairing_pairs(g1Ptr, g2Ptr, nPairs)

    do {
        let result = try pairing.multiMillerPairing(pairs: pairs)
        // Fp12 = 12 Fp elements = 384 bytes. Write c0 (Fp6) then c1 (Fp6).
        // Fp6 = c0,c1,c2 (Fp2 each). Fp2 = c0,c1 (Fp each).
        var offset = 0
        func writeFp(_ fp: Fp) {
            withUnsafeBytes(of: fp) { src in
                memcpy(resultPtr.advanced(by: offset), src.baseAddress!, 32)
            }
            offset += 32
        }
        func writeFp2(_ fp2: Fp2) { writeFp(fp2.c0); writeFp(fp2.c1) }
        func writeFp6(_ fp6: Fp6) { writeFp2(fp6.c0); writeFp2(fp6.c1); writeFp2(fp6.c2) }
        writeFp6(result.c0)
        writeFp6(result.c1)
        return ZKMETAL_SUCCESS
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

private func _pairing_check_impl(
    _ pairing: BN254PairingEngine,
    _ g1Ptr: UnsafePointer<UInt8>,
    _ g2Ptr: UnsafePointer<UInt8>,
    _ nPairs: UInt32
) -> Int32 {
    let n = Int(nPairs)
    if n == 0 { return ZKMETAL_ERR_INVALID_INPUT }

    let pairs = _decode_pairing_pairs(g1Ptr, g2Ptr, nPairs)

    do {
        let passed = try pairing.pairingCheck(pairs: pairs)
        return passed ? ZKMETAL_SUCCESS : ZKMETAL_ERR_INVALID_INPUT
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

// MARK: - Small-Scalar MSM Variants (engine-based)

@_cdecl("zkmetal_bn254_msm_u8")
public func zkmetal_bn254_msm_u8(
    _ engine: UnsafeMutableRawPointer,
    _ pointsPtr: UnsafePointer<UInt8>,
    _ scalarsPtr: UnsafePointer<UInt8>,
    _ nPoints: UInt32,
    _ resultX: UnsafeMutablePointer<UInt8>,
    _ resultY: UnsafeMutablePointer<UInt8>,
    _ resultZ: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let msm = Unmanaged<MetalMSM>.fromOpaque(engine).takeUnretainedValue()
    return _msm_small_scalar_impl(msm, pointsPtr, scalarsPtr, nPoints, 1, resultX, resultY, resultZ)
}

@_cdecl("zkmetal_bn254_msm_u16")
public func zkmetal_bn254_msm_u16(
    _ engine: UnsafeMutableRawPointer,
    _ pointsPtr: UnsafePointer<UInt8>,
    _ scalarsPtr: UnsafePointer<UInt8>,
    _ nPoints: UInt32,
    _ resultX: UnsafeMutablePointer<UInt8>,
    _ resultY: UnsafeMutablePointer<UInt8>,
    _ resultZ: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let msm = Unmanaged<MetalMSM>.fromOpaque(engine).takeUnretainedValue()
    return _msm_small_scalar_impl(msm, pointsPtr, scalarsPtr, nPoints, 2, resultX, resultY, resultZ)
}

@_cdecl("zkmetal_bn254_msm_u32")
public func zkmetal_bn254_msm_u32(
    _ engine: UnsafeMutableRawPointer,
    _ pointsPtr: UnsafePointer<UInt8>,
    _ scalarsPtr: UnsafePointer<UInt8>,
    _ nPoints: UInt32,
    _ resultX: UnsafeMutablePointer<UInt8>,
    _ resultY: UnsafeMutablePointer<UInt8>,
    _ resultZ: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let msm = Unmanaged<MetalMSM>.fromOpaque(engine).takeUnretainedValue()
    return _msm_small_scalar_impl(msm, pointsPtr, scalarsPtr, nPoints, 4, resultX, resultY, resultZ)
}

// MARK: - Small-Scalar MSM Variants (convenience — lazy singleton)

@_cdecl("zkmetal_bn254_msm_u8_auto")
public func zkmetal_bn254_msm_u8_auto(
    _ pointsPtr: UnsafePointer<UInt8>,
    _ scalarsPtr: UnsafePointer<UInt8>,
    _ nPoints: UInt32,
    _ resultX: UnsafeMutablePointer<UInt8>,
    _ resultY: UnsafeMutablePointer<UInt8>,
    _ resultZ: UnsafeMutablePointer<UInt8>
) -> Int32 {
    do {
        let msm = try LazyEngines.shared.msm()
        return _msm_small_scalar_impl(msm, pointsPtr, scalarsPtr, nPoints, 1, resultX, resultY, resultZ)
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

@_cdecl("zkmetal_bn254_msm_u16_auto")
public func zkmetal_bn254_msm_u16_auto(
    _ pointsPtr: UnsafePointer<UInt8>,
    _ scalarsPtr: UnsafePointer<UInt8>,
    _ nPoints: UInt32,
    _ resultX: UnsafeMutablePointer<UInt8>,
    _ resultY: UnsafeMutablePointer<UInt8>,
    _ resultZ: UnsafeMutablePointer<UInt8>
) -> Int32 {
    do {
        let msm = try LazyEngines.shared.msm()
        return _msm_small_scalar_impl(msm, pointsPtr, scalarsPtr, nPoints, 2, resultX, resultY, resultZ)
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

@_cdecl("zkmetal_bn254_msm_u32_auto")
public func zkmetal_bn254_msm_u32_auto(
    _ pointsPtr: UnsafePointer<UInt8>,
    _ scalarsPtr: UnsafePointer<UInt8>,
    _ nPoints: UInt32,
    _ resultX: UnsafeMutablePointer<UInt8>,
    _ resultY: UnsafeMutablePointer<UInt8>,
    _ resultZ: UnsafeMutablePointer<UInt8>
) -> Int32 {
    do {
        let msm = try LazyEngines.shared.msm()
        return _msm_small_scalar_impl(msm, pointsPtr, scalarsPtr, nPoints, 4, resultX, resultY, resultZ)
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

private func _msm_small_scalar_impl(
    _ msm: MetalMSM,
    _ pointsPtr: UnsafePointer<UInt8>,
    _ scalarsPtr: UnsafePointer<UInt8>,
    _ nPoints: UInt32,
    _ scalarBytes: Int,
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

    // Convert small scalars to [[UInt32]] format (8 limbs, zero-extended)
    var scalars = [[UInt32]](repeating: [UInt32](repeating: 0, count: scalarLimbs), count: n)
    for i in 0..<n {
        var val: UInt32 = 0
        switch scalarBytes {
        case 1:
            val = UInt32(scalarsPtr[i])
        case 2:
            val = UInt32(scalarsPtr[i * 2]) | (UInt32(scalarsPtr[i * 2 + 1]) << 8)
        case 4:
            val = UInt32(scalarsPtr[i * 4])
                | (UInt32(scalarsPtr[i * 4 + 1]) << 8)
                | (UInt32(scalarsPtr[i * 4 + 2]) << 16)
                | (UInt32(scalarsPtr[i * 4 + 3]) << 24)
        default:
            break
        }
        scalars[i][0] = val
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

// MARK: - Batch MSM (engine-based)

@_cdecl("zkmetal_bn254_msm_batch")
public func zkmetal_bn254_msm_batch(
    _ engine: UnsafeMutableRawPointer,
    _ allPointsPtr: UnsafePointer<UInt8>,
    _ allScalarsPtr: UnsafePointer<UInt8>,
    _ countsPtr: UnsafePointer<UInt32>,
    _ nMsms: UInt32,
    _ resultsPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let msm = Unmanaged<MetalMSM>.fromOpaque(engine).takeUnretainedValue()
    return _msm_batch_impl(msm, allPointsPtr, allScalarsPtr, countsPtr, nMsms, resultsPtr)
}

// MARK: - Batch MSM (convenience — lazy singleton)

@_cdecl("zkmetal_bn254_msm_batch_auto")
public func zkmetal_bn254_msm_batch_auto(
    _ allPointsPtr: UnsafePointer<UInt8>,
    _ allScalarsPtr: UnsafePointer<UInt8>,
    _ countsPtr: UnsafePointer<UInt32>,
    _ nMsms: UInt32,
    _ resultsPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    do {
        let msm = try LazyEngines.shared.msm()
        return _msm_batch_impl(msm, allPointsPtr, allScalarsPtr, countsPtr, nMsms, resultsPtr)
    } catch MSMError.noGPU {
        return ZKMETAL_ERR_NO_GPU
    } catch {
        return ZKMETAL_ERR_GPU_ERROR
    }
}

private func _msm_batch_impl(
    _ msm: MetalMSM,
    _ allPointsPtr: UnsafePointer<UInt8>,
    _ allScalarsPtr: UnsafePointer<UInt8>,
    _ countsPtr: UnsafePointer<UInt32>,
    _ nMsms: UInt32,
    _ resultsPtr: UnsafeMutablePointer<UInt8>
) -> Int32 {
    let numMsms = Int(nMsms)
    if numMsms == 0 { return ZKMETAL_ERR_INVALID_INPUT }

    let pointSize = MemoryLayout<PointAffine>.stride
    let scalarLimbs = 8

    var pointOffset = 0
    var scalarOffset = 0

    for m in 0..<numMsms {
        let count = Int(countsPtr[m])
        if count == 0 {
            // Write identity point (zero projective)
            memset(resultsPtr.advanced(by: m * 96), 0, 96)
            continue
        }

        var points = [PointAffine](repeating: PointAffine(x: Fp(v: (0,0,0,0,0,0,0,0)), y: Fp(v: (0,0,0,0,0,0,0,0))), count: count)
        _ = points.withUnsafeMutableBytes { dst in
            memcpy(dst.baseAddress!, allPointsPtr.advanced(by: pointOffset), count * pointSize)
        }

        var scalars = [[UInt32]](repeating: [UInt32](repeating: 0, count: scalarLimbs), count: count)
        let scalarSrc = allScalarsPtr.advanced(by: scalarOffset)
        scalarSrc.withMemoryRebound(to: UInt32.self, capacity: count * scalarLimbs) { src in
            for i in 0..<count {
                for j in 0..<scalarLimbs {
                    scalars[i][j] = src[i * scalarLimbs + j]
                }
            }
        }

        do {
            let result = try msm.msm(points: points, scalars: scalars)
            let dst = resultsPtr.advanced(by: m * 96)
            _ = withUnsafeBytes(of: result.x) { src in memcpy(dst, src.baseAddress!, 32) }
            _ = withUnsafeBytes(of: result.y) { src in memcpy(dst.advanced(by: 32), src.baseAddress!, 32) }
            _ = withUnsafeBytes(of: result.z) { src in memcpy(dst.advanced(by: 64), src.baseAddress!, 32) }
        } catch MSMError.invalidInput {
            return ZKMETAL_ERR_INVALID_INPUT
        } catch {
            return ZKMETAL_ERR_GPU_ERROR
        }

        pointOffset += count * pointSize
        scalarOffset += count * 32
    }

    return ZKMETAL_SUCCESS
}

// MARK: - Pasta Endo-Combine (CPU C kernel, batch g1 + g2.scale(scalar))

// External C kernel for pasta endo-combine
@_silgen_name("batch_pallas_endo_combine")
private func _c_batch_pallas_endo_combine(
    _ g1_x: UnsafePointer<UInt64>,
    _ g1_y: UnsafePointer<UInt64>,
    _ g2_x: UnsafePointer<UInt64>,
    _ g2_y: UnsafePointer<UInt64>,
    _ endo_coeff: UnsafePointer<UInt64>,
    _ scalars: UnsafePointer<UInt64>,
    _ count: UInt32,
    _ result_x: UnsafeMutablePointer<UInt64>,
    _ result_y: UnsafeMutablePointer<UInt64>
)

@_silgen_name("batch_vesta_endo_combine")
private func _c_batch_vesta_endo_combine(
    _ g1_x: UnsafePointer<UInt64>,
    _ g1_y: UnsafePointer<UInt64>,
    _ g2_x: UnsafePointer<UInt64>,
    _ g2_y: UnsafePointer<UInt64>,
    _ endo_coeff: UnsafePointer<UInt64>,
    _ scalars: UnsafePointer<UInt64>,
    _ count: UInt32,
    _ result_x: UnsafeMutablePointer<UInt64>,
    _ result_y: UnsafeMutablePointer<UInt64>
)

@_cdecl("zkmetal_pallas_endo_combine_auto")
public func zkmetal_pallas_endo_combine_auto(
    _ g1_x: UnsafePointer<UInt8>,
    _ g1_y: UnsafePointer<UInt8>,
    _ g2_x: UnsafePointer<UInt8>,
    _ g2_y: UnsafePointer<UInt8>,
    _ endo_coeff: UnsafePointer<UInt8>,
    _ scalars: UnsafePointer<UInt8>,
    _ count: UInt32,
    _ result_x: UnsafeMutablePointer<UInt8>,
    _ result_y: UnsafeMutablePointer<UInt8>
) -> Int32 {
    if count == 0 { return ZKMETAL_SUCCESS }

    // Compute buffer sizes: g1_x/g1_y/g2_x/g2_y: count * 32 bytes (4 u64 each)
    // scalars: count * 64 bytes (8 u64 each)
    // result_x/result_y: count * 32 bytes (4 u64 each)
    let count32 = Int(count)
    let g1_x_ptr = g1_x.withMemoryRebound(to: UInt64.self, capacity: count32 * 4) { $0 }
    let g1_y_ptr = g1_y.withMemoryRebound(to: UInt64.self, capacity: count32 * 4) { $0 }
    let g2_x_ptr = g2_x.withMemoryRebound(to: UInt64.self, capacity: count32 * 4) { $0 }
    let g2_y_ptr = g2_y.withMemoryRebound(to: UInt64.self, capacity: count32 * 4) { $0 }
    let endo_ptr = endo_coeff.withMemoryRebound(to: UInt64.self, capacity: 4) { $0 }
    let scalars_ptr = scalars.withMemoryRebound(to: UInt64.self, capacity: count32 * 8) { $0 }
    let result_x_ptr = result_x.withMemoryRebound(to: UInt64.self, capacity: count32 * 4) { $0 }
    let result_y_ptr = result_y.withMemoryRebound(to: UInt64.self, capacity: count32 * 4) { $0 }

    _c_batch_pallas_endo_combine(
        g1_x_ptr, g1_y_ptr, g2_x_ptr, g2_y_ptr,
        endo_ptr, scalars_ptr, count,
        result_x_ptr, result_y_ptr
    )
    return ZKMETAL_SUCCESS
}

@_cdecl("zkmetal_vesta_endo_combine_auto")
public func zkmetal_vesta_endo_combine_auto(
    _ g1_x: UnsafePointer<UInt8>,
    _ g1_y: UnsafePointer<UInt8>,
    _ g2_x: UnsafePointer<UInt8>,
    _ g2_y: UnsafePointer<UInt8>,
    _ endo_coeff: UnsafePointer<UInt8>,
    _ scalars: UnsafePointer<UInt8>,
    _ count: UInt32,
    _ result_x: UnsafeMutablePointer<UInt8>,
    _ result_y: UnsafeMutablePointer<UInt8>
) -> Int32 {
    if count == 0 { return ZKMETAL_SUCCESS }

    let count32 = Int(count)
    let g1_x_ptr = g1_x.withMemoryRebound(to: UInt64.self, capacity: count32 * 4) { $0 }
    let g1_y_ptr = g1_y.withMemoryRebound(to: UInt64.self, capacity: count32 * 4) { $0 }
    let g2_x_ptr = g2_x.withMemoryRebound(to: UInt64.self, capacity: count32 * 4) { $0 }
    let g2_y_ptr = g2_y.withMemoryRebound(to: UInt64.self, capacity: count32 * 4) { $0 }
    let endo_ptr = endo_coeff.withMemoryRebound(to: UInt64.self, capacity: 4) { $0 }
    let scalars_ptr = scalars.withMemoryRebound(to: UInt64.self, capacity: count32 * 8) { $0 }
    let result_x_ptr = result_x.withMemoryRebound(to: UInt64.self, capacity: count32 * 4) { $0 }
    let result_y_ptr = result_y.withMemoryRebound(to: UInt64.self, capacity: count32 * 4) { $0 }

    _c_batch_vesta_endo_combine(
        g1_x_ptr, g1_y_ptr, g2_x_ptr, g2_y_ptr,
        endo_ptr, scalars_ptr, count,
        result_x_ptr, result_y_ptr
    )
    return ZKMETAL_SUCCESS
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
