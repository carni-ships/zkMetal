// Auto-tuning system for zkMetal
// Runs a quick calibration sweep on first use, caches results per GPU device.
// Re-calibrates automatically when the GPU changes.

import Foundation
import Metal

/// Tuning parameters discovered by calibration.
public struct TuningConfig: Codable {
    public var deviceName: String
    public var nttThreadgroupSize: Int       // threadgroup cap for NTT butterfly/bitrev kernels
    public var nttFourStepThreshold: Int     // min global stages to trigger four-step FFT
    public var msmThreadgroupSize: Int       // threadgroup cap for MSM kernels
    public var msmWindowBitsLarge: Int       // window bits for large point counts (>32K)
    public var hashThreadgroupSize: Int      // threadgroup cap for hash kernels
    public var friThreadgroupSize: Int       // threadgroup cap for FRI fold kernels
    public var sumcheckFusedTGSize: Int      // threads per threadgroup in fused sumcheck
    public var sumcheckPerRoundTGSize: Int   // threads per threadgroup in per-round sumcheck

    /// Default M3 Pro-tuned values (fallback if calibration fails).
    public static let defaults = TuningConfig(
        deviceName: "unknown",
        nttThreadgroupSize: 256,
        nttFourStepThreshold: 10,
        msmThreadgroupSize: 256,
        msmWindowBitsLarge: 16,
        hashThreadgroupSize: 256,  // was 64; synthetic XOR cal biased low — Poseidon2 S-box is register-heavy
        friThreadgroupSize: 256,
        sumcheckFusedTGSize: 128,
        sumcheckPerRoundTGSize: 256
    )
}

/// Manages calibration and caching of tuning parameters.
public class TuningManager {
    public static let shared = TuningManager()

    private static let cacheDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".zkmetal")
    private static let cacheFile = cacheDir.appendingPathComponent("tuning.json")

    private var _config: TuningConfig?
    private let lock = NSLock()

    /// Get the current tuning config, running calibration if needed.
    public func config(device: MTLDevice) -> TuningConfig {
        lock.lock()
        defer { lock.unlock() }

        if let c = _config, c.deviceName == device.name {
            return c
        }

        // Try loading from cache
        if let cached = loadCache(), cached.deviceName == device.name {
            _config = cached
            return cached
        }

        // Run calibration
        let calibrated = calibrate(device: device)
        _config = calibrated
        saveCache(calibrated)
        return calibrated
    }

    /// Force re-calibration (e.g., after a Metal driver update).
    public func recalibrate(device: MTLDevice) -> TuningConfig {
        lock.lock()
        defer { lock.unlock() }
        let calibrated = calibrate(device: device)
        _config = calibrated
        saveCache(calibrated)
        return calibrated
    }

    // MARK: - Calibration

    private func calibrate(device: MTLDevice) -> TuningConfig {
        print("[zkMetal] Calibrating for \(device.name)...")
        let start = CFAbsoluteTimeGetCurrent()

        var config = TuningConfig.defaults
        config.deviceName = device.name

        // Calibrate NTT threadgroup size using a simple butterfly-like workload
        if let nttTG = calibrateThreadgroupSize(device: device, workloadType: .ntt) {
            config.nttThreadgroupSize = nttTG
        }

        // Calibrate hash threadgroup size
        if let hashTG = calibrateThreadgroupSize(device: device, workloadType: .hash) {
            config.hashThreadgroupSize = hashTG
        }

        // Calibrate MSM threadgroup size
        if let msmTG = calibrateThreadgroupSize(device: device, workloadType: .msm) {
            config.msmThreadgroupSize = msmTG
        }

        // FRI and sumcheck share similar patterns — use NTT result as baseline
        config.friThreadgroupSize = config.nttThreadgroupSize
        config.sumcheckPerRoundTGSize = config.nttThreadgroupSize

        // Sumcheck fused TG: try half of the main TG size since fused kernels use more registers
        config.sumcheckFusedTGSize = max(64, config.nttThreadgroupSize / 2)

        // NTT four-step threshold: on devices with fewer GPU cores, four-step overhead
        // may not pay off until later. Use core heuristic.
        let maxThreads = device.maxThreadsPerThreadgroup.width
        if maxThreads >= 1024 {
            config.nttFourStepThreshold = 10  // high-end: four-step at >=2^20
        } else {
            config.nttFourStepThreshold = 12  // lower-end: delay four-step
        }

        // MSM window bits: larger GPUs benefit from wider windows
        if maxThreads >= 1024 {
            config.msmWindowBitsLarge = 16
        } else {
            config.msmWindowBitsLarge = 14
        }

        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        print("[zkMetal] Calibration complete in \(String(format: "%.0f", elapsed))ms")
        return config
    }

    private enum WorkloadType {
        case ntt, hash, msm
    }

    /// Sweep threadgroup sizes [32, 64, 128, 256, 512, 1024] on a synthetic Metal kernel.
    /// Returns the best threadgroup size, or nil if calibration fails.
    private func calibrateThreadgroupSize(device: MTLDevice, workloadType: WorkloadType) -> Int? {
        // Use a simple buffer-copy kernel to test dispatch overhead vs. occupancy
        let shaderSource: String
        switch workloadType {
        case .ntt:
            // Simulate field multiply-heavy workload (multiply-accumulate)
            shaderSource = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void calibrate_ntt(device uint *data [[buffer(0)]],
                                      uint gid [[thread_position_in_grid]]) {
                uint v = data[gid];
                for (int i = 0; i < 64; i++) { v = v * 0x12345678u + v; }
                data[gid] = v;
            }
            """
        case .hash:
            // Simulate hash-like workload (register-heavy XOR chains)
            shaderSource = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void calibrate_hash(device uint *data [[buffer(0)]],
                                       uint gid [[thread_position_in_grid]]) {
                uint a = data[gid], b = a ^ 0xDEADBEEF, c = a + 0xCAFEBABE;
                for (int i = 0; i < 64; i++) { a ^= b + c; b ^= a + c; c ^= a + b; }
                data[gid] = a ^ b ^ c;
            }
            """
        case .msm:
            // Simulate MSM-like workload (memory + compute mixed)
            shaderSource = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void calibrate_msm(device uint *data [[buffer(0)]],
                                      uint gid [[thread_position_in_grid]]) {
                uint idx = gid;
                uint v = data[idx];
                for (int i = 0; i < 32; i++) {
                    idx = (idx * 2654435761u) & 0xFFFF;
                    v += data[idx];
                }
                data[gid] = v;
            }
            """
        }

        let funcName: String
        switch workloadType {
        case .ntt: funcName = "calibrate_ntt"
        case .hash: funcName = "calibrate_hash"
        case .msm: funcName = "calibrate_msm"
        }

        guard let library = try? device.makeLibrary(source: shaderSource, options: nil),
              let function = library.makeFunction(name: funcName),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let queue = device.makeCommandQueue() else {
            return nil
        }

        let n = 1 << 16  // 64K elements
        guard let buf = device.makeBuffer(length: n * MemoryLayout<UInt32>.stride,
                                          options: .storageModeShared) else {
            return nil
        }
        // Fill buffer
        let ptr = buf.contents().bindMemory(to: UInt32.self, capacity: n)
        for i in 0..<n { ptr[i] = UInt32(i) }

        let maxTG = min(Int(pipeline.maxTotalThreadsPerThreadgroup), 1024)
        let candidates = [32, 64, 128, 256, 512, 1024].filter { $0 <= maxTG }

        var bestTG = 256
        var bestTime = Double.infinity

        // Warmup
        for _ in 0..<2 {
            guard let cmd = queue.makeCommandBuffer() else { return nil }
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(buf, offset: 0, index: 0)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        for tgSize in candidates {
            var times = [Double]()
            for _ in 0..<5 {
                guard let cmd = queue.makeCommandBuffer() else { return nil }
                let enc = cmd.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(buf, offset: 0, index: 0)
                enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
                times.append(cmd.gpuEndTime - cmd.gpuStartTime)
            }
            times.sort()
            let median = times[2]
            if median < bestTime {
                bestTime = median
                bestTG = tgSize
            }
        }

        return bestTG
    }

    // MARK: - Cache I/O

    private func loadCache() -> TuningConfig? {
        guard let data = try? Data(contentsOf: TuningManager.cacheFile),
              let config = try? JSONDecoder().decode(TuningConfig.self, from: data) else {
            return nil
        }
        return config
    }

    private func saveCache(_ config: TuningConfig) {
        let fm = FileManager.default
        try? fm.createDirectory(at: TuningManager.cacheDir, withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(config) {
            try? data.write(to: TuningManager.cacheFile)
        }
    }
}
