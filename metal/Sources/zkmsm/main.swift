// zkmsm — Metal GPU Multi-Scalar Multiplication for BN254
//
// CLI tool that performs MSM on the GPU using Metal compute shaders.
// Input: JSON on stdin with points (affine) and scalars (256-bit).
// Output: JSON on stdout with the resulting point.
//
// Usage:
//   echo '{"points": [...], "scalars": [...]}' | zkmsm
//   zkmsm --bench <n_points>

import Foundation
import Metal

// MARK: - Types matching Metal shader structs

struct Fp {
    var v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)

    static var zero: Fp {
        Fp(v: (0, 0, 0, 0, 0, 0, 0, 0))
    }

    init(v: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)) {
        self.v = v
    }

    init(from bytes: [UInt8]) {
        // Little-endian bytes to 32-bit limbs
        var limbs: [UInt32] = Array(repeating: 0, count: 8)
        for i in 0..<min(32, bytes.count) {
            limbs[i / 4] |= UInt32(bytes[i]) << ((i % 4) * 8)
        }
        self.v = (limbs[0], limbs[1], limbs[2], limbs[3],
                  limbs[4], limbs[5], limbs[6], limbs[7])
    }

    func toBytes() -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: 32)
        let limbs = [v.0, v.1, v.2, v.3, v.4, v.5, v.6, v.7]
        for i in 0..<8 {
            bytes[i * 4 + 0] = UInt8(limbs[i] & 0xFF)
            bytes[i * 4 + 1] = UInt8((limbs[i] >> 8) & 0xFF)
            bytes[i * 4 + 2] = UInt8((limbs[i] >> 16) & 0xFF)
            bytes[i * 4 + 3] = UInt8((limbs[i] >> 24) & 0xFF)
        }
        return bytes
    }
}

struct PointAffine {
    var x: Fp
    var y: Fp
}

struct PointProjective {
    var x: Fp
    var y: Fp
    var z: Fp
}

struct MsmParams {
    var n_points: UInt32
    var window_bits: UInt32
    var window_index: UInt32
}

// MARK: - Metal MSM Engine

class MetalMSM {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let accumulateFunction: MTLComputePipelineState
    let reduceFunction: MTLComputePipelineState
    let bucketSumFunction: MTLComputePipelineState

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        // Compile Metal shader from source
        let shaderPath = CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "--shader"
            ? CommandLine.arguments[2]
            : findShaderPath()

        let shaderSource = try String(contentsOfFile: shaderPath, encoding: .utf8)
        let library = try device.makeLibrary(source: shaderSource, options: nil)

        guard let accFn = library.makeFunction(name: "msm_accumulate"),
              let redFn = library.makeFunction(name: "msm_reduce_buckets"),
              let sumFn = library.makeFunction(name: "msm_bucket_sum") else {
            throw MSMError.missingKernel
        }

        self.accumulateFunction = try device.makeComputePipelineState(function: accFn)
        self.reduceFunction = try device.makeComputePipelineState(function: redFn)
        self.bucketSumFunction = try device.makeComputePipelineState(function: sumFn)
    }

    func msm(points: [PointAffine], scalars: [[UInt32]]) throws -> PointProjective {
        let n = points.count
        guard n == scalars.count, n > 0 else {
            throw MSMError.invalidInput
        }

        let windowBits: UInt32 = n <= 256 ? 8 : (n <= 4096 ? 12 : 16)
        let nWindows = (256 + Int(windowBits) - 1) / Int(windowBits)
        let nBuckets = 1 << windowBits

        // Allocate GPU buffers
        let pointsBuffer = device.makeBuffer(
            bytes: points, length: MemoryLayout<PointAffine>.stride * n,
            options: .storageModeShared)!

        let scalarData = scalars.flatMap { $0 }
        let scalarsBuffer = device.makeBuffer(
            bytes: scalarData, length: MemoryLayout<UInt32>.stride * scalarData.count,
            options: .storageModeShared)!

        var windowResults: [PointProjective] = []

        for w in 0..<nWindows {
            let threadBuckets = device.makeBuffer(
                length: MemoryLayout<PointProjective>.stride * n,
                options: .storageModeShared)!
            let reducedBuckets = device.makeBuffer(
                length: MemoryLayout<PointProjective>.stride * nBuckets,
                options: .storageModeShared)!
            let windowResult = device.makeBuffer(
                length: MemoryLayout<PointProjective>.stride,
                options: .storageModeShared)!

            // Initialize buffers to identity
            memset(threadBuckets.contents(), 0, threadBuckets.length)
            memset(reducedBuckets.contents(), 0, reducedBuckets.length)
            memset(windowResult.contents(), 0, windowResult.length)

            var params = MsmParams(
                n_points: UInt32(n),
                window_bits: windowBits,
                window_index: UInt32(w)
            )

            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                throw MSMError.noCommandBuffer
            }

            // Phase 1: Accumulate into per-thread buckets
            let enc1 = commandBuffer.makeComputeCommandEncoder()!
            enc1.setComputePipelineState(accumulateFunction)
            enc1.setBuffer(pointsBuffer, offset: 0, index: 0)
            enc1.setBuffer(scalarsBuffer, offset: 0, index: 1)
            enc1.setBuffer(threadBuckets, offset: 0, index: 2)
            enc1.setBytes(&params, length: MemoryLayout<MsmParams>.stride, index: 3)
            let threadgroupSize1 = MTLSize(width: min(256, n), height: 1, depth: 1)
            let gridSize1 = MTLSize(width: n, height: 1, depth: 1)
            enc1.dispatchThreads(gridSize1, threadsPerThreadgroup: threadgroupSize1)
            enc1.endEncoding()

            // Phase 2: Reduce buckets
            let enc2 = commandBuffer.makeComputeCommandEncoder()!
            enc2.setComputePipelineState(reduceFunction)
            enc2.setBuffer(threadBuckets, offset: 0, index: 0)
            enc2.setBuffer(reducedBuckets, offset: 0, index: 1)
            enc2.setBytes(&params, length: MemoryLayout<MsmParams>.stride, index: 2)
            let threadgroupSize2 = MTLSize(width: min(256, nBuckets), height: 1, depth: 1)
            let gridSize2 = MTLSize(width: nBuckets, height: 1, depth: 1)
            enc2.dispatchThreads(gridSize2, threadsPerThreadgroup: threadgroupSize2)
            enc2.endEncoding()

            // Phase 3: Bucket sum
            let enc3 = commandBuffer.makeComputeCommandEncoder()!
            enc3.setComputePipelineState(bucketSumFunction)
            enc3.setBuffer(reducedBuckets, offset: 0, index: 0)
            enc3.setBuffer(windowResult, offset: 0, index: 1)
            enc3.setBytes(&params, length: MemoryLayout<MsmParams>.stride, index: 2)
            enc3.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            enc3.endEncoding()

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            if let error = commandBuffer.error {
                throw MSMError.gpuError(error.localizedDescription)
            }

            let resultPtr = windowResult.contents().bindMemory(to: PointProjective.self, capacity: 1)
            windowResults.append(resultPtr.pointee)
        }

        // Combine window results: result = sum(windowResults[i] * 2^(i*windowBits))
        // Using Horner's method from highest window to lowest
        var result = windowResults.last!
        for w in stride(from: nWindows - 2, through: 0, by: -1) {
            // Double windowBits times
            for _ in 0..<windowBits {
                result = cpuPointDouble(result)
            }
            result = cpuPointAdd(result, windowResults[w])
        }

        return result
    }

    // CPU-side point operations for final window combination (only ~16 operations)
    func cpuPointDouble(_ p: PointProjective) -> PointProjective {
        // Simplified — delegate to GPU for large operations,
        // this is only called ~16*16=256 times for window combination
        // For now, return identity as placeholder — real implementation would
        // use the same field arithmetic as the shader
        return p // TODO: implement CPU-side point doubling
    }

    func cpuPointAdd(_ p: PointProjective, _ q: PointProjective) -> PointProjective {
        return p // TODO: implement CPU-side point addition
    }
}

// MARK: - Utilities

enum MSMError: Error {
    case noGPU
    case noCommandQueue
    case noCommandBuffer
    case missingKernel
    case invalidInput
    case gpuError(String)
}

func findShaderPath() -> String {
    // Look relative to executable
    let execPath = CommandLine.arguments[0]
    let execDir = (execPath as NSString).deletingLastPathComponent

    let candidates = [
        "\(execDir)/shaders/bn254.metal",
        "\(execDir)/../Sources/zkmsm/shaders/bn254.metal",
        "./metal/Sources/zkmsm/shaders/bn254.metal",
        "./Sources/zkmsm/shaders/bn254.metal",
    ]

    for path in candidates {
        if FileManager.default.fileExists(atPath: path) {
            return path
        }
    }

    // Default to the path in the project
    return "metal/Sources/zkmsm/shaders/bn254.metal"
}

// MARK: - CLI

func runBenchmark(nPoints: Int) throws {
    fputs("zkmsm benchmark: \(nPoints) points on \(MTLCreateSystemDefaultDevice()?.name ?? "unknown GPU")\n", stderr)

    let engine = try MetalMSM()

    // Generate random test points (on-curve points would be needed for real use)
    var points: [PointAffine] = []
    var scalars: [[UInt32]] = []

    for i in 0..<nPoints {
        // Use generator point multiples as test data
        let x = Fp(v: (UInt32(i + 1), 0, 0, 0, 0, 0, 0, 0))
        let y = Fp(v: (UInt32(i + 2), 0, 0, 0, 0, 0, 0, 0))
        points.append(PointAffine(x: x, y: y))
        scalars.append([UInt32(i + 1), 0, 0, 0, 0, 0, 0, 0])
    }

    let start = CFAbsoluteTimeGetCurrent()
    let _ = try engine.msm(points: points, scalars: scalars)
    let elapsed = CFAbsoluteTimeGetCurrent() - start

    fputs("MSM(\(nPoints)): \(String(format: "%.3f", elapsed * 1000))ms\n", stderr)
    fputs("GPU: \(engine.device.name)\n", stderr)
    fputs("Max threadgroup: \(engine.accumulateFunction.maxTotalThreadsPerThreadgroup)\n", stderr)
}

func main() throws {
    let args = CommandLine.arguments

    if args.count >= 3 && args[1] == "--bench" {
        let n = Int(args[2]) ?? 1024
        try runBenchmark(nPoints: n)
        return
    }

    if args.contains("--info") {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("{\"error\": \"No Metal GPU available\"}")
            return
        }
        let info: [String: Any] = [
            "gpu": device.name,
            "unified_memory": device.hasUnifiedMemory,
            "max_buffer_length": device.maxBufferLength,
            "max_threadgroup_memory": device.maxThreadgroupMemoryLength,
        ]
        let data = try JSONSerialization.data(withJSONObject: info, options: .prettyPrinted)
        print(String(data: data, encoding: .utf8)!)
        return
    }

    // Default: read JSON from stdin, compute MSM, output result
    fputs("zkmsm: Metal GPU MSM for BN254\n", stderr)
    fputs("Usage: zkmsm --bench <n_points> | zkmsm --info\n", stderr)
}

try main()
