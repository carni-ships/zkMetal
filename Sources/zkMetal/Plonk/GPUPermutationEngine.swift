// GPUPermutationEngine — GPU-accelerated permutation argument for Plonk copy constraints
//
// Offloads the entire permutation argument to Metal GPU:
//   1. Compute per-row numerator/denominator products across all wires
//   2. Batch-invert denominators on GPU via GPUBatchInverseEngine
//   3. Compute running grand product Z(x) via GPUGrandProductEngine
//   4. Verify the grand product identity (boundary + transition checks)
//
// The permutation polynomial Z(x) is defined on domain {omega^0, ..., omega^{n-1}}:
//   Z(omega^0) = 1
//   Z(omega^{i+1}) = Z(omega^i) * prod_j (w_j[i] + beta*id_j(i) + gamma)
//                                        / (w_j[i] + beta*sigma_j[i] + gamma)
//
// where id_j(i) = cosetMul(j) * omega^i is the identity permutation, and
// sigma_j encodes the actual wire routing (copy constraints).
//
// GPU acceleration strategy:
//   - A Metal compute kernel computes per-row numerator and denominator products
//     across all wires in parallel (one thread per row).
//   - GPUBatchInverseEngine inverts all denominators in one GPU pass.
//   - GPUGrandProductEngine computes the prefix product of num*inv_den ratios.
//   - Verification reuses the same kernel infrastructure.

import Foundation
import Metal
import NeonFieldOps

public class GPUPermutationEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let grandProductEngine: GPUGrandProductEngine
    private let batchInverseEngine: GPUBatchInverseEngine

    // Metal pipeline for computing per-row numerator/denominator products
    private let numDenPipeline: MTLComputePipelineState
    private let threadgroupSize: Int

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue
        self.threadgroupSize = 256

        self.grandProductEngine = try GPUGrandProductEngine()
        self.batchInverseEngine = try GPUBatchInverseEngine()

        let library = try GPUPermutationEngine.compileShaders(device: device)

        guard let fn = library.makeFunction(name: "permutation_num_den") else {
            throw MSMError.missingKernel
        }
        self.numDenPipeline = try device.makeComputePipelineState(function: fn)
    }

    // MARK: - Public API

    /// Compute the permutation grand product polynomial Z(x) in evaluation form.
    ///
    /// z[0] = 1
    /// z[i+1] = z[i] * prod_j (w_j[i] + beta*id_j(i) + gamma)
    ///                       / (w_j[i] + beta*sigma_j[i] + gamma)
    ///
    /// - Parameters:
    ///   - witness: Per-wire witness evaluations. witness[j] has n elements for wire j.
    ///   - sigmaPolys: Per-wire sigma permutation evaluations. sigmaPolys[j] has n elements.
    ///   - beta: Permutation challenge beta.
    ///   - gamma: Permutation challenge gamma.
    /// - Returns: Grand product polynomial Z(x) in evaluation form, length n.
    public func computePermutationPoly(
        witness: [[Fr]],
        sigmaPolys: [[Fr]],
        beta: Fr,
        gamma: Fr
    ) -> [Fr] {
        let numWires = witness.count
        let n = witness[0].count
        precondition(sigmaPolys.count == numWires, "sigma count must match witness count")
        for j in 0..<numWires {
            precondition(witness[j].count == n, "witness[\(j)] length mismatch")
            precondition(sigmaPolys[j].count == n, "sigma[\(j)] length mismatch")
        }
        precondition(n > 0, "domain size must be positive")
        precondition(n & (n - 1) == 0, "domain size must be a power of 2")

        // Build evaluation domain
        let logN = Int(log2(Double(n)))
        let omega = computeNthRootOfUnity(logN: logN)
        var domain = [Fr](repeating: Fr.zero, count: n)
        domain[0] = Fr.one
        for i in 1..<n {
            domain[i] = frMul(domain[i - 1], omega)
        }

        // Build coset generators: column 0 -> 1, column j -> j+1
        var cosetMuls = [Fr](repeating: Fr.one, count: numWires)
        for j in 1..<numWires {
            cosetMuls[j] = frFromInt(UInt64(j + 1))
        }

        // Compute per-row numerators and denominators on GPU
        let (numerators, denominators) = computeNumDen(
            witness: witness, sigmaPolys: sigmaPolys,
            beta: beta, gamma: gamma, domain: domain,
            cosetMuls: cosetMuls, n: n, numWires: numWires
        )

        // Use GPUGrandProductEngine for batch inverse + prefix product
        return grandProductEngine.permutationProduct(
            numerators: numerators, denominators: denominators
        )
    }

    /// Verify the permutation argument: checks both boundary and transition identities.
    ///
    /// 1. Z[0] == 1  (boundary)
    /// 2. For all i in 0..<n:
    ///    Z[i+1 mod n] * prod_j(w_j[i] + beta*sigma_j[i] + gamma)
    ///    == Z[i] * prod_j(w_j[i] + beta*id_j(i) + gamma)
    /// 3. The full product equals 1 (Z wraps around)
    ///
    /// - Parameters:
    ///   - z: Grand product polynomial evaluations (length n).
    ///   - witness: Per-wire witness evaluations.
    ///   - sigmaPolys: Per-wire sigma permutation evaluations.
    ///   - beta: Permutation challenge.
    ///   - gamma: Permutation challenge.
    /// - Returns: True if the permutation argument is valid.
    public func verifyPermutation(
        z: [Fr],
        witness: [[Fr]],
        sigmaPolys: [[Fr]],
        beta: Fr,
        gamma: Fr
    ) -> Bool {
        let numWires = witness.count
        let n = z.count
        guard n > 0 else { return false }
        guard witness.count == sigmaPolys.count else { return false }
        for j in 0..<numWires {
            guard witness[j].count == n, sigmaPolys[j].count == n else { return false }
        }

        // Check 1: Z[0] == 1
        guard z[0] == Fr.one else { return false }

        // Build evaluation domain
        let logN = Int(log2(Double(n)))
        let omega = computeNthRootOfUnity(logN: logN)
        var domain = [Fr](repeating: Fr.zero, count: n)
        domain[0] = Fr.one
        for i in 1..<n {
            domain[i] = frMul(domain[i - 1], omega)
        }

        // Coset generators
        var cosetMuls = [Fr](repeating: Fr.one, count: numWires)
        for j in 1..<numWires {
            cosetMuls[j] = frFromInt(UInt64(j + 1))
        }

        // Check 2: Transition identity for every row
        // Z[i+1] = Z[i] * num[i] / den[i]
        // Equivalently: Z[i+1] * den[i] == Z[i] * num[i]
        for i in 0..<n {
            var numProd = Fr.one
            var denProd = Fr.one

            for j in 0..<numWires {
                let idVal = frMul(cosetMuls[j], domain[i])
                // numerator: w_j[i] + beta * id_j(i) + gamma
                let numTerm = frAdd(frAdd(witness[j][i], frMul(beta, idVal)), gamma)
                numProd = frMul(numProd, numTerm)

                // denominator: w_j[i] + beta * sigma_j[i] + gamma
                let denTerm = frAdd(frAdd(witness[j][i], frMul(beta, sigmaPolys[j][i])), gamma)
                denProd = frMul(denProd, denTerm)
            }

            let iNext = (i + 1) % n
            // Check: Z[iNext] * denProd == Z[i] * numProd
            let lhs = frMul(z[iNext], denProd)
            let rhs = frMul(z[i], numProd)
            if !frSub(lhs, rhs).isZero {
                return false
            }
        }

        return true
    }

    // MARK: - GPU Numerator/Denominator Computation

    /// Compute per-row numerator and denominator products on GPU.
    /// Each thread handles one row, iterating over all wire columns.
    private func computeNumDen(
        witness: [[Fr]],
        sigmaPolys: [[Fr]],
        beta: Fr,
        gamma: Fr,
        domain: [Fr],
        cosetMuls: [Fr],
        n: Int,
        numWires: Int
    ) -> (numerators: [Fr], denominators: [Fr]) {
        let frStride = MemoryLayout<Fr>.stride

        // For small sizes, use CPU
        if n < 1024 {
            return cpuNumDen(
                witness: witness, sigmaPolys: sigmaPolys,
                beta: beta, gamma: gamma, domain: domain,
                cosetMuls: cosetMuls, n: n, numWires: numWires
            )
        }

        // Flatten witness and sigma into contiguous buffers: [wire0_row0, wire0_row1, ..., wire1_row0, ...]
        var flatWitness = [Fr](repeating: Fr.zero, count: numWires * n)
        var flatSigma = [Fr](repeating: Fr.zero, count: numWires * n)
        for j in 0..<numWires {
            for i in 0..<n {
                flatWitness[j * n + i] = witness[j][i]
                flatSigma[j * n + i] = sigmaPolys[j][i]
            }
        }

        let wireDataSize = numWires * n * frStride
        let rowDataSize = n * frStride
        let cosetSize = numWires * frStride

        guard let witBuf = device.makeBuffer(length: wireDataSize, options: .storageModeShared),
              let sigBuf = device.makeBuffer(length: wireDataSize, options: .storageModeShared),
              let domBuf = device.makeBuffer(length: rowDataSize, options: .storageModeShared),
              let cosetBuf = device.makeBuffer(length: cosetSize, options: .storageModeShared),
              let numBuf = device.makeBuffer(length: rowDataSize, options: .storageModeShared),
              let denBuf = device.makeBuffer(length: rowDataSize, options: .storageModeShared) else {
            return cpuNumDen(
                witness: witness, sigmaPolys: sigmaPolys,
                beta: beta, gamma: gamma, domain: domain,
                cosetMuls: cosetMuls, n: n, numWires: numWires
            )
        }

        flatWitness.withUnsafeBytes { src in memcpy(witBuf.contents(), src.baseAddress!, wireDataSize) }
        flatSigma.withUnsafeBytes { src in memcpy(sigBuf.contents(), src.baseAddress!, wireDataSize) }
        domain.withUnsafeBytes { src in memcpy(domBuf.contents(), src.baseAddress!, rowDataSize) }
        cosetMuls.withUnsafeBytes { src in memcpy(cosetBuf.contents(), src.baseAddress!, cosetSize) }

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return cpuNumDen(
                witness: witness, sigmaPolys: sigmaPolys,
                beta: beta, gamma: gamma, domain: domain,
                cosetMuls: cosetMuls, n: n, numWires: numWires
            )
        }

        encoder.setComputePipelineState(numDenPipeline)
        encoder.setBuffer(witBuf, offset: 0, index: 0)
        encoder.setBuffer(sigBuf, offset: 0, index: 1)
        encoder.setBuffer(domBuf, offset: 0, index: 2)
        encoder.setBuffer(cosetBuf, offset: 0, index: 3)
        encoder.setBuffer(numBuf, offset: 0, index: 4)
        encoder.setBuffer(denBuf, offset: 0, index: 5)

        var betaVal = beta
        var gammaVal = gamma
        var nVal = UInt32(n)
        var numWiresVal = UInt32(numWires)
        encoder.setBytes(&betaVal, length: frStride, index: 6)
        encoder.setBytes(&gammaVal, length: frStride, index: 7)
        encoder.setBytes(&nVal, length: 4, index: 8)
        encoder.setBytes(&numWiresVal, length: 4, index: 9)

        let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        let gridSize = MTLSize(width: n, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if cmdBuf.error != nil {
            return cpuNumDen(
                witness: witness, sigmaPolys: sigmaPolys,
                beta: beta, gamma: gamma, domain: domain,
                cosetMuls: cosetMuls, n: n, numWires: numWires
            )
        }

        var numerators = [Fr](repeating: Fr.zero, count: n)
        var denominators = [Fr](repeating: Fr.zero, count: n)
        numerators.withUnsafeMutableBytes { dst in memcpy(dst.baseAddress!, numBuf.contents(), rowDataSize) }
        denominators.withUnsafeMutableBytes { dst in memcpy(dst.baseAddress!, denBuf.contents(), rowDataSize) }
        return (numerators, denominators)
    }

    // MARK: - CPU Fallback

    private func cpuNumDen(
        witness: [[Fr]],
        sigmaPolys: [[Fr]],
        beta: Fr,
        gamma: Fr,
        domain: [Fr],
        cosetMuls: [Fr],
        n: Int,
        numWires: Int
    ) -> (numerators: [Fr], denominators: [Fr]) {
        var numerators = [Fr](repeating: Fr.one, count: n)
        var denominators = [Fr](repeating: Fr.one, count: n)

        for i in 0..<n {
            for j in 0..<numWires {
                let idVal = frMul(cosetMuls[j], domain[i])
                let numTerm = frAdd(frAdd(witness[j][i], frMul(beta, idVal)), gamma)
                numerators[i] = frMul(numerators[i], numTerm)

                let denTerm = frAdd(frAdd(witness[j][i], frMul(beta, sigmaPolys[j][i])), gamma)
                denominators[i] = frMul(denominators[i], denTerm)
            }
        }

        return (numerators, denominators)
    }

    // MARK: - Metal Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)

        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let kernelSource = """

        // Per-row numerator/denominator product for Plonk permutation argument.
        // Each thread computes the product over all wires for one row.
        //
        // numerator[i] = prod_j (witness[j*n+i] + beta * cosetMuls[j] * domain[i] + gamma)
        // denominator[i] = prod_j (witness[j*n+i] + beta * sigma[j*n+i] + gamma)
        kernel void permutation_num_den(
            device const Fr *witness [[buffer(0)]],
            device const Fr *sigma [[buffer(1)]],
            device const Fr *domain [[buffer(2)]],
            device const Fr *cosetMuls [[buffer(3)]],
            device Fr *numerators [[buffer(4)]],
            device Fr *denominators [[buffer(5)]],
            constant Fr &beta [[buffer(6)]],
            constant Fr &gamma [[buffer(7)]],
            constant uint &n [[buffer(8)]],
            constant uint &numWires [[buffer(9)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= n) return;

            Fr numProd = fr_one();
            Fr denProd = fr_one();
            Fr domainVal = domain[gid];

            for (uint j = 0; j < numWires; j++) {
                uint idx = j * n + gid;
                Fr w = witness[idx];
                Fr s = sigma[idx];
                Fr coset = cosetMuls[j];

                // id_j(i) = cosetMuls[j] * domain[i]
                Fr idVal = fr_mul(coset, domainVal);

                // numerator term: w + beta * id + gamma
                Fr betaId = fr_mul(beta, idVal);
                Fr numTerm = fr_add(fr_add(w, betaId), gamma);
                numProd = fr_mul(numProd, numTerm);

                // denominator term: w + beta * sigma + gamma
                Fr betaSig = fr_mul(beta, s);
                Fr denTerm = fr_add(fr_add(w, betaSig), gamma);
                denProd = fr_mul(denProd, denTerm);
            }

            numerators[gid] = numProd;
            denominators[gid] = denProd;
        }
        """

        let combined = frClean + "\n" + kernelSource
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    private static func findShaderDir() -> String {
        let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                if FileManager.default.fileExists(atPath: url.appendingPathComponent("fields/bn254_fr.metal").path) {
                    return url.path
                }
            }
        }
        let candidates = [
            execDir + "/Shaders",
            execDir + "/../share/zkMetal/Shaders",
            "Sources/zkMetal/Shaders",
            FileManager.default.currentDirectoryPath + "/Sources/zkMetal/Shaders",
        ]
        for c in candidates {
            if FileManager.default.fileExists(atPath: c + "/fields/bn254_fr.metal") {
                return c
            }
        }
        return "Sources/zkMetal/Shaders"
    }
}
