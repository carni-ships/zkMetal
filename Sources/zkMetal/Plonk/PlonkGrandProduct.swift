// PlonkGrandProduct — GPU-accelerated grand product computation
//
// Computes the running product (prefix product) of a sequence of field elements:
//   output[0] = 1
//   output[i] = input[0] * input[1] * ... * input[i-1]
//
// Used by:
//   - Permutation argument: running product of numerator/denominator ratios
//   - Lookup argument: sorted polynomial accumulator
//   - Any protocol requiring a grand product witness
//
// GPU path uses a parallel prefix product with Metal compute shaders.
// CPU fallback for small arrays or when GPU is unavailable.

import Foundation
import Metal
import NeonFieldOps

// MARK: - Grand Product Engine

/// Engine for computing running (prefix) products of field element sequences.
///
/// The running product is defined as:
///   result[0] = 1
///   result[i] = values[0] * values[1] * ... * values[i-1]
///
/// For the Plonk permutation argument, the input values are the per-row
/// numerator/denominator ratios, and the output is Z(omega^i).
public class PlonkGrandProductEngine {

    /// GPU dispatch threshold: arrays smaller than this use CPU.
    public static let gpuThreshold = 4096

    /// Metal device (nil if GPU unavailable)
    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let prefixProductKernel: MTLComputePipelineState?
    private let propagateKernel: MTLComputePipelineState?

    /// Threadgroup size for prefix product kernel.
    private let threadgroupSize: Int

    public init() {
        // Try to initialize GPU; fall back to CPU-only if unavailable
        guard let dev = MTLCreateSystemDefaultDevice(),
              let queue = dev.makeCommandQueue() else {
            self.device = nil
            self.commandQueue = nil
            self.prefixProductKernel = nil
            self.propagateKernel = nil
            self.threadgroupSize = 256
            return
        }

        self.device = dev
        self.commandQueue = queue
        self.threadgroupSize = min(256, dev.maxThreadsPerThreadgroup.width)

        // Compile the prefix product kernel from source
        let kernelSource = PlonkGrandProductEngine.metalKernelSource()
        var ppKernel: MTLComputePipelineState? = nil
        var propKernel: MTLComputePipelineState? = nil
        if let library = try? dev.makeLibrary(source: kernelSource, options: nil) {
            if let fn = library.makeFunction(name: "prefix_product_local") {
                ppKernel = try? dev.makeComputePipelineState(function: fn)
            }
            if let fn = library.makeFunction(name: "prefix_product_propagate") {
                propKernel = try? dev.makeComputePipelineState(function: fn)
            }
        }
        self.prefixProductKernel = ppKernel
        self.propagateKernel = propKernel
    }

    // MARK: - Public API

    /// Compute the running product of `values`.
    ///
    /// result[0] = 1
    /// result[i] = values[0] * values[1] * ... * values[i-1]
    /// result[n] is NOT included (output has same length as input).
    ///
    /// For Plonk permutation: pass the per-row num/den ratios, get Z evaluations.
    ///
    /// - Parameter values: Input field elements.
    /// - Returns: Running product array of the same length.
    public func gpuGrandProduct(values: [Fr]) -> [Fr] {
        let n = values.count
        guard n > 0 else { return [] }

        // Use GPU if available and array is large enough
        if n >= PlonkGrandProductEngine.gpuThreshold,
           prefixProductKernel != nil, propagateKernel != nil {
            if let result = gpuPrefixProduct(values) {
                return result
            }
        }

        // CPU fallback
        return cpuPrefixProduct(values)
    }

    /// Compute the running product on CPU using sequential scan.
    ///
    /// This is the reference implementation: O(n) sequential multiplications.
    /// Used for small arrays or as fallback when GPU is unavailable.
    public static func cpuGrandProduct(values: [Fr]) -> [Fr] {
        let n = values.count
        guard n > 0 else { return [] }
        var result = [Fr](repeating: Fr.zero, count: n)
        result[0] = Fr.one
        for i in 1..<n {
            result[i] = frMul(result[i - 1], values[i - 1])
        }
        return result
    }

    /// Compute the full product of all elements: values[0] * values[1] * ... * values[n-1].
    public static func fullProduct(_ values: [Fr]) -> Fr {
        guard !values.isEmpty else { return Fr.one }
        var acc = Fr.one
        for v in values {
            acc = frMul(acc, v)
        }
        return acc
    }

    /// Compute per-row numerator/denominator ratios for the Plonk permutation.
    ///
    /// For each row i:
    ///   ratio[i] = prod_j (witness_j[i] + beta * id_j(i) + gamma) /
    ///                      (witness_j[i] + beta * sigma_j[i] + gamma)
    ///
    /// The running product of these ratios gives Z(omega^i).
    ///
    /// - Parameters:
    ///   - witness: Per-wire witness evaluations.
    ///   - sigma: Per-wire sigma permutation evaluations.
    ///   - beta: Permutation challenge.
    ///   - gamma: Permutation challenge.
    ///   - domain: Evaluation domain.
    ///   - permArg: Permutation argument instance (for coset generators).
    /// - Returns: Array of per-row ratios.
    public static func computePermutationRatios(
        witness: [[Fr]],
        sigma: [[Fr]],
        beta: Fr,
        gamma: Fr,
        domain: [Fr],
        permArg: PermutationArgument
    ) -> [Fr] {
        let n = domain.count
        let numWires = permArg.numWires

        var numerators = [Fr](repeating: Fr.one, count: n)
        var denominators = [Fr](repeating: Fr.one, count: n)

        for i in 0..<n {
            for j in 0..<numWires {
                let kj = permArg.cosetMultiplier(forWire: j)
                let idVal = frMul(kj, domain[i])
                let numTerm = frAdd(frAdd(witness[j][i], frMul(beta, idVal)), gamma)
                numerators[i] = frMul(numerators[i], numTerm)

                let denTerm = frAdd(frAdd(witness[j][i], frMul(beta, sigma[j][i])), gamma)
                denominators[i] = frMul(denominators[i], denTerm)
            }
        }

        // Batch invert denominators
        var invDenominators = [Fr](repeating: Fr.zero, count: n)
        denominators.withUnsafeBytes { denBuf in
            invDenominators.withUnsafeMutableBytes { invBuf in
                bn254_fr_batch_inverse(
                    denBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n),
                    invBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }

        // Element-wise multiply: ratios[i] = numerators[i] * invDenominators[i]
        var ratios = [Fr](repeating: Fr.zero, count: n)
        numerators.withUnsafeBytes { nBuf in
            invDenominators.withUnsafeBytes { iBuf in
                ratios.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul_parallel(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        nBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        iBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return ratios
    }

    // MARK: - CPU Implementation

    private func cpuPrefixProduct(_ values: [Fr]) -> [Fr] {
        return PlonkGrandProductEngine.cpuGrandProduct(values: values)
    }

    // MARK: - GPU Implementation

    /// GPU parallel prefix product using a two-pass algorithm:
    ///   Pass 1: Each threadgroup computes local prefix products and writes
    ///           its block's total product to an auxiliary buffer.
    ///   Pass 2: Propagate block totals to subsequent blocks.
    ///
    /// For very large arrays, this recurses on the block totals.
    private func gpuPrefixProduct(_ values: [Fr]) -> [Fr]? {
        guard let device = device,
              let queue = commandQueue,
              let localKernel = prefixProductKernel,
              let propKernel = propagateKernel else {
            return nil
        }

        let n = values.count
        let elemSize = MemoryLayout<Fr>.stride
        let blockSize = threadgroupSize

        // Number of blocks
        let numBlocks = (n + blockSize - 1) / blockSize

        // Allocate GPU buffers
        guard let inputBuf = device.makeBuffer(length: n * elemSize, options: .storageModeShared),
              let outputBuf = device.makeBuffer(length: n * elemSize, options: .storageModeShared),
              let blockTotalsBuf = device.makeBuffer(length: numBlocks * elemSize, options: .storageModeShared) else {
            return nil
        }

        // Copy input
        values.withUnsafeBytes { src in
            inputBuf.contents().copyMemory(from: src.baseAddress!, byteCount: n * elemSize)
        }

        // Pass 1: Local prefix products per block
        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return nil
        }

        encoder.setComputePipelineState(localKernel)
        encoder.setBuffer(inputBuf, offset: 0, index: 0)
        encoder.setBuffer(outputBuf, offset: 0, index: 1)
        encoder.setBuffer(blockTotalsBuf, offset: 0, index: 2)
        var nVal = UInt32(n)
        encoder.setBytes(&nVal, length: 4, index: 3)

        let threadsPerGroup = MTLSize(width: blockSize, height: 1, depth: 1)
        let numGroups = MTLSize(width: numBlocks, height: 1, depth: 1)
        encoder.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        // If multiple blocks, propagate block totals
        if numBlocks > 1 {
            // Compute prefix product of block totals on CPU (typically small)
            // Then propagate to each block
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            // Read block totals
            var blockTotals = [Fr](repeating: Fr.zero, count: numBlocks)
            blockTotals.withUnsafeMutableBytes { dst in
                dst.copyBytes(from: UnsafeRawBufferPointer(
                    start: blockTotalsBuf.contents(),
                    count: numBlocks * elemSize
                ))
            }

            // Prefix product of block totals
            var blockPrefixes = [Fr](repeating: Fr.zero, count: numBlocks)
            blockPrefixes[0] = Fr.one
            for i in 1..<numBlocks {
                blockPrefixes[i] = frMul(blockPrefixes[i - 1], blockTotals[i - 1])
            }

            // Write back prefix products
            guard let prefixBuf = device.makeBuffer(length: numBlocks * elemSize, options: .storageModeShared) else {
                return nil
            }
            blockPrefixes.withUnsafeBytes { src in
                prefixBuf.contents().copyMemory(from: src.baseAddress!, byteCount: numBlocks * elemSize)
            }

            // Pass 2: Multiply each element by its block's prefix
            guard let cmdBuf2 = queue.makeCommandBuffer(),
                  let encoder2 = cmdBuf2.makeComputeCommandEncoder() else {
                return nil
            }

            encoder2.setComputePipelineState(propKernel)
            encoder2.setBuffer(outputBuf, offset: 0, index: 0)
            encoder2.setBuffer(prefixBuf, offset: 0, index: 1)
            var bsVal = UInt32(blockSize)
            encoder2.setBytes(&nVal, length: 4, index: 2)
            encoder2.setBytes(&bsVal, length: 4, index: 3)

            let totalThreads = MTLSize(width: n, height: 1, depth: 1)
            encoder2.dispatchThreads(totalThreads, threadsPerThreadgroup: threadsPerGroup)
            encoder2.endEncoding()

            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()
        } else {
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }

        // Read result
        var result = [Fr](repeating: Fr.zero, count: n)
        result.withUnsafeMutableBytes { dst in
            dst.copyBytes(from: UnsafeRawBufferPointer(
                start: outputBuf.contents(),
                count: n * elemSize
            ))
        }
        return result
    }

    // MARK: - Metal Shader Source

    /// Metal shader source for parallel prefix product over BN254 Fr.
    ///
    /// Uses the same 256-bit Montgomery representation as the rest of zkMetal.
    /// Two kernels:
    ///   1. prefix_product_local: per-threadgroup prefix product + block total
    ///   2. prefix_product_propagate: multiply each element by its block's accumulated prefix
    private static func metalKernelSource() -> String {
        return """
        #include <metal_stdlib>
        using namespace metal;

        // BN254 Fr: 4 x uint64_t in Montgomery form
        struct Fr {
            ulong4 v;  // v.x = limb0, v.y = limb1, v.z = limb2, v.w = limb3
        };

        // BN254 Fr modulus
        constant ulong4 FR_P = ulong4(0x43e1f593f0000001UL, 0x2833e84879b97091UL,
                                       0xb85045b68181585dUL, 0x30644e72e131a029UL);

        // Montgomery R mod r (representation of 1)
        constant ulong4 FR_ONE = ulong4(0xac96341c4ffffffbUL, 0x36fc76959f60cd29UL,
                                         0x666ea36f7879462eUL, 0x0e0a77c19a07df2fUL);

        // -r^(-1) mod 2^64
        constant ulong FR_INV = 0xc2e1f593efffffffUL;

        // 128-bit multiply helper
        inline ulong2 mul128(ulong a, ulong b) {
            ulong hi = mulhi(a, b);
            ulong lo = a * b;
            return ulong2(lo, hi);
        }

        // Add with carry
        inline ulong adc(ulong a, ulong b, thread ulong &carry) {
            ulong r = a + b + carry;
            carry = (r < a || (carry && r == a)) ? 1UL : 0UL;
            return r;
        }

        // Subtract with borrow
        inline ulong sbb(ulong a, ulong b, thread ulong &borrow) {
            ulong r = a - b - borrow;
            borrow = (a < b + borrow || (borrow && b == 0xFFFFFFFFFFFFFFFFUL)) ? 1UL : 0UL;
            return r;
        }

        // Montgomery multiplication: a * b mod r
        inline Fr fr_mul(Fr a, Fr b) {
            // Schoolbook 4x4 multiply with Montgomery reduction
            ulong t[8] = {0,0,0,0,0,0,0,0};

            // Multiply
            for (int i = 0; i < 4; i++) {
                ulong ai = (i == 0) ? a.v.x : (i == 1) ? a.v.y : (i == 2) ? a.v.z : a.v.w;
                ulong carry = 0;
                for (int j = 0; j < 4; j++) {
                    ulong bj = (j == 0) ? b.v.x : (j == 1) ? b.v.y : (j == 2) ? b.v.z : b.v.w;
                    ulong2 prod = mul128(ai, bj);
                    ulong lo = prod.x + t[i+j] + carry;
                    carry = prod.y + ((lo < prod.x) ? 1UL : 0UL) + ((lo < t[i+j] && lo >= prod.x) ? 1UL : 0UL);
                    t[i+j] = lo;
                }
                t[i+4] = carry;
            }

            // Montgomery reduction
            for (int i = 0; i < 4; i++) {
                ulong m = t[i] * FR_INV;
                ulong carry = 0;
                for (int j = 0; j < 4; j++) {
                    ulong pj = (j == 0) ? FR_P.x : (j == 1) ? FR_P.y : (j == 2) ? FR_P.z : FR_P.w;
                    ulong2 prod = mul128(m, pj);
                    ulong lo = prod.x + t[i+j] + carry;
                    carry = prod.y + ((lo < prod.x) ? 1UL : 0UL) + ((lo < t[i+j] && lo >= prod.x) ? 1UL : 0UL);
                    t[i+j] = lo;
                }
                // Propagate carry
                for (int k = i + 4; k < 8; k++) {
                    ulong s = t[k] + carry;
                    carry = (s < t[k]) ? 1UL : 0UL;
                    t[k] = s;
                    if (carry == 0) break;
                }
            }

            // Result is in t[4..7], reduce mod p if needed
            Fr r;
            r.v = ulong4(t[4], t[5], t[6], t[7]);

            // Conditional subtraction
            ulong borrow = 0;
            ulong s0 = sbb(r.v.x, FR_P.x, borrow);
            ulong s1 = sbb(r.v.y, FR_P.y, borrow);
            ulong s2 = sbb(r.v.z, FR_P.z, borrow);
            ulong s3 = sbb(r.v.w, FR_P.w, borrow);

            if (borrow == 0) {
                r.v = ulong4(s0, s1, s2, s3);
            }
            return r;
        }

        // Per-threadgroup prefix product
        // Each threadgroup computes local prefix products and writes
        // the total product of its block to blockTotals.
        kernel void prefix_product_local(
            device const Fr *input [[buffer(0)]],
            device Fr *output [[buffer(1)]],
            device Fr *blockTotals [[buffer(2)]],
            constant uint &n [[buffer(3)]],
            uint tid [[thread_index_in_threadgroup]],
            uint gid [[thread_position_in_grid]],
            uint groupId [[threadgroup_position_in_grid]],
            uint groupSize [[threads_per_threadgroup]]
        ) {
            // Load element (or identity if out of bounds)
            threadgroup Fr shared_data[256];

            Fr val;
            if (gid < n) {
                val = input[gid];
            } else {
                val.v = FR_ONE;
            }
            shared_data[tid] = val;

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Inclusive prefix product using up-sweep / down-sweep
            // Up-sweep (reduce)
            for (uint stride = 1; stride < groupSize; stride <<= 1) {
                uint idx = (tid + 1) * (stride << 1) - 1;
                if (idx < groupSize) {
                    shared_data[idx] = fr_mul(shared_data[idx - stride], shared_data[idx]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Save block total before down-sweep clears it
            if (tid == 0) {
                blockTotals[groupId] = shared_data[groupSize - 1];
            }

            // Set last element to identity for exclusive scan
            if (tid == groupSize - 1) {
                shared_data[groupSize - 1].v = FR_ONE;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Down-sweep
            for (uint stride = groupSize >> 1; stride > 0; stride >>= 1) {
                uint idx = (tid + 1) * (stride << 1) - 1;
                if (idx < groupSize) {
                    Fr temp = shared_data[idx];
                    shared_data[idx] = fr_mul(shared_data[idx], shared_data[idx - stride]);
                    shared_data[idx - stride] = temp;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Write exclusive prefix product
            if (gid < n) {
                output[gid] = shared_data[tid];
            }
        }

        // Propagate block prefixes: multiply each element by its block's accumulated prefix
        kernel void prefix_product_propagate(
            device Fr *data [[buffer(0)]],
            device const Fr *blockPrefixes [[buffer(1)]],
            constant uint &n [[buffer(2)]],
            constant uint &blockSize [[buffer(3)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= n) return;
            uint blockIdx = gid / blockSize;
            if (blockIdx == 0) return;  // First block needs no adjustment
            data[gid] = fr_mul(blockPrefixes[blockIdx], data[gid]);
        }
        """;
    }
}
