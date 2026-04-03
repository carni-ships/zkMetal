// Parallel CPU NTT — multithreaded Cooley-Tukey butterfly
// Uses GCD concurrentPerform to parallelize independent butterfly groups.
// Does NOT replace vanilla cpuNTT (which remains as the correctness reference).

import Foundation
import NeonFieldOps

// MARK: - Bit-reversal permutation (shared)

private func bitRev(_ i: Int, _ logN: Int) -> Int {
    var r = 0
    var x = i
    for _ in 0..<logN {
        r = (r << 1) | (x & 1)
        x >>= 1
    }
    return r
}

// MARK: - BN254 Fr parallel NTT

public func parallelNTT_Fr(_ input: [Fr], logN: Int) -> [Fr] {
    let n = input.count
    // Bit-reverse permutation
    var data = [Fr](repeating: Fr.zero, count: n)
    for i in 0..<n {
        data[bitRev(i, logN)] = input[i]
    }

    let nThreads = ProcessInfo.processInfo.activeProcessorCount
    let omega = frRootOfUnity(logN: logN)

    for s in 0..<logN {
        let halfBlock = 1 << s
        let blockSize = halfBlock << 1
        let nBlocks = n / blockSize

        // Precompute twiddle: w_m = omega^(n/blockSize)
        var w_m = Fr.one
        var temp = omega
        var k = n / blockSize
        while k > 0 {
            if k & 1 == 1 { w_m = frMul(w_m, temp) }
            temp = frSqr(temp)
            k >>= 1
        }

        // Precompute all twiddle factors for this stage
        var twiddles = [Fr](repeating: Fr.one, count: halfBlock)
        for j in 1..<halfBlock {
            twiddles[j] = frMul(twiddles[j-1], w_m)
        }

        // Parallel over blocks when there are enough
        if nBlocks >= nThreads {
            DispatchQueue.concurrentPerform(iterations: nBlocks) { blockIdx in
                let base = blockIdx * blockSize
                for j in 0..<halfBlock {
                    let u = data[base + j]
                    let v = frMul(twiddles[j], data[base + j + halfBlock])
                    data[base + j] = frAdd(u, v)
                    data[base + j + halfBlock] = frSub(u, v)
                }
            }
        } else if halfBlock >= 64 {
            // Few large blocks — parallelize within each block
            for blockIdx in 0..<nBlocks {
                let base = blockIdx * blockSize
                let chunkSize = max(1, halfBlock / nThreads)
                DispatchQueue.concurrentPerform(iterations: (halfBlock + chunkSize - 1) / chunkSize) { chunkIdx in
                    let jStart = chunkIdx * chunkSize
                    let jEnd = min(jStart + chunkSize, halfBlock)
                    for j in jStart..<jEnd {
                        let u = data[base + j]
                        let v = frMul(twiddles[j], data[base + j + halfBlock])
                        data[base + j] = frAdd(u, v)
                        data[base + j + halfBlock] = frSub(u, v)
                    }
                }
            }
        } else {
            // Small stage — serial
            for blockIdx in 0..<nBlocks {
                let base = blockIdx * blockSize
                for j in 0..<halfBlock {
                    let u = data[base + j]
                    let v = frMul(twiddles[j], data[base + j + halfBlock])
                    data[base + j] = frAdd(u, v)
                    data[base + j + halfBlock] = frSub(u, v)
                }
            }
        }
    }
    return data
}

public func parallelINTT_Fr(_ input: [Fr], logN: Int) -> [Fr] {
    let n = input.count
    let omega = frRootOfUnity(logN: logN)
    let omegaInv = frInverse(omega)

    let nThreads = ProcessInfo.processInfo.activeProcessorCount
    var data = input

    // DIF stages (top-down)
    for si in 0..<logN {
        let s = logN - 1 - si
        let halfBlock = 1 << s
        let blockSize = halfBlock << 1
        let nBlocks = n / blockSize

        var w_m = Fr.one
        var temp = omegaInv
        var k = n / blockSize
        while k > 0 {
            if k & 1 == 1 { w_m = frMul(w_m, temp) }
            temp = frSqr(temp)
            k >>= 1
        }

        var twiddles = [Fr](repeating: Fr.one, count: halfBlock)
        for j in 1..<halfBlock {
            twiddles[j] = frMul(twiddles[j-1], w_m)
        }

        if nBlocks >= nThreads {
            DispatchQueue.concurrentPerform(iterations: nBlocks) { blockIdx in
                let base = blockIdx * blockSize
                for j in 0..<halfBlock {
                    let u = data[base + j]
                    let v = data[base + j + halfBlock]
                    data[base + j] = frAdd(u, v)
                    data[base + j + halfBlock] = frMul(twiddles[j], frSub(u, v))
                }
            }
        } else if halfBlock >= 64 {
            for blockIdx in 0..<nBlocks {
                let base = blockIdx * blockSize
                let chunkSize = max(1, halfBlock / nThreads)
                DispatchQueue.concurrentPerform(iterations: (halfBlock + chunkSize - 1) / chunkSize) { chunkIdx in
                    let jStart = chunkIdx * chunkSize
                    let jEnd = min(jStart + chunkSize, halfBlock)
                    for j in jStart..<jEnd {
                        let u = data[base + j]
                        let v = data[base + j + halfBlock]
                        data[base + j] = frAdd(u, v)
                        data[base + j + halfBlock] = frMul(twiddles[j], frSub(u, v))
                    }
                }
            }
        } else {
            for blockIdx in 0..<nBlocks {
                let base = blockIdx * blockSize
                for j in 0..<halfBlock {
                    let u = data[base + j]
                    let v = data[base + j + halfBlock]
                    data[base + j] = frAdd(u, v)
                    data[base + j + halfBlock] = frMul(twiddles[j], frSub(u, v))
                }
            }
        }
    }

    // Bit-reverse permutation
    var result = [Fr](repeating: Fr.zero, count: n)
    for i in 0..<n {
        result[bitRev(i, logN)] = data[i]
    }

    // Multiply by n^{-1}
    let nInv = frInverse(frFromInt(UInt64(n)))
    DispatchQueue.concurrentPerform(iterations: n) { i in
        result[i] = frMul(result[i], nInv)
    }
    return result
}

// MARK: - BabyBear parallel NTT

public func parallelNTT_Bb(_ input: [Bb], logN: Int) -> [Bb] {
    let n = input.count
    var data = [Bb](repeating: Bb.zero, count: n)
    for i in 0..<n {
        data[bitRev(i, logN)] = input[i]
    }

    let nThreads = ProcessInfo.processInfo.activeProcessorCount
    let omega = bbRootOfUnity(logN: logN)

    for s in 0..<logN {
        let halfBlock = 1 << s
        let blockSize = halfBlock << 1
        let nBlocks = n / blockSize

        var w_m = Bb.one
        var temp = omega
        var k = n / blockSize
        while k > 0 {
            if k & 1 == 1 { w_m = bbMul(w_m, temp) }
            temp = bbSqr(temp)
            k >>= 1
        }

        var twiddles = [Bb](repeating: Bb.one, count: halfBlock)
        for j in 1..<halfBlock {
            twiddles[j] = bbMul(twiddles[j-1], w_m)
        }

        if nBlocks >= nThreads {
            DispatchQueue.concurrentPerform(iterations: nBlocks) { blockIdx in
                let base = blockIdx * blockSize
                for j in 0..<halfBlock {
                    let u = data[base + j]
                    let v = bbMul(twiddles[j], data[base + j + halfBlock])
                    data[base + j] = bbAdd(u, v)
                    data[base + j + halfBlock] = bbSub(u, v)
                }
            }
        } else if halfBlock >= 64 {
            for blockIdx in 0..<nBlocks {
                let base = blockIdx * blockSize
                let chunkSize = max(1, halfBlock / nThreads)
                DispatchQueue.concurrentPerform(iterations: (halfBlock + chunkSize - 1) / chunkSize) { chunkIdx in
                    let jStart = chunkIdx * chunkSize
                    let jEnd = min(jStart + chunkSize, halfBlock)
                    for j in jStart..<jEnd {
                        let u = data[base + j]
                        let v = bbMul(twiddles[j], data[base + j + halfBlock])
                        data[base + j] = bbAdd(u, v)
                        data[base + j + halfBlock] = bbSub(u, v)
                    }
                }
            }
        } else {
            for blockIdx in 0..<nBlocks {
                let base = blockIdx * blockSize
                for j in 0..<halfBlock {
                    let u = data[base + j]
                    let v = bbMul(twiddles[j], data[base + j + halfBlock])
                    data[base + j] = bbAdd(u, v)
                    data[base + j + halfBlock] = bbSub(u, v)
                }
            }
        }
    }
    return data
}

// MARK: - Goldilocks parallel NTT

public func parallelNTT_Gl(_ input: [Gl], logN: Int) -> [Gl] {
    let n = input.count
    var data = [Gl](repeating: Gl.zero, count: n)
    for i in 0..<n {
        data[bitRev(i, logN)] = input[i]
    }

    let nThreads = ProcessInfo.processInfo.activeProcessorCount
    let omega = glRootOfUnity(logN: logN)

    for s in 0..<logN {
        let halfBlock = 1 << s
        let blockSize = halfBlock << 1
        let nBlocks = n / blockSize

        var w_m = Gl.one
        var temp = omega
        var k = n / blockSize
        while k > 0 {
            if k & 1 == 1 { w_m = glMul(w_m, temp) }
            temp = glSqr(temp)
            k >>= 1
        }

        var twiddles = [Gl](repeating: Gl.one, count: halfBlock)
        for j in 1..<halfBlock {
            twiddles[j] = glMul(twiddles[j-1], w_m)
        }

        if nBlocks >= nThreads {
            DispatchQueue.concurrentPerform(iterations: nBlocks) { blockIdx in
                let base = blockIdx * blockSize
                for j in 0..<halfBlock {
                    let u = data[base + j]
                    let v = glMul(twiddles[j], data[base + j + halfBlock])
                    data[base + j] = glAdd(u, v)
                    data[base + j + halfBlock] = glSub(u, v)
                }
            }
        } else if halfBlock >= 64 {
            for blockIdx in 0..<nBlocks {
                let base = blockIdx * blockSize
                let chunkSize = max(1, halfBlock / nThreads)
                DispatchQueue.concurrentPerform(iterations: (halfBlock + chunkSize - 1) / chunkSize) { chunkIdx in
                    let jStart = chunkIdx * chunkSize
                    let jEnd = min(jStart + chunkSize, halfBlock)
                    for j in jStart..<jEnd {
                        let u = data[base + j]
                        let v = glMul(twiddles[j], data[base + j + halfBlock])
                        data[base + j] = glAdd(u, v)
                        data[base + j + halfBlock] = glSub(u, v)
                    }
                }
            }
        } else {
            for blockIdx in 0..<nBlocks {
                let base = blockIdx * blockSize
                for j in 0..<halfBlock {
                    let u = data[base + j]
                    let v = glMul(twiddles[j], data[base + j + halfBlock])
                    data[base + j] = glAdd(u, v)
                    data[base + j + halfBlock] = glSub(u, v)
                }
            }
        }
    }
    return data
}

// MARK: - BabyBear NEON NTT (C/ARM NEON intrinsics)

/// Forward NTT on BabyBear using ARM NEON intrinsics (4-wide SIMD).
/// Significantly faster than scalar Swift due to NEON Barrett multiplication.
public func neonNTT_Bb(_ input: [Bb], logN: Int) -> [Bb] {
    let n = input.count
    precondition(n == 1 << logN, "Input size must be 2^logN")
    var data = input
    data.withUnsafeMutableBytes { buf in
        let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt32.self)
        babybear_ntt_neon(ptr, Int32(logN))
    }
    return data
}

/// Inverse NTT on BabyBear using ARM NEON intrinsics (4-wide SIMD).
public func neonINTT_Bb(_ input: [Bb], logN: Int) -> [Bb] {
    let n = input.count
    precondition(n == 1 << logN, "Input size must be 2^logN")
    var data = input
    data.withUnsafeMutableBytes { buf in
        let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt32.self)
        babybear_intt_neon(ptr, Int32(logN))
    }
    return data
}

// MARK: - Goldilocks optimized C NTT

/// Forward NTT on Goldilocks using optimized ARM64 scalar C (__uint128_t mul pipelining).
public func cNTT_Gl(_ input: [Gl], logN: Int) -> [Gl] {
    let n = input.count
    precondition(n == 1 << logN, "Input size must be 2^logN")
    var data = input
    data.withUnsafeMutableBytes { buf in
        let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt64.self)
        goldilocks_ntt(ptr, Int32(logN))
    }
    return data
}

/// Inverse NTT on Goldilocks using optimized ARM64 scalar C.
public func cINTT_Gl(_ input: [Gl], logN: Int) -> [Gl] {
    let n = input.count
    precondition(n == 1 << logN, "Input size must be 2^logN")
    var data = input
    data.withUnsafeMutableBytes { buf in
        let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt64.self)
        goldilocks_intt(ptr, Int32(logN))
    }
    return data
}

// MARK: - BN254 Fr optimized C NTT

/// Forward NTT on BN254 Fr using optimized ARM64 C (unrolled 4-limb CIOS Montgomery).
/// Input elements are Fr structs (8x32-bit limbs), which are reinterpreted as 4x64-bit limbs.
public func cNTT_Fr(_ input: [Fr], logN: Int) -> [Fr] {
    let n = input.count
    precondition(n == 1 << logN, "Input size must be 2^logN")
    var data = input
    data.withUnsafeMutableBytes { buf in
        let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt64.self)
        bn254_fr_ntt(ptr, Int32(logN))
    }
    return data
}

/// Inverse NTT on BN254 Fr using optimized ARM64 C.
public func cINTT_Fr(_ input: [Fr], logN: Int) -> [Fr] {
    let n = input.count
    precondition(n == 1 << logN, "Input size must be 2^logN")
    var data = input
    data.withUnsafeMutableBytes { buf in
        let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt64.self)
        bn254_fr_intt(ptr, Int32(logN))
    }
    return data
}
