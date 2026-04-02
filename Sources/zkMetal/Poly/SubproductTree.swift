// Subproduct Tree for Fast Multi-Point Polynomial Evaluation
// Algorithm: O(n log²n) instead of O(n²) Horner
// Reference: von zur Gathen & Gerhard, "Modern Computer Algebra"
import Foundation
import Metal

extension PolyEngine {

    // MARK: - Subproduct Tree Multi-Point Evaluation

    public func evaluateTree(_ coeffs: [Fr], at points: [Fr]) throws -> [Fr] {
        let n = points.count
        if n <= 256 || coeffs.count <= 256 {
            return try evaluate(coeffs, at: points)
        }

        let logN = ceilLog2(n)
        let N = 1 << logN

        var pts = points
        while pts.count < N { pts.append(Fr.one) }

        var c = coeffs
        while c.count < N { c.append(Fr.zero) }

        let tree = try buildSubproductTree(pts, logN: logN)
        let results = try remainderTreeDescent(poly: c, tree: tree, logN: logN)
        return Array(results.prefix(n))
    }

    // MARK: - GPU-Resident Polynomial Multiply

    /// Multiply two polynomials on GPU. Inputs are padded arrays of size nttN = 2^logN.
    /// Returns result buffer of size nttN. Single command buffer submit.
    private func multiplyGPU(_ a: [Fr], _ b: [Fr], nttLogN: Int) throws -> [Fr] {
        let nttN = 1 << nttLogN
        let stride = MemoryLayout<Fr>.stride

        // Pad to NTT size
        var aPad = a
        while aPad.count < nttN { aPad.append(Fr.zero) }
        var bPad = b
        while bPad.count < nttN { bPad.append(Fr.zero) }

        let aBuf = createBuffer(aPad)
        let bBuf = createBuffer(bPad)

        guard let cmdBuf = nttEngine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        nttEngine.encodeNTT(data: aBuf, logN: nttLogN, cmdBuf: cmdBuf)
        nttEngine.encodeNTT(data: bBuf, logN: nttLogN, cmdBuf: cmdBuf)

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(hadamardFunction)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(aBuf, offset: 0, index: 2)
        var nVal = UInt32(nttN)
        enc.setBytes(&nVal, length: 4, index: 3)
        let tg = min(tuning.nttThreadgroupSize, Int(hadamardFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: nttN, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        nttEngine.encodeINTT(data: aBuf, logN: nttLogN, cmdBuf: cmdBuf)

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }

        return readBuffer(aBuf, count: nttN)
    }

    /// NTT size needed for multiplying two polys of given sizes.
    private func nttLogNForMultiply(_ aSize: Int, _ bSize: Int) -> Int {
        ceilLog2(aSize + bSize - 1)
    }

    // MARK: - Polynomial Inverse (Newton iteration)

    /// Compute h such that f * h ≡ 1 (mod x^modDeg).
    /// Encodes ALL Newton steps into ONE command buffer.
    /// Uses max NTT size throughout with zero-truncation between steps.
    func polyInverse(f: [Fr], modDeg: Int) throws -> [Fr] {
        precondition(!f.isEmpty && !f[0].isZero)

        // Use max NTT size for all steps (wastes compute on early steps but only 1 submit)
        let maxMulSize = modDeg * 2
        let nttLogN = ceilLog2(maxMulSize)
        let nttN = 1 << nttLogN
        let stride = MemoryLayout<Fr>.stride
        let tg = 256

        // Prepare f buffer (padded to nttN)
        var fPad = Array(f.prefix(modDeg))
        while fPad.count < nttN { fPad.append(Fr.zero) }
        let fBuf = createBuffer(fPad)

        // h starts as [1/f[0], 0, 0, ...]
        var hInit = [Fr](repeating: Fr.zero, count: nttN)
        hInit[0] = frInverse(f[0])
        let hBuf = createBuffer(hInit)

        // Working buffers
        let hCopyBuf = device.makeBuffer(length: nttN * stride, options: .storageModeShared)!
        let fNTTBuf = device.makeBuffer(length: nttN * stride, options: .storageModeShared)!
        let prodBuf = device.makeBuffer(length: nttN * stride, options: .storageModeShared)!
        let tmfhBuf = device.makeBuffer(length: nttN * stride, options: .storageModeShared)!

        guard let cmdBuf = nttEngine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var precision = 1
        while precision < modDeg {
            let nextPrecision = min(precision * 2, modDeg)

            // Copy f to fNTTBuf (we need fBuf intact for every step)
            let blitF = cmdBuf.makeBlitCommandEncoder()!
            blitF.copy(from: fBuf, sourceOffset: 0, to: fNTTBuf, destinationOffset: 0, size: nttN * stride)
            // Copy h to hCopyBuf (NTT is in-place, we need original h for step 3)
            blitF.copy(from: hBuf, sourceOffset: 0, to: hCopyBuf, destinationOffset: 0, size: nttN * stride)
            blitF.endEncoding()

            // Step 1: NTT(f), NTT(h), Hadamard → prodBuf, iNTT → f*h
            nttEngine.encodeNTT(data: fNTTBuf, logN: nttLogN, cmdBuf: cmdBuf)
            nttEngine.encodeNTT(data: hBuf, logN: nttLogN, cmdBuf: cmdBuf)

            let encH1 = cmdBuf.makeComputeCommandEncoder()!
            encH1.setComputePipelineState(hadamardFunction)
            encH1.setBuffer(fNTTBuf, offset: 0, index: 0)
            encH1.setBuffer(hBuf, offset: 0, index: 1)
            encH1.setBuffer(prodBuf, offset: 0, index: 2)
            var nttNVal = UInt32(nttN)
            encH1.setBytes(&nttNVal, length: 4, index: 3)
            encH1.dispatchThreads(MTLSize(width: nttN, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            encH1.endEncoding()
            nttEngine.encodeINTT(data: prodBuf, logN: nttLogN, cmdBuf: cmdBuf)

            // Step 2: Compute 2 - f*h → tmfhBuf
            let encTM = cmdBuf.makeComputeCommandEncoder()!
            encTM.setComputePipelineState(twoMinusFunction!)
            encTM.setBuffer(prodBuf, offset: 0, index: 0)
            encTM.setBuffer(tmfhBuf, offset: 0, index: 1)
            encTM.setBytes(&nttNVal, length: 4, index: 2)
            var polyStrideVal = UInt32(nttN)
            encTM.setBytes(&polyStrideVal, length: 4, index: 3)
            encTM.dispatchThreads(MTLSize(width: nttN, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            encTM.endEncoding()

            // Step 3: NTT(hCopy), NTT(tmfh), Hadamard → hBuf, iNTT → new h
            nttEngine.encodeNTT(data: hCopyBuf, logN: nttLogN, cmdBuf: cmdBuf)
            nttEngine.encodeNTT(data: tmfhBuf, logN: nttLogN, cmdBuf: cmdBuf)
            let encH2 = cmdBuf.makeComputeCommandEncoder()!
            encH2.setComputePipelineState(hadamardFunction)
            encH2.setBuffer(hCopyBuf, offset: 0, index: 0)
            encH2.setBuffer(tmfhBuf, offset: 0, index: 1)
            encH2.setBuffer(hBuf, offset: 0, index: 2)  // result goes back to hBuf
            encH2.setBytes(&nttNVal, length: 4, index: 3)
            encH2.dispatchThreads(MTLSize(width: nttN, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            encH2.endEncoding()
            nttEngine.encodeINTT(data: hBuf, logN: nttLogN, cmdBuf: cmdBuf)

            // Truncate: zero out h[nextPrecision..nttN) to maintain mod x^nextPrecision invariant
            if nextPrecision < nttN {
                let blitZ = cmdBuf.makeBlitCommandEncoder()!
                blitZ.fill(buffer: hBuf, range: nextPrecision * stride..<nttN * stride, value: 0)
                blitZ.endEncoding()
            }

            precision = nextPrecision
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }

        return Array(readBuffer(hBuf, count: nttN).prefix(modDeg))
    }

    // MARK: - Polynomial Remainder

    func polyRemainder(f: [Fr], g: [Fr]) throws -> [Fr] {
        let degF = f.count - 1
        let degG = g.count - 1

        if degF < degG {
            return f
        }

        let quotientDeg = degF - degG
        let quotientSize = quotientDeg + 1

        let revG = Array(g.reversed())

        // Compute inverse of rev(g) mod x^quotientSize (single submit for all Newton steps)
        let invRevG = try polyInverse(f: revG, modDeg: quotientSize)

        // Quotient and remainder in one command buffer:
        // q_rev = rev(f) * invRevG (truncated), then q * g, then f - q*g
        let revF = Array(f.reversed())
        let revFTrunc = Array(revF.prefix(quotientSize))

        // Use a single NTT size for both multiplies (quotient and q*g)
        let maxSize = max(quotientSize + invRevG.count - 1, quotientSize + g.count - 1)
        let nttLogN = ceilLog2(maxSize)
        let nttN = 1 << nttLogN
        let stride = MemoryLayout<Fr>.stride

        // Multiply 1: revF * invRevG → quotient (reversed)
        let qRevProd = try multiplyGPU(revFTrunc, invRevG, nttLogN: nttLogN)
        let qRev = Array(qRevProd.prefix(quotientSize))
        let q = Array(qRev.reversed())

        // Multiply 2: q * g → then subtract from f
        let qg = try multiplyGPU(q, g, nttLogN: nttLogN)
        var r = [Fr](repeating: Fr.zero, count: degG)
        for i in 0..<degG {
            r[i] = frSub(f[i], i < qg.count ? qg[i] : Fr.zero)
        }

        return r
    }

    // MARK: - Tree Build (bottom-up)

    private func buildSubproductTree(_ points: [Fr], logN: Int) throws -> [MTLBuffer] {
        let N = 1 << logN
        var tree = [MTLBuffer]()

        // Level 0: linear factors (x - p_i), stored as N × 2 elements
        var leaves = [Fr]()
        leaves.reserveCapacity(N * 2)
        for i in 0..<N {
            leaves.append(frSub(Fr.zero, points[i]))
            leaves.append(Fr.one)
        }
        tree.append(createBuffer(leaves))

        // Level 1: linear pairs → degree-2 polys (N/2 × 3 coefficients)
        let level1Buf = device.makeBuffer(length: (N / 2) * 3 * MemoryLayout<Fr>.stride,
                                          options: .storageModeShared)!
        try dispatchLinearPairs(points: createBuffer(points), output: level1Buf, n: N)
        tree.append(level1Buf)

        // Levels 2..logN: schoolbook for small, NTT for large
        let schoolbookMaxDeg = 256
        for k in 2...logN {
            let numPolys = N >> k
            let inDeg = 1 << (k - 1)
            let outDeg = 1 << k
            let inSize = inDeg + 1
            let outSize = outDeg + 1

            if outDeg <= schoolbookMaxDeg {
                let outBuf = device.makeBuffer(length: numPolys * outSize * MemoryLayout<Fr>.stride,
                                               options: .storageModeShared)!
                let leftBuf = device.makeBuffer(length: numPolys * inSize * MemoryLayout<Fr>.stride,
                                                options: .storageModeShared)!
                let rightBuf = device.makeBuffer(length: numPolys * inSize * MemoryLayout<Fr>.stride,
                                                 options: .storageModeShared)!
                splitLR(prev: tree[k - 1], left: leftBuf, right: rightBuf,
                        numPolys: numPolys, polySize: inSize)
                try dispatchSchoolbookMultiply(left: leftBuf, right: rightBuf, output: outBuf,
                                               dPlus1: UInt32(inSize), outSize: UInt32(outSize),
                                               count: UInt32(numPolys))
                tree.append(outBuf)
            } else {
                let outBuf = try nttMultiplyLevel(prev: tree[k - 1], numPolys: numPolys,
                                                   inSize: inSize, outSize: outSize, outDeg: outDeg)
                tree.append(outBuf)
            }
        }

        return tree
    }

    private func splitLR(prev: MTLBuffer, left: MTLBuffer, right: MTLBuffer,
                         numPolys: Int, polySize: Int) {
        let stride = MemoryLayout<Fr>.stride
        let srcPtr = prev.contents().bindMemory(to: Fr.self, capacity: numPolys * 2 * polySize)
        let leftPtr = left.contents().bindMemory(to: Fr.self, capacity: numPolys * polySize)
        let rightPtr = right.contents().bindMemory(to: Fr.self, capacity: numPolys * polySize)
        for i in 0..<numPolys {
            memcpy(leftPtr.advanced(by: i * polySize),
                   srcPtr.advanced(by: (2 * i) * polySize),
                   polySize * stride)
            memcpy(rightPtr.advanced(by: i * polySize),
                   srcPtr.advanced(by: (2 * i + 1) * polySize),
                   polySize * stride)
        }
    }

    private func nttMultiplyLevel(prev: MTLBuffer, numPolys: Int,
                                   inSize: Int, outSize: Int, outDeg: Int) throws -> MTLBuffer {
        let nttLogN = ceilLog2(outSize)
        let nttN = 1 << nttLogN
        let stride = MemoryLayout<Fr>.stride
        let prevPtr = prev.contents().bindMemory(to: Fr.self, capacity: numPolys * 2 * inSize)

        let aBuf = device.makeBuffer(length: numPolys * nttN * stride, options: .storageModeShared)!
        let bBuf = device.makeBuffer(length: numPolys * nttN * stride, options: .storageModeShared)!
        memset(aBuf.contents(), 0, numPolys * nttN * stride)
        memset(bBuf.contents(), 0, numPolys * nttN * stride)

        let aPtr = aBuf.contents().bindMemory(to: Fr.self, capacity: numPolys * nttN)
        let bPtr = bBuf.contents().bindMemory(to: Fr.self, capacity: numPolys * nttN)

        for i in 0..<numPolys {
            memcpy(aPtr.advanced(by: i * nttN), prevPtr.advanced(by: (2 * i) * inSize), inSize * stride)
            memcpy(bPtr.advanced(by: i * nttN), prevPtr.advanced(by: (2 * i + 1) * inSize), inSize * stride)
        }

        guard let cmdBuf = nttEngine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        for i in 0..<numPolys {
            let offset = i * nttN * stride
            nttEngine.encodeNTT(data: aBuf, offset: offset, logN: nttLogN, cmdBuf: cmdBuf)
            nttEngine.encodeNTT(data: bBuf, offset: offset, logN: nttLogN, cmdBuf: cmdBuf)
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(hadamardFunction)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(aBuf, offset: 0, index: 2)
        var totalN = UInt32(numPolys * nttN)
        enc.setBytes(&totalN, length: 4, index: 3)
        let tg = min(tuning.nttThreadgroupSize, Int(hadamardFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: Int(totalN), height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        for i in 0..<numPolys {
            nttEngine.encodeINTT(data: aBuf, offset: i * nttN * stride, logN: nttLogN, cmdBuf: cmdBuf)
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }

        let outBuf = device.makeBuffer(length: numPolys * outSize * stride, options: .storageModeShared)!
        let outPtr = outBuf.contents().bindMemory(to: Fr.self, capacity: numPolys * outSize)
        let resPtr = aBuf.contents().bindMemory(to: Fr.self, capacity: numPolys * nttN)
        for i in 0..<numPolys {
            memcpy(outPtr.advanced(by: i * outSize), resPtr.advanced(by: i * nttN), outSize * stride)
        }

        return outBuf
    }

    // MARK: - Remainder Tree Descent (top-down)

    private func remainderTreeDescent(poly: [Fr], tree: [MTLBuffer], logN: Int) throws -> [Fr] {
        let N = 1 << logN
        let stride = MemoryLayout<Fr>.stride
        var remainders = [MTLBuffer]()

        let rootBuf = createBuffer(poly)
        remainders.append(rootBuf)

        for k in 0..<logN {
            let numParents = 1 << k
            let parentSize = N >> k
            let childDeg = N >> (k + 1)
            let childSize = childDeg + 1
            let childRemSize = max(childDeg, 1)

            let treeLevel = logN - k - 1
            let childNodes = tree[treeLevel]

            let numChildren = numParents * 2
            let nextBuf = device.makeBuffer(length: numChildren * childRemSize * stride,
                                            options: .storageModeShared)!

            if childDeg <= 64 {
                try computeRemaindersSchoolbook(
                    parents: remainders[k], parentSize: parentSize,
                    childNodes: childNodes, childSize: childSize,
                    output: nextBuf, childRemSize: childRemSize,
                    numParents: numParents)
            } else {
                try computeRemaindersNTT(
                    parents: remainders[k], parentSize: parentSize,
                    childNodes: childNodes, childSize: childSize,
                    output: nextBuf, childRemSize: childRemSize,
                    numParents: numParents)
            }

            remainders.append(nextBuf)
        }

        let lastBuf = remainders[logN]
        let ptr = lastBuf.contents().bindMemory(to: Fr.self, capacity: N)
        return Array(UnsafeBufferPointer(start: ptr, count: N))
    }

    private func computeRemaindersSchoolbook(
        parents: MTLBuffer, parentSize: Int,
        childNodes: MTLBuffer, childSize: Int,
        output: MTLBuffer, childRemSize: Int,
        numParents: Int
    ) throws {
        let stride = MemoryLayout<Fr>.stride
        let numChildren = numParents * 2
        let fSize = parentSize

        let fBuf = device.makeBuffer(length: numChildren * fSize * stride, options: .storageModeShared)!
        let fPtr = fBuf.contents().bindMemory(to: Fr.self, capacity: numChildren * fSize)
        let pPtr = parents.contents().bindMemory(to: Fr.self, capacity: numParents * parentSize)

        for i in 0..<numParents {
            memcpy(fPtr.advanced(by: (2 * i) * fSize), pPtr.advanced(by: i * parentSize), fSize * stride)
            memcpy(fPtr.advanced(by: (2 * i + 1) * fSize), pPtr.advanced(by: i * parentSize), fSize * stride)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(treeRemainderSchoolbookFunction!)
        enc.setBuffer(fBuf, offset: 0, index: 0)
        enc.setBuffer(childNodes, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var fSizeVal = UInt32(fSize), gSizeVal = UInt32(childSize)
        var outSizeVal = UInt32(childRemSize), countVal = UInt32(numChildren)
        enc.setBytes(&fSizeVal, length: 4, index: 3)
        enc.setBytes(&gSizeVal, length: 4, index: 4)
        enc.setBytes(&outSizeVal, length: 4, index: 5)
        enc.setBytes(&countVal, length: 4, index: 6)
        let tg = min(tuning.nttThreadgroupSize, Int(treeRemainderSchoolbookFunction!.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: numChildren, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }
    }

    /// Batched NTT-based remainder for an entire tree level.
    /// Computes all 2*numParents remainders using batched GPU operations.
    /// Total submits: 3 (batched inverse + batched quotient multiply + batched qg multiply).
    private func computeRemaindersNTT(
        parents: MTLBuffer, parentSize: Int,
        childNodes: MTLBuffer, childSize: Int,
        output: MTLBuffer, childRemSize: Int,
        numParents: Int
    ) throws {
        let stride = MemoryLayout<Fr>.stride
        let numChildren = numParents * 2
        let childDeg = childSize - 1
        let quotientSize = parentSize - childDeg  // parentSize - 1 - childDeg + 1

        // NTT size for multiplications
        let maxProdSize = max(quotientSize * 2, quotientSize + childSize - 1)
        let nttLogN = ceilLog2(maxProdSize)
        let nttN = 1 << nttLogN

        // Step 1: Prepare all reversed divisors and compute batched inverses
        // Extract and reverse all child polynomials, then batch-compute their inverses
        var allRevG = [Fr](repeating: Fr.zero, count: numChildren * nttN)
        let cPtr = childNodes.contents().bindMemory(to: Fr.self, capacity: numChildren * childSize)
        for c in 0..<numChildren {
            for i in 0..<childSize {
                allRevG[c * nttN + i] = cPtr[c * childSize + (childSize - 1 - i)]  // reverse
            }
        }

        let allInv = try polyInverseBatch(fs: allRevG, count: numChildren,
                                           polyStride: nttN, modDeg: quotientSize, nttLogN: nttLogN)

        // Step 2: Prepare all reversed dividends (truncated to quotientSize) and compute quotients
        var allRevFTrunc = [Fr](repeating: Fr.zero, count: numChildren * nttN)
        let pPtr = parents.contents().bindMemory(to: Fr.self, capacity: numParents * parentSize)
        for p in 0..<numParents {
            for side in 0..<2 {
                let c = p * 2 + side
                // rev(f) truncated to quotientSize
                for i in 0..<min(quotientSize, parentSize) {
                    allRevFTrunc[c * nttN + i] = pPtr[p * parentSize + (parentSize - 1 - i)]
                }
            }
        }

        // Multiply: revFTrunc[c] * inv[c] for all c → quotient (reversed)
        let allQRev = try batchMultiplyGPU(a: allRevFTrunc, b: readBuffer(allInv, count: numChildren * nttN),
                                            count: numChildren, polyStride: nttN, nttLogN: nttLogN)

        // Reverse quotients and prepare for q*g multiplication
        var allQ = [Fr](repeating: Fr.zero, count: numChildren * nttN)
        for c in 0..<numChildren {
            for i in 0..<quotientSize {
                allQ[c * nttN + i] = allQRev[c * nttN + (quotientSize - 1 - i)]
            }
        }

        // Step 3: Multiply q * g for all children
        var allG = [Fr](repeating: Fr.zero, count: numChildren * nttN)
        for c in 0..<numChildren {
            for i in 0..<childSize {
                allG[c * nttN + i] = cPtr[c * childSize + i]
            }
        }

        let allQG = try batchMultiplyGPU(a: allQ, b: allG, count: numChildren,
                                          polyStride: nttN, nttLogN: nttLogN)

        // Step 4: Compute remainders r[c] = f[parent(c)] - qg[c], first childDeg coefficients
        let oPtr = output.contents().bindMemory(to: Fr.self, capacity: numChildren * childRemSize)
        for p in 0..<numParents {
            for side in 0..<2 {
                let c = p * 2 + side
                for i in 0..<childRemSize {
                    let fi = i < parentSize ? pPtr[p * parentSize + i] : Fr.zero
                    let qgi = allQG[c * nttN + i]
                    oPtr[c * childRemSize + i] = frSub(fi, qgi)
                }
            }
        }
    }

    // MARK: - Batched Polynomial Inverse

    /// Compute inverses of `count` polynomials simultaneously.
    /// All polynomials stored in `fs` with stride `polyStride` elements.
    /// Returns buffer with `count * polyStride` elements (inverses padded with zeros).
    private func polyInverseBatch(fs: [Fr], count: Int, polyStride: Int,
                                   modDeg: Int, nttLogN: Int) throws -> MTLBuffer {
        let nttN = 1 << nttLogN
        let stride = MemoryLayout<Fr>.stride
        let tg = 256

        precondition(polyStride == nttN)

        // f buffers (constant across Newton steps)
        let fBuf = createBuffer(fs)

        // h buffers: initialize all inverses with 1/f[0]
        var hInit = [Fr](repeating: Fr.zero, count: count * nttN)
        for c in 0..<count {
            let f0 = fs[c * nttN]
            precondition(!f0.isZero, "f[0] must be nonzero")
            hInit[c * nttN] = frInverse(f0)
        }
        let hBuf = createBuffer(hInit)

        // Working buffers
        let fNTTBuf = device.makeBuffer(length: count * nttN * stride, options: .storageModeShared)!
        let hCopyBuf = device.makeBuffer(length: count * nttN * stride, options: .storageModeShared)!
        let prodBuf = device.makeBuffer(length: count * nttN * stride, options: .storageModeShared)!
        let tmfhBuf = device.makeBuffer(length: count * nttN * stride, options: .storageModeShared)!

        guard let cmdBuf = nttEngine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var precision = 1
        while precision < modDeg {
            let nextPrecision = min(precision * 2, modDeg)

            // Copy f → fNTT, h → hCopy
            let blit1 = cmdBuf.makeBlitCommandEncoder()!
            blit1.copy(from: fBuf, sourceOffset: 0, to: fNTTBuf,
                      destinationOffset: 0, size: count * nttN * stride)
            blit1.copy(from: hBuf, sourceOffset: 0, to: hCopyBuf,
                      destinationOffset: 0, size: count * nttN * stride)
            blit1.endEncoding()

            // NTT all f's and h's
            for c in 0..<count {
                let off = c * nttN * stride
                nttEngine.encodeNTT(data: fNTTBuf, offset: off, logN: nttLogN, cmdBuf: cmdBuf)
                nttEngine.encodeNTT(data: hBuf, offset: off, logN: nttLogN, cmdBuf: cmdBuf)
            }

            // Hadamard: prodBuf = fNTT * hNTT (all at once since contiguous)
            let encH1 = cmdBuf.makeComputeCommandEncoder()!
            encH1.setComputePipelineState(hadamardFunction)
            encH1.setBuffer(fNTTBuf, offset: 0, index: 0)
            encH1.setBuffer(hBuf, offset: 0, index: 1)
            encH1.setBuffer(prodBuf, offset: 0, index: 2)
            var totalN = UInt32(count * nttN)
            encH1.setBytes(&totalN, length: 4, index: 3)
            encH1.dispatchThreads(MTLSize(width: Int(totalN), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            encH1.endEncoding()

            // iNTT all products
            for c in 0..<count {
                nttEngine.encodeINTT(data: prodBuf, offset: c * nttN * stride, logN: nttLogN, cmdBuf: cmdBuf)
            }

            // 2 - f*h for all (each poly has nttN elements)
            let encTM = cmdBuf.makeComputeCommandEncoder()!
            encTM.setComputePipelineState(twoMinusFunction!)
            encTM.setBuffer(prodBuf, offset: 0, index: 0)
            encTM.setBuffer(tmfhBuf, offset: 0, index: 1)
            encTM.setBytes(&totalN, length: 4, index: 2)
            var polyStrideVal = UInt32(nttN)
            encTM.setBytes(&polyStrideVal, length: 4, index: 3)
            encTM.dispatchThreads(MTLSize(width: Int(totalN), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            encTM.endEncoding()

            // NTT hCopy and tmfh
            for c in 0..<count {
                let off = c * nttN * stride
                nttEngine.encodeNTT(data: hCopyBuf, offset: off, logN: nttLogN, cmdBuf: cmdBuf)
                nttEngine.encodeNTT(data: tmfhBuf, offset: off, logN: nttLogN, cmdBuf: cmdBuf)
            }

            // Hadamard: hBuf = hCopy * tmfh
            let encH2 = cmdBuf.makeComputeCommandEncoder()!
            encH2.setComputePipelineState(hadamardFunction)
            encH2.setBuffer(hCopyBuf, offset: 0, index: 0)
            encH2.setBuffer(tmfhBuf, offset: 0, index: 1)
            encH2.setBuffer(hBuf, offset: 0, index: 2)
            encH2.setBytes(&totalN, length: 4, index: 3)
            encH2.dispatchThreads(MTLSize(width: Int(totalN), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            encH2.endEncoding()

            // iNTT all new h's
            for c in 0..<count {
                nttEngine.encodeINTT(data: hBuf, offset: c * nttN * stride, logN: nttLogN, cmdBuf: cmdBuf)
            }

            // Truncate: zero h[nextPrecision..nttN) for each polynomial
            if nextPrecision < nttN {
                let blitZ = cmdBuf.makeBlitCommandEncoder()!
                for c in 0..<count {
                    let rangeStart = (c * nttN + nextPrecision) * stride
                    let rangeEnd = (c * nttN + nttN) * stride
                    blitZ.fill(buffer: hBuf, range: rangeStart..<rangeEnd, value: 0)
                }
                blitZ.endEncoding()
            }

            precision = nextPrecision
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }

        return hBuf
    }

    /// Batch multiply: a[c] * b[c] for c in 0..<count, all at same NTT size.
    /// Returns flat array of count * polyStride results.
    private func batchMultiplyGPU(a: [Fr], b: [Fr], count: Int,
                                   polyStride: Int, nttLogN: Int) throws -> [Fr] {
        let nttN = 1 << nttLogN
        let stride = MemoryLayout<Fr>.stride
        let tg = 256

        let aBuf = createBuffer(a)
        let bBuf = createBuffer(b)

        guard let cmdBuf = nttEngine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        for c in 0..<count {
            let off = c * nttN * stride
            nttEngine.encodeNTT(data: aBuf, offset: off, logN: nttLogN, cmdBuf: cmdBuf)
            nttEngine.encodeNTT(data: bBuf, offset: off, logN: nttLogN, cmdBuf: cmdBuf)
        }

        let encH = cmdBuf.makeComputeCommandEncoder()!
        encH.setComputePipelineState(hadamardFunction)
        encH.setBuffer(aBuf, offset: 0, index: 0)
        encH.setBuffer(bBuf, offset: 0, index: 1)
        encH.setBuffer(aBuf, offset: 0, index: 2)
        var totalN = UInt32(count * nttN)
        encH.setBytes(&totalN, length: 4, index: 3)
        encH.dispatchThreads(MTLSize(width: Int(totalN), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        encH.endEncoding()

        for c in 0..<count {
            nttEngine.encodeINTT(data: aBuf, offset: c * nttN * stride, logN: nttLogN, cmdBuf: cmdBuf)
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }

        return readBuffer(aBuf, count: count * nttN)
    }

    // MARK: - Helpers

    func ceilLog2(_ n: Int) -> Int {
        var logN = 0; var m = 1
        while m < n { m <<= 1; logN += 1 }
        return logN
    }

    // MARK: - GPU Dispatch Helpers

    private func dispatchLinearPairs(points: MTLBuffer, output: MTLBuffer, n: Int) throws {
        guard let fn = treeBuildLinearPairsFunction else { throw MSMError.missingKernel }
        guard let cmdBuf = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fn)
        enc.setBuffer(points, offset: 0, index: 0)
        enc.setBuffer(output, offset: 0, index: 1)
        var nVal = UInt32(n)
        enc.setBytes(&nVal, length: 4, index: 2)
        let tg = min(tuning.nttThreadgroupSize, Int(fn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n / 2, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }
    }

    private func dispatchSchoolbookMultiply(left: MTLBuffer, right: MTLBuffer, output: MTLBuffer,
                                             dPlus1: UInt32, outSize: UInt32, count: UInt32) throws {
        guard let fn = treeBuildSchoolbookFunction else { throw MSMError.missingKernel }
        guard let cmdBuf = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fn)
        enc.setBuffer(left, offset: 0, index: 0)
        enc.setBuffer(right, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var dp1 = dPlus1, os = outSize, cnt = count
        enc.setBytes(&dp1, length: 4, index: 3)
        enc.setBytes(&os, length: 4, index: 4)
        enc.setBytes(&cnt, length: 4, index: 5)
        let totalThreads = Int(count) * Int(outSize)
        let tg = min(tuning.nttThreadgroupSize, Int(fn.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: totalThreads, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }
    }
}
