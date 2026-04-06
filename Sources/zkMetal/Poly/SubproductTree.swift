// Subproduct Tree for Fast Multi-Point Polynomial Evaluation & Interpolation
// Algorithm: O(n log²n) instead of O(n²) Horner / Lagrange
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

    // MARK: - Subproduct Tree Batch Interpolation

    /// Interpolate: given N (point, value) pairs, find polynomial p of degree < N
    /// such that p(points[i]) = values[i] for all i.
    /// Uses O(n log²n) algorithm via subproduct tree (dual of multi-point evaluation).
    /// Reference: Algorithm 10.11, von zur Gathen & Gerhard.
    public func interpolateTree(points: [Fr], values: [Fr]) throws -> [Fr] {
        precondition(points.count == values.count, "points and values must have same length")
        let n = points.count

        // Fallback to Lagrange for small inputs
        if n <= 256 {
            return lagrangeInterpolate(points: points, values: values)
        }

        let logN = ceilLog2(n)
        let N = 1 << logN

        // Pad points/values to power-of-two
        var pts = points
        while pts.count < N { pts.append(frFromInt(UInt64(pts.count + 1000000))) } // distinct padding points
        var vals = values
        while vals.count < N { vals.append(Fr.zero) }

        // Step 1: Build subproduct tree from points
        let tree = try buildSubproductTree(pts, logN: logN)

        // Step 2: Compute M(x) = product of all (x - z_i), then M'(x) (formal derivative)
        let rootLevel = logN
        let rootSize = N + 1 // degree N polynomial has N+1 coefficients
        let rootPtr = tree[rootLevel].contents().bindMemory(to: Fr.self, capacity: rootSize)
        var rootPoly = [Fr](repeating: Fr.zero, count: rootSize)
        for i in 0..<rootSize { rootPoly[i] = rootPtr[i] }

        let mPrime = polyDerivative(rootPoly)

        // Step 3: Multi-point evaluate M'(x) at all points → weights w_i = M'(z_i)
        var mPrimePadded = mPrime
        while mPrimePadded.count < N { mPrimePadded.append(Fr.zero) }
        let weights = try remainderTreeDescent(poly: mPrimePadded, tree: tree, logN: logN)

        // Step 4: Compute scaled values s_i = y_i / w_i
        // Batch-invert all n weights (padding weights are unused since value=0)
        let activeWeights = Array(weights.prefix(n))
        var wPrefix = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n { wPrefix[i] = frMul(wPrefix[i - 1], activeWeights[i - 1]) }
        var wAcc = frInverse(frMul(wPrefix[n - 1], activeWeights[n - 1]))
        var weightInvs = [Fr](repeating: Fr.zero, count: n)
        for i in Swift.stride(from: n - 1, through: 0, by: -1) {
            weightInvs[i] = frMul(wAcc, wPrefix[i])
            wAcc = frMul(wAcc, activeWeights[i])
        }
        var scaledValues = [Fr](repeating: Fr.zero, count: N)
        for i in 0..<n {
            scaledValues[i] = frMul(vals[i], weightInvs[i])
        }

        // Step 5: Linear combination ascent — build result polynomial bottom-up
        let result = try linearCombinationAscent(scaledValues: scaledValues, tree: tree, logN: logN)

        // Return only the first n coefficients (degree < n)
        return Array(result.prefix(n))
    }

    /// Formal derivative of a polynomial: p'(x) = sum_{i=1}^{deg} i * p[i] * x^{i-1}
    private func polyDerivative(_ p: [Fr]) -> [Fr] {
        if p.count <= 1 { return [Fr.zero] }
        var result = [Fr](repeating: Fr.zero, count: p.count - 1)
        for i in 1..<p.count {
            // Multiply coefficient by index i
            result[i - 1] = frMul(p[i], frFromInt(UInt64(i)))
        }
        return result
    }

    /// CPU Lagrange interpolation for small inputs (O(n²) but fast for n <= 256).
    private func lagrangeInterpolate(points: [Fr], values: [Fr]) -> [Fr] {
        let n = points.count
        if n == 0 { return [] }
        if n == 1 { return [values[0]] }

        var result = [Fr](repeating: Fr.zero, count: n)

        // Precompute all Lagrange denominators and batch-invert
        var denoms = [Fr](repeating: Fr.one, count: n)
        for i in 0..<n {
            for j in 0..<n where j != i {
                denoms[i] = frMul(denoms[i], frSub(points[i], points[j]))
            }
        }
        var dPrefix = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n { dPrefix[i] = frMul(dPrefix[i - 1], denoms[i - 1]) }
        var dAcc = frInverse(frMul(dPrefix[n - 1], denoms[n - 1]))
        var denomInvs = [Fr](repeating: Fr.zero, count: n)
        for i in Swift.stride(from: n - 1, through: 0, by: -1) {
            denomInvs[i] = frMul(dAcc, dPrefix[i])
            dAcc = frMul(dAcc, denoms[i])
        }

        for i in 0..<n {
            let weight = frMul(values[i], denomInvs[i])

            // Build numerator polynomial: product_{j!=i} (x - z_j)
            var basis = [Fr](repeating: Fr.zero, count: n)
            basis[0] = Fr.one
            var basisLen = 1
            for j in 0..<n where j != i {
                let negZj = frSub(Fr.zero, points[j])
                var newBasis = [Fr](repeating: Fr.zero, count: basisLen + 1)
                for k in 0...basisLen {
                    var val = Fr.zero
                    if k > 0 { val = frAdd(val, basis[k - 1]) }
                    if k < basisLen { val = frAdd(val, frMul(negZj, basis[k])) }
                    newBasis[k] = val
                }
                for k in 0..<newBasis.count { basis[k] = newBasis[k] }
                basisLen += 1
            }

            for k in 0..<n {
                result[k] = frAdd(result[k], frMul(weight, basis[k]))
            }
        }

        return result
    }

    /// Linear combination ascent: given scaled values s_i and the subproduct tree,
    /// compute the interpolation polynomial bottom-up.
    /// At each level, combine left and right subtree results:
    ///   result[parent] = left_result * right_subproduct + right_result * left_subproduct
    private func linearCombinationAscent(scaledValues: [Fr], tree: [MTLBuffer], logN: Int) throws -> [Fr] {
        let N = 1 << logN
        let stride = MemoryLayout<Fr>.stride

        // Level 0 (leaves): each polynomial is just the scalar s_i (degree 0)
        // Stored as N polynomials of 1 coefficient each.
        var currentLevel = [Fr](repeating: Fr.zero, count: N)
        for i in 0..<N { currentLevel[i] = scaledValues[i] }

        // Ascend: at level k, we have N/(2^k) polynomials of degree < 2^k
        for k in 0..<logN {
            let numPolys = N >> (k + 1)  // number of parent nodes
            let childDeg = 1 << k        // max degree of child results
            let childSize = childDeg      // coefficients per child result (degree < childDeg means childDeg coeffs)
            let parentDeg = 1 << (k + 1) // max degree of parent result
            let parentSize = parentDeg    // coefficients per parent result

            // Read subproduct tree nodes for this level
            let treeLevel = k  // tree[k] has the subproduct polys at level k
            let treeNodes = tree[treeLevel]

            // Size of each subproduct poly at this level
            let subprodSize: Int
            if k == 0 {
                subprodSize = 2  // linear: [-z_i, 1]
            } else {
                subprodSize = (1 << k) + 1  // degree 2^k → 2^k + 1 coeffs
            }

            let treePtr = treeNodes.contents().bindMemory(to: Fr.self, capacity: (N >> k) * subprodSize)

            if parentDeg <= 64 {
                // CPU schoolbook for small polynomials
                var nextLevel = [Fr](repeating: Fr.zero, count: numPolys * parentSize)
                for p in 0..<numPolys {
                    let leftIdx = 2 * p
                    let rightIdx = 2 * p + 1

                    // left_result = currentLevel[leftIdx * childSize ..< (leftIdx+1) * childSize]
                    // right_result = currentLevel[rightIdx * childSize ..< (rightIdx+1) * childSize]
                    // left_subprod = treePtr[leftIdx * subprodSize ..< (leftIdx+1) * subprodSize]
                    // right_subprod = treePtr[rightIdx * subprodSize ..< (rightIdx+1) * subprodSize]

                    // parent = left_result * right_subprod + right_result * left_subprod
                    for i in 0..<childSize {
                        let lCoeff = currentLevel[leftIdx * childSize + i]
                        let rCoeff = currentLevel[rightIdx * childSize + i]
                        // left_result[i] * right_subprod[j]
                        for j in 0..<subprodSize {
                            let rSub = treePtr[rightIdx * subprodSize + j]
                            let prod = frMul(lCoeff, rSub)
                            let outIdx = p * parentSize + i + j
                            if outIdx < (p + 1) * parentSize {
                                nextLevel[outIdx] = frAdd(nextLevel[outIdx], prod)
                            }
                        }
                        // right_result[i] * left_subprod[j]
                        for j in 0..<subprodSize {
                            let lSub = treePtr[leftIdx * subprodSize + j]
                            let prod = frMul(rCoeff, lSub)
                            let outIdx = p * parentSize + i + j
                            if outIdx < (p + 1) * parentSize {
                                nextLevel[outIdx] = frAdd(nextLevel[outIdx], prod)
                            }
                        }
                    }
                }
                currentLevel = nextLevel
            } else {
                // GPU NTT-based multiplication for larger polynomials
                // For each parent: result = left_result * right_subprod + right_result * left_subprod
                // We batch all left*rightSub and right*leftSub multiplies separately, then add.

                let mulResultSize = childSize + subprodSize - 1
                let nttLogN = ceilLog2(mulResultSize)
                let nttN = 1 << nttLogN

                // Prepare batch arrays: 2*numPolys multiplications total
                let batchCount = numPolys * 2
                var aArray = [Fr](repeating: Fr.zero, count: batchCount * nttN)
                var bArray = [Fr](repeating: Fr.zero, count: batchCount * nttN)

                for p in 0..<numPolys {
                    let leftIdx = 2 * p
                    let rightIdx = 2 * p + 1

                    // Multiply 0: left_result * right_subprod
                    let m0 = p * 2
                    for i in 0..<childSize {
                        aArray[m0 * nttN + i] = currentLevel[leftIdx * childSize + i]
                    }
                    for j in 0..<subprodSize {
                        bArray[m0 * nttN + j] = treePtr[rightIdx * subprodSize + j]
                    }

                    // Multiply 1: right_result * left_subprod
                    let m1 = p * 2 + 1
                    for i in 0..<childSize {
                        aArray[m1 * nttN + i] = currentLevel[rightIdx * childSize + i]
                    }
                    for j in 0..<subprodSize {
                        bArray[m1 * nttN + j] = treePtr[leftIdx * subprodSize + j]
                    }
                }

                let products = try batchMultiplyGPU(a: aArray, b: bArray, count: batchCount,
                                                     polyStride: nttN, nttLogN: nttLogN)

                // Combine: parent[p] = products[2*p] + products[2*p+1]
                var nextLevel = [Fr](repeating: Fr.zero, count: numPolys * parentSize)
                for p in 0..<numPolys {
                    for i in 0..<parentSize {
                        let v0 = (i < nttN) ? products[(p * 2) * nttN + i] : Fr.zero
                        let v1 = (i < nttN) ? products[(p * 2 + 1) * nttN + i] : Fr.zero
                        nextLevel[p * parentSize + i] = frAdd(v0, v1)
                    }
                }
                currentLevel = nextLevel
            }
        }

        // currentLevel now has 1 polynomial of degree < N
        return currentLevel
    }

    // MARK: - GPU-Resident Polynomial Multiply

    /// Multiply two polynomials on GPU. Inputs are padded arrays of size nttN = 2^logN.
    /// Returns result buffer of size nttN. Single command buffer submit.
    private func multiplyGPU(_ a: [Fr], _ b: [Fr], nttLogN: Int) throws -> [Fr] {
        let nttN = 1 << nttLogN

        // Pad to NTT size
        var aPad = a
        while aPad.count < nttN { aPad.append(Fr.zero) }
        var bPad = b
        while bPad.count < nttN { bPad.append(Fr.zero) }

        let stride = MemoryLayout<Fr>.stride
        let bufSize = nttN * stride
        let aBuf = getCachedBuffer(slot: "mulA", minBytes: bufSize)
        let bBuf = getCachedBuffer(slot: "mulB", minBytes: bufSize)
        aPad.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, bufSize) }
        bPad.withUnsafeBytes { src in memcpy(bBuf.contents(), src.baseAddress!, bufSize) }

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

        // Working buffers (cached, grow-only)
        let bufBytes = nttN * stride
        let hCopyBuf = getCachedBuffer(slot: "invHCopy", minBytes: bufBytes)
        let fNTTBuf = getCachedBuffer(slot: "invFNTT", minBytes: bufBytes)
        let prodBuf = getCachedBuffer(slot: "invProd", minBytes: bufBytes)
        let tmfhBuf = getCachedBuffer(slot: "invTmfh", minBytes: bufBytes)

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

        // Chain linear pairs + all schoolbook levels into a single command buffer.
        // This eliminates one CB+wait per schoolbook level.
        let schoolbookMaxDeg = 256

        // Pre-scan to find how many schoolbook levels we can chain
        var schoolbookLevels: [(numPolys: Int, inSize: Int, outSize: Int)] = []
        var schoolbookOutBufs: [MTLBuffer] = []
        var schoolbookLeftBufs: [MTLBuffer] = []
        var schoolbookRightBufs: [MTLBuffer] = []

        for k in 2...logN {
            let numPolys = N >> k
            let inDeg = 1 << (k - 1)
            let outDeg = 1 << k
            if outDeg > schoolbookMaxDeg { break }
            let inSize = inDeg + 1
            let outSize = outDeg + 1
            schoolbookLevels.append((numPolys, inSize, outSize))
        }

        // Pre-allocate all schoolbook buffers before encoding (Metal requirement)
        for (i, level) in schoolbookLevels.enumerated() {
            let outBuf = device.makeBuffer(length: level.numPolys * level.outSize * MemoryLayout<Fr>.stride,
                                           options: .storageModeShared)!
            schoolbookOutBufs.append(outBuf)
            // Use indexed cache slots so we don't clobber between levels
            let leftBuf = getCachedBuffer(slot: "sbLeft", minBytes: level.numPolys * level.inSize * MemoryLayout<Fr>.stride)
            let rightBuf = getCachedBuffer(slot: "sbRight", minBytes: level.numPolys * level.inSize * MemoryLayout<Fr>.stride)
            schoolbookLeftBufs.append(leftBuf)
            schoolbookRightBufs.append(rightBuf)
        }

        // Build chained command buffer: linearPairs + schoolbook levels
        let chain = MetalPipelineChain(queue: commandQueue)
        let chainCB = try chain.getCommandBuffer()
        let pointsBuf = createBuffer(points)
        try encodeLinearPairs(points: pointsBuf, output: level1Buf, n: N, cmdBuf: chainCB)
        tree.append(level1Buf)

        // We need the linear pairs result to do splitLR (CPU read), so we must wait here.
        // But schoolbook levels can be chained if we prepare splitLR data eagerly.
        // Unfortunately splitLR reads from the previous level's GPU buffer, so each level
        // depends on the previous GPU output. We MUST wait between the chain and CPU splits.
        try chain.execute()

        // Now chain schoolbook levels: for each, splitLR is CPU (reads prev buffer), then GPU dispatch.
        // We can't avoid the CPU splitLR, but we CAN batch multiple schoolbook GPU dispatches
        // if we prepare all CPU data first. However, each level depends on the previous level's output,
        // so we truly need sequential: CPU split → GPU schoolbook → CPU split → GPU schoolbook.
        // The only win is if we have independent levels, which we don't.
        // So we fall back to per-level dispatch but use the encode pattern for consistency.
        for (i, level) in schoolbookLevels.enumerated() {
            let k = i + 2
            splitLR(prev: tree[k - 1], left: schoolbookLeftBufs[i], right: schoolbookRightBufs[i],
                    numPolys: level.numPolys, polySize: level.inSize)
            try dispatchSchoolbookMultiply(left: schoolbookLeftBufs[i], right: schoolbookRightBufs[i],
                                           output: schoolbookOutBufs[i],
                                           dPlus1: UInt32(level.inSize), outSize: UInt32(level.outSize),
                                           count: UInt32(level.numPolys))
            tree.append(schoolbookOutBufs[i])
        }

        // NTT levels (already internally chained)
        let firstNTTLevel = 2 + schoolbookLevels.count
        for k in firstNTTLevel...logN {
            let numPolys = N >> k
            let inDeg = 1 << (k - 1)
            let outDeg = 1 << k
            let inSize = inDeg + 1
            let outSize = outDeg + 1
            let outBuf = try nttMultiplyLevel(prev: tree[k - 1], numPolys: numPolys,
                                               inSize: inSize, outSize: outSize, outDeg: outDeg)
            tree.append(outBuf)
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

        let totalBytes = numPolys * nttN * stride
        let aBuf = getCachedBuffer(slot: "nttLvlA", minBytes: totalBytes)
        let bBuf = getCachedBuffer(slot: "nttLvlB", minBytes: totalBytes)
        memset(aBuf.contents(), 0, totalBytes)
        memset(bBuf.contents(), 0, totalBytes)

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

        // Output buffer can't be cached — it's stored in the tree array and must persist
        let outBytes = numPolys * outSize * stride
        let outBuf = device.makeBuffer(length: outBytes, options: .storageModeShared)!
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

        let fBuf = getCachedBuffer(slot: "remSbF", minBytes: numChildren * fSize * stride)
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
    /// Total submits: 2 (chained inverse+quotient multiply, then q*g multiply).
    /// Previously 3 submits — saves 1 round-trip by chaining inverse into quotient multiply.
    private func computeRemaindersNTT(
        parents: MTLBuffer, parentSize: Int,
        childNodes: MTLBuffer, childSize: Int,
        output: MTLBuffer, childRemSize: Int,
        numParents: Int
    ) throws {
        let stride = MemoryLayout<Fr>.stride
        let numChildren = numParents * 2
        let childDeg = childSize - 1
        let quotientSize = parentSize - childDeg

        let maxProdSize = max(quotientSize * 2, quotientSize + childSize - 1)
        let nttLogN = ceilLog2(maxProdSize)
        let nttN = 1 << nttLogN

        // Step 1+2 CHAINED: Inverse + quotient multiply in a single command buffer.
        // Prepare all reversed divisors
        var allRevG = [Fr](repeating: Fr.zero, count: numChildren * nttN)
        let cPtr = childNodes.contents().bindMemory(to: Fr.self, capacity: numChildren * childSize)
        for c in 0..<numChildren {
            for i in 0..<childSize {
                allRevG[c * nttN + i] = cPtr[c * childSize + (childSize - 1 - i)]
            }
        }

        // Prepare all reversed dividends (truncated to quotientSize)
        var allRevFTrunc = [Fr](repeating: Fr.zero, count: numChildren * nttN)
        let pPtr = parents.contents().bindMemory(to: Fr.self, capacity: numParents * parentSize)
        for p in 0..<numParents {
            for side in 0..<2 {
                let c = p * 2 + side
                for i in 0..<min(quotientSize, parentSize) {
                    allRevFTrunc[c * nttN + i] = pPtr[p * parentSize + (parentSize - 1 - i)]
                }
            }
        }

        // Pre-fill batchMultiply input buffers BEFORE creating CB
        let totalBytes = numChildren * nttN * stride
        let mulABuf = getCachedBuffer(slot: "bMulA", minBytes: totalBytes)
        let mulBBuf = getCachedBuffer(slot: "bMulB", minBytes: totalBytes)
        allRevFTrunc.withUnsafeBytes { src in memcpy(mulABuf.contents(), src.baseAddress!, totalBytes) }
        // mulBBuf will be filled from inverse result via blit

        // Single CB: polyInverseBatch → blit inv→mulBBuf → batchMultiply
        guard let cmdBuf = nttEngine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let invBuf = try encodePolyInverseBatch(fs: allRevG, count: numChildren,
                                                  polyStride: nttN, modDeg: quotientSize,
                                                  nttLogN: nttLogN, cmdBuf: cmdBuf)

        // Blit inverse result into mulBBuf (GPU-to-GPU copy, no CPU round-trip)
        let blit = cmdBuf.makeBlitCommandEncoder()!
        blit.copy(from: invBuf, sourceOffset: 0, to: mulBBuf, destinationOffset: 0, size: totalBytes)
        blit.endEncoding()

        // Encode batch multiply: mulABuf (revFTrunc) * mulBBuf (inv) → result in mulABuf
        encodeBatchMultiplyGPU(aBuf: mulABuf, bBuf: mulBBuf, count: numChildren,
                                nttLogN: nttLogN, cmdBuf: cmdBuf)

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }

        // CPU: read quotient result and reverse
        let allQRev = readBuffer(mulABuf, count: numChildren * nttN)
        var allQ = [Fr](repeating: Fr.zero, count: numChildren * nttN)
        for c in 0..<numChildren {
            for i in 0..<quotientSize {
                allQ[c * nttN + i] = allQRev[c * nttN + (quotientSize - 1 - i)]
            }
        }

        // Step 3: Multiply q * g for all children (separate CB — needs CPU-prepared data)
        var allG = [Fr](repeating: Fr.zero, count: numChildren * nttN)
        for c in 0..<numChildren {
            for i in 0..<childSize {
                allG[c * nttN + i] = cPtr[c * childSize + i]
            }
        }

        let allQG = try batchMultiplyGPU(a: allQ, b: allG, count: numChildren,
                                          polyStride: nttN, nttLogN: nttLogN)

        // Step 4: Compute remainders r[c] = f[parent(c)] - qg[c]
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

    /// Encode all Newton iteration steps for batch polynomial inverse into an existing command buffer.
    /// Returns the hBuf containing results (must be read after CB completes).
    private func encodePolyInverseBatch(fs: [Fr], count: Int, polyStride: Int,
                                          modDeg: Int, nttLogN: Int,
                                          cmdBuf: MTLCommandBuffer) throws -> MTLBuffer {
        let nttN = 1 << nttLogN
        let stride = MemoryLayout<Fr>.stride
        let tg = 256

        precondition(polyStride == nttN)

        let fBuf = createBuffer(fs)

        var hInit = [Fr](repeating: Fr.zero, count: count * nttN)
        for c in 0..<count {
            let f0 = fs[c * nttN]
            precondition(!f0.isZero, "f[0] must be nonzero")
            hInit[c * nttN] = frInverse(f0)
        }
        let hBuf = createBuffer(hInit)

        let batchBufBytes = count * nttN * stride
        let fNTTBuf = getCachedBuffer(slot: "bInvFNTT", minBytes: batchBufBytes)
        let hCopyBuf = getCachedBuffer(slot: "bInvHCopy", minBytes: batchBufBytes)
        let prodBuf = getCachedBuffer(slot: "bInvProd", minBytes: batchBufBytes)
        let tmfhBuf = getCachedBuffer(slot: "bInvTmfh", minBytes: batchBufBytes)

        var precision = 1
        while precision < modDeg {
            let nextPrecision = min(precision * 2, modDeg)

            let blit1 = cmdBuf.makeBlitCommandEncoder()!
            blit1.copy(from: fBuf, sourceOffset: 0, to: fNTTBuf,
                      destinationOffset: 0, size: count * nttN * stride)
            blit1.copy(from: hBuf, sourceOffset: 0, to: hCopyBuf,
                      destinationOffset: 0, size: count * nttN * stride)
            blit1.endEncoding()

            for c in 0..<count {
                let off = c * nttN * stride
                nttEngine.encodeNTT(data: fNTTBuf, offset: off, logN: nttLogN, cmdBuf: cmdBuf)
                nttEngine.encodeNTT(data: hBuf, offset: off, logN: nttLogN, cmdBuf: cmdBuf)
            }

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

            for c in 0..<count {
                nttEngine.encodeINTT(data: prodBuf, offset: c * nttN * stride, logN: nttLogN, cmdBuf: cmdBuf)
            }

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

            for c in 0..<count {
                let off = c * nttN * stride
                nttEngine.encodeNTT(data: hCopyBuf, offset: off, logN: nttLogN, cmdBuf: cmdBuf)
                nttEngine.encodeNTT(data: tmfhBuf, offset: off, logN: nttLogN, cmdBuf: cmdBuf)
            }

            let encH2 = cmdBuf.makeComputeCommandEncoder()!
            encH2.setComputePipelineState(hadamardFunction)
            encH2.setBuffer(hCopyBuf, offset: 0, index: 0)
            encH2.setBuffer(tmfhBuf, offset: 0, index: 1)
            encH2.setBuffer(hBuf, offset: 0, index: 2)
            encH2.setBytes(&totalN, length: 4, index: 3)
            encH2.dispatchThreads(MTLSize(width: Int(totalN), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            encH2.endEncoding()

            for c in 0..<count {
                nttEngine.encodeINTT(data: hBuf, offset: c * nttN * stride, logN: nttLogN, cmdBuf: cmdBuf)
            }

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

        return hBuf
    }

    /// Compute inverses of `count` polynomials simultaneously.
    /// All polynomials stored in `fs` with stride `polyStride` elements.
    /// Returns buffer with `count * polyStride` elements (inverses padded with zeros).
    private func polyInverseBatch(fs: [Fr], count: Int, polyStride: Int,
                                   modDeg: Int, nttLogN: Int) throws -> MTLBuffer {
        guard let cmdBuf = nttEngine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let hBuf = try encodePolyInverseBatch(fs: fs, count: count, polyStride: polyStride,
                                                modDeg: modDeg, nttLogN: nttLogN, cmdBuf: cmdBuf)

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }

        return hBuf
    }

    /// Encode batch multiply into an existing command buffer (no submit).
    /// aBuf and bBuf must be pre-filled. Result lands in aBuf.
    private func encodeBatchMultiplyGPU(aBuf: MTLBuffer, bBuf: MTLBuffer, count: Int,
                                         nttLogN: Int, cmdBuf: MTLCommandBuffer) {
        let nttN = 1 << nttLogN
        let stride = MemoryLayout<Fr>.stride
        let tg = 256

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
    }

    /// Batch multiply: a[c] * b[c] for c in 0..<count, all at same NTT size.
    /// Returns flat array of count * polyStride results.
    private func batchMultiplyGPU(a: [Fr], b: [Fr], count: Int,
                                   polyStride: Int, nttLogN: Int) throws -> [Fr] {
        let nttN = 1 << nttLogN
        let stride = MemoryLayout<Fr>.stride

        let totalBytes = a.count * stride
        let aBuf = getCachedBuffer(slot: "bMulA", minBytes: totalBytes)
        let bBuf = getCachedBuffer(slot: "bMulB", minBytes: totalBytes)
        a.withUnsafeBytes { src in memcpy(aBuf.contents(), src.baseAddress!, totalBytes) }
        b.withUnsafeBytes { src in memcpy(bBuf.contents(), src.baseAddress!, totalBytes) }

        guard let cmdBuf = nttEngine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        encodeBatchMultiplyGPU(aBuf: aBuf, bBuf: bBuf, count: count, nttLogN: nttLogN, cmdBuf: cmdBuf)

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

    /// Encode linear pairs dispatch into an existing command buffer (no submit).
    private func encodeLinearPairs(points: MTLBuffer, output: MTLBuffer, n: Int,
                                    cmdBuf: MTLCommandBuffer) throws {
        guard let fn = treeBuildLinearPairsFunction else { throw MSMError.missingKernel }
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
    }

    private func dispatchLinearPairs(points: MTLBuffer, output: MTLBuffer, n: Int) throws {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        try encodeLinearPairs(points: points, output: output, n: n, cmdBuf: cmdBuf)
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }
    }

    /// Encode schoolbook multiply dispatch into an existing command buffer (no submit).
    private func encodeSchoolbookMultiply(left: MTLBuffer, right: MTLBuffer, output: MTLBuffer,
                                            dPlus1: UInt32, outSize: UInt32, count: UInt32,
                                            cmdBuf: MTLCommandBuffer) throws {
        guard let fn = treeBuildSchoolbookFunction else { throw MSMError.missingKernel }
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
    }

    private func dispatchSchoolbookMultiply(left: MTLBuffer, right: MTLBuffer, output: MTLBuffer,
                                             dPlus1: UInt32, outSize: UInt32, count: UInt32) throws {
        guard let cmdBuf = commandQueue.makeCommandBuffer() else { throw MSMError.noCommandBuffer }
        try encodeSchoolbookMultiply(left: left, right: right, output: output,
                                      dPlus1: dPlus1, outSize: outSize, count: count, cmdBuf: cmdBuf)
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw MSMError.gpuError(error.localizedDescription) }
    }
}
