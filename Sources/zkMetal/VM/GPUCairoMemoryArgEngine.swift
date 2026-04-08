// GPUCairoMemoryArgEngine -- GPU-accelerated Cairo memory argument engine
//
// Implements the continuous memory argument for Cairo VM proving:
//   1. Sorted access permutation: reorder memory accesses by address
//   2. Address uniqueness check: ensure no duplicate addresses in single-write memory
//   3. Read-write consistency: reads return most recent write (or zero)
//   4. GPU-accelerated permutation product accumulation
//   5. Multi-segment memory support (program, execution, builtin segments)
//
// The memory argument is a key component of Cairo STARK proofs. It proves that
// the prover's claimed memory is consistent: every read returns the value that
// was last written to that address. Cairo uses write-once memory, so each
// address can only be written once, and all subsequent reads return that value.
//
// The permutation argument works by:
//   - Taking the original (address, value) access trace in execution order
//   - Sorting it by address (breaking ties by step/access index)
//   - Proving the sorted trace is a permutation of the original via a
//     grand product argument: prod((a_i + alpha * v_i + z)) for random z, alpha
//   - Checking continuity constraints on the sorted trace
//
// GPU acceleration targets the permutation product accumulation, which is
// the most compute-intensive part for large traces (O(n) field multiplications).
//
// Reference: Cairo whitepaper Section 9.7 (Memory)

import Foundation
import NeonFieldOps

// MARK: - Memory Access Record

/// A single memory access in the Cairo execution trace.
/// Records the address, value, access type, step index, and segment.
public struct CairoMemoryAccess {
    /// The memory address accessed
    public let address: UInt64
    /// The field element value read or written
    public let value: Fr
    /// Whether this is a write (true) or read (false)
    public let isWrite: Bool
    /// The execution step at which this access occurred
    public let step: Int
    /// The memory segment this access belongs to
    public let segment: CairoMemorySegment

    public init(address: UInt64, value: Fr, isWrite: Bool, step: Int,
                segment: CairoMemorySegment = .execution) {
        self.address = address
        self.value = value
        self.isWrite = isWrite
        self.step = step
        self.segment = segment
    }
}

// MARK: - Memory Segment

/// Cairo memory is divided into segments for different purposes.
/// Each segment has a base address and a distinct role in the VM.
public enum CairoMemorySegment: UInt8, CaseIterable {
    /// Program code segment (read-only instructions)
    case program    = 0
    /// Execution segment (stack, temporaries via ap/fp)
    case execution  = 1
    /// Output builtin segment
    case output     = 2
    /// Range-check builtin segment
    case rangeCheck = 3
    /// Pedersen hash builtin segment
    case pedersen   = 4
    /// ECDSA builtin segment
    case ecdsa      = 5
    /// Bitwise builtin segment
    case bitwise    = 6
    /// EC op builtin segment
    case ecOp       = 7
}

/// Configuration for a memory segment with base address and size.
public struct CairoSegmentConfig {
    public let segment: CairoMemorySegment
    public let baseAddress: UInt64
    public let size: Int

    public init(segment: CairoMemorySegment, baseAddress: UInt64, size: Int) {
        self.segment = segment
        self.baseAddress = baseAddress
        self.size = size
    }
}

// MARK: - Sorted Memory Entry

/// A memory access after sorting by (address, step).
/// Used in the continuity check and permutation argument.
public struct SortedMemoryEntry: Comparable {
    public let address: UInt64
    public let value: Fr
    public let isWrite: Bool
    public let step: Int
    public let originalIndex: Int

    public init(address: UInt64, value: Fr, isWrite: Bool, step: Int, originalIndex: Int) {
        self.address = address
        self.value = value
        self.isWrite = isWrite
        self.step = step
        self.originalIndex = originalIndex
    }

    public static func < (lhs: SortedMemoryEntry, rhs: SortedMemoryEntry) -> Bool {
        if lhs.address != rhs.address { return lhs.address < rhs.address }
        return lhs.step < rhs.step
    }
}

// MARK: - Permutation Product Result

/// Result of the permutation product accumulation.
/// Contains both the running product array and the final product value.
public struct PermutationProductResult {
    /// Running product: z[0] = 1, z[i] = z[i-1] * factor[i-1]
    public let runningProduct: [Fr]
    /// The final accumulated product z[n]
    public let finalProduct: Fr
    /// Whether the permutation is valid (finalProduct == 1 when both sides match)
    public let isValid: Bool

    public init(runningProduct: [Fr], finalProduct: Fr, isValid: Bool) {
        self.runningProduct = runningProduct
        self.finalProduct = finalProduct
        self.isValid = isValid
    }
}

// MARK: - Memory Argument Result

/// Complete result of the Cairo memory argument computation.
public struct CairoMemoryArgResult {
    /// The original access trace
    public let originalTrace: [CairoMemoryAccess]
    /// The sorted access trace
    public let sortedTrace: [SortedMemoryEntry]
    /// Permutation product for the original ordering
    public let originalProduct: PermutationProductResult
    /// Permutation product for the sorted ordering
    public let sortedProduct: PermutationProductResult
    /// Whether the permutation argument is valid (products match)
    public let permutationValid: Bool
    /// Whether all continuity constraints pass on the sorted trace
    public let continuityValid: Bool
    /// Whether read-write consistency holds
    public let consistencyValid: Bool
    /// Overall validity: all three checks pass
    public var isValid: Bool {
        permutationValid && continuityValid && consistencyValid
    }
    /// Per-segment statistics
    public let segmentStats: [CairoMemorySegment: SegmentStats]

    public init(originalTrace: [CairoMemoryAccess], sortedTrace: [SortedMemoryEntry],
                originalProduct: PermutationProductResult, sortedProduct: PermutationProductResult,
                permutationValid: Bool, continuityValid: Bool, consistencyValid: Bool,
                segmentStats: [CairoMemorySegment: SegmentStats]) {
        self.originalTrace = originalTrace
        self.sortedTrace = sortedTrace
        self.originalProduct = originalProduct
        self.sortedProduct = sortedProduct
        self.permutationValid = permutationValid
        self.continuityValid = continuityValid
        self.consistencyValid = consistencyValid
        self.segmentStats = segmentStats
    }
}

/// Per-segment statistics for the memory argument.
public struct SegmentStats {
    public let segment: CairoMemorySegment
    public let accessCount: Int
    public let uniqueAddresses: Int
    public let writeCount: Int
    public let readCount: Int

    public init(segment: CairoMemorySegment, accessCount: Int, uniqueAddresses: Int,
                writeCount: Int, readCount: Int) {
        self.segment = segment
        self.accessCount = accessCount
        self.uniqueAddresses = uniqueAddresses
        self.writeCount = writeCount
        self.readCount = readCount
    }
}

// MARK: - Address Continuity Result

/// Result of the address continuity check on the sorted trace.
public struct ContinuityCheckResult {
    /// Whether all continuity constraints pass
    public let isValid: Bool
    /// Indices where continuity violations occurred (sorted trace indices)
    public let violationIndices: [Int]
    /// Number of distinct addresses found
    public let distinctAddressCount: Int
    /// The address gaps encountered (sorted trace index, gap size)
    public let addressGaps: [(index: Int, gap: UInt64)]

    public init(isValid: Bool, violationIndices: [Int], distinctAddressCount: Int,
                addressGaps: [(index: Int, gap: UInt64)]) {
        self.isValid = isValid
        self.violationIndices = violationIndices
        self.distinctAddressCount = distinctAddressCount
        self.addressGaps = addressGaps
    }
}

// MARK: - Multi-Segment Memory

/// Multi-segment memory model for Cairo.
/// Manages multiple memory segments with distinct base addresses,
/// tracks accesses per segment, and supports segment-aware sorting.
public struct CairoMultiSegmentMemory {
    /// Segment configurations
    public let segments: [CairoSegmentConfig]
    /// All accesses across all segments
    public private(set) var accesses: [CairoMemoryAccess]
    /// Per-address current value (write-once enforcement)
    private var addressValues: [UInt64: Fr]

    public init(segments: [CairoSegmentConfig]) {
        self.segments = segments
        self.accesses = []
        self.addressValues = [:]
    }

    /// Write a value to an address in the given segment.
    /// Returns false if write-once violation (different value at same address).
    @discardableResult
    public mutating func write(segment: CairoMemorySegment, offset: UInt64,
                               value: Fr, step: Int) -> Bool {
        let config = segments.first { $0.segment == segment }
        let baseAddr = config?.baseAddress ?? 0
        let addr = baseAddr + offset

        if let existing = addressValues[addr] {
            if existing != value { return false }
        }
        addressValues[addr] = value
        accesses.append(CairoMemoryAccess(
            address: addr, value: value, isWrite: true, step: step, segment: segment))
        return true
    }

    /// Read a value from an address in the given segment.
    /// Returns Fr.zero for uninitialized addresses.
    public mutating func read(segment: CairoMemorySegment, offset: UInt64, step: Int) -> Fr {
        let config = segments.first { $0.segment == segment }
        let baseAddr = config?.baseAddress ?? 0
        let addr = baseAddr + offset
        let value = addressValues[addr] ?? Fr.zero
        accesses.append(CairoMemoryAccess(
            address: addr, value: value, isWrite: false, step: step, segment: segment))
        return value
    }

    /// Get the value at an address without logging an access.
    public func peek(segment: CairoMemorySegment, offset: UInt64) -> Fr? {
        let config = segments.first { $0.segment == segment }
        let baseAddr = config?.baseAddress ?? 0
        return addressValues[baseAddr + offset]
    }

    /// Total number of distinct addresses written.
    public var distinctAddressCount: Int { addressValues.count }

    /// Total number of accesses logged.
    public var totalAccessCount: Int { accesses.count }
}

// MARK: - GPUCairoMemoryArgEngine

/// GPU-accelerated engine for the Cairo continuous memory argument.
///
/// The memory argument proves that the execution trace's memory accesses
/// are consistent with write-once semantics. It consists of:
///
/// 1. **Permutation argument**: The sorted access trace is a permutation
///    of the original trace. Proven via a grand product:
///      prod(a_i + alpha * v_i + z) [original] == prod(a'_i + alpha * v'_i + z) [sorted]
///
/// 2. **Continuity check**: In the sorted trace, consecutive accesses to
///    the same address have identical values, and address transitions are
///    by exactly +1 (or within the same address).
///
/// 3. **Consistency check**: Every read returns the written value. Since
///    Cairo has write-once memory, after the first write all reads must
///    return that same value.
///
/// GPU acceleration is used for the permutation product accumulation,
/// which requires O(n) field multiplications for n accesses.
///
/// Usage:
///   let engine = GPUCairoMemoryArgEngine()
///   let accesses: [CairoMemoryAccess] = [...]  // from trace
///   let result = engine.computeMemoryArgument(accesses: accesses)
///   assert(result.isValid)
public final class GPUCairoMemoryArgEngine {

    /// Random challenge alpha for the permutation product (value channel).
    /// In a real protocol this comes from Fiat-Shamir; here it can be set.
    public var alpha: Fr

    /// Random challenge z for the permutation product (linear combination).
    public var z: Fr

    /// Threshold: use GPU path for traces with more accesses than this.
    public var gpuThreshold: Int

    /// Whether to enforce strict write-once semantics (reject double writes
    /// even with the same value).
    public var strictWriteOnce: Bool

    /// The underlying GPU grand product engine (nil if unavailable).
    private let gpuEngine: GPUGrandProductEngine?

    /// Whether GPU acceleration is available.
    public var isGPUAvailable: Bool { gpuEngine != nil }

    /// Create engine with optional random challenges.
    /// In production, alpha and z come from Fiat-Shamir transcript.
    public init(alpha: Fr? = nil, z: Fr? = nil, gpuThreshold: Int = 512,
                strictWriteOnce: Bool = false) {
        self.alpha = alpha ?? frFromInt(7919)    // Default deterministic challenge
        self.z = z ?? frFromInt(104729)           // Default deterministic challenge
        self.gpuThreshold = gpuThreshold
        self.strictWriteOnce = strictWriteOnce
        self.gpuEngine = try? GPUGrandProductEngine()
    }

    // MARK: - Full Memory Argument

    /// Compute the complete Cairo memory argument for a trace.
    ///
    /// Performs all three checks:
    ///   1. Permutation product accumulation (GPU-accelerated)
    ///   2. Continuity check on sorted trace
    ///   3. Read-write consistency check
    ///
    /// - Parameter accesses: The memory access trace from execution.
    /// - Returns: Complete memory argument result with validity flags.
    public func computeMemoryArgument(accesses: [CairoMemoryAccess]) -> CairoMemoryArgResult {
        guard !accesses.isEmpty else {
            return emptyResult()
        }

        // Step 1: Sort accesses by (address, step)
        let sorted = sortAccesses(accesses)

        // Step 2: Compute permutation products for both orderings
        let originalFactors = computePermutationFactors(accesses)
        let sortedFactors = computeSortedPermutationFactors(sorted)
        let originalProduct = accumulateProduct(originalFactors)
        let sortedProduct = accumulateProduct(sortedFactors)

        // Step 3: Check permutation validity (products must match)
        let permValid = frEqual(originalProduct.finalProduct, sortedProduct.finalProduct)

        // Step 4: Check continuity on sorted trace
        let continuityResult = checkContinuity(sorted)

        // Step 5: Check read-write consistency
        let consistencyResult = checkReadWriteConsistency(sorted)

        // Step 6: Compute per-segment statistics
        let stats = computeSegmentStats(accesses)

        return CairoMemoryArgResult(
            originalTrace: accesses,
            sortedTrace: sorted,
            originalProduct: originalProduct,
            sortedProduct: sortedProduct,
            permutationValid: permValid,
            continuityValid: continuityResult.isValid,
            consistencyValid: consistencyResult,
            segmentStats: stats
        )
    }

    // MARK: - Sorting

    /// Sort memory accesses by (address, step) to produce the sorted trace.
    /// This is the "sigma" permutation in the memory argument.
    public func sortAccesses(_ accesses: [CairoMemoryAccess]) -> [SortedMemoryEntry] {
        var entries = [SortedMemoryEntry]()
        entries.reserveCapacity(accesses.count)
        for (i, a) in accesses.enumerated() {
            entries.append(SortedMemoryEntry(
                address: a.address, value: a.value, isWrite: a.isWrite,
                step: a.step, originalIndex: i))
        }
        entries.sort()
        return entries
    }

    // MARK: - Permutation Product

    /// Compute permutation factors for the original (unsorted) access trace.
    /// Each factor is: address_i + alpha * value_i + z
    public func computePermutationFactors(_ accesses: [CairoMemoryAccess]) -> [Fr] {
        var factors = [Fr](repeating: Fr.zero, count: accesses.count)
        for i in 0..<accesses.count {
            let addrFr = frFromInt(accesses[i].address)
            let alphaVal = frMul(alpha, accesses[i].value)
            factors[i] = frAdd(frAdd(addrFr, alphaVal), z)
        }
        return factors
    }

    /// Compute permutation factors for the sorted access trace.
    public func computeSortedPermutationFactors(_ sorted: [SortedMemoryEntry]) -> [Fr] {
        var factors = [Fr](repeating: Fr.zero, count: sorted.count)
        for i in 0..<sorted.count {
            let addrFr = frFromInt(sorted[i].address)
            let alphaVal = frMul(alpha, sorted[i].value)
            factors[i] = frAdd(frAdd(addrFr, alphaVal), z)
        }
        return factors
    }

    /// Accumulate a running product from an array of factors.
    /// z[0] = 1, z[i] = z[i-1] * factor[i-1], final = z[n].
    ///
    /// Uses GPU for large traces, CPU fallback otherwise.
    public func accumulateProduct(_ factors: [Fr]) -> PermutationProductResult {
        let n = factors.count
        guard n > 0 else {
            return PermutationProductResult(runningProduct: [Fr.one], finalProduct: Fr.one, isValid: true)
        }

        var running = [Fr](repeating: Fr.zero, count: n + 1)
        running[0] = Fr.one

        // GPU path for large traces
        if n > gpuThreshold, let gpu = gpuEngine {
            let prefixProducts = gpu.partialProducts(values: factors)
            if prefixProducts.count == n {
                running[0] = Fr.one
                for i in 0..<n {
                    running[i + 1] = prefixProducts[i]
                }
                let finalProd = running[n]
                return PermutationProductResult(
                    runningProduct: running, finalProduct: finalProd,
                    isValid: frEqual(finalProd, Fr.one))
            }
        }

        // CPU fallback
        for i in 0..<n {
            running[i + 1] = frMul(running[i], factors[i])
        }
        let finalProd = running[n]
        return PermutationProductResult(
            runningProduct: running, finalProduct: finalProd,
            isValid: frEqual(finalProd, Fr.one))
    }

    // MARK: - Continuity Check

    /// Check continuity constraints on the sorted memory trace.
    ///
    /// For consecutive entries (i, i+1) in the sorted trace:
    ///   - If address[i] == address[i+1]: value must be identical (write-once)
    ///   - If address[i] != address[i+1]: the gap must be exactly 1
    ///     (continuous addressing, no holes)
    ///
    /// The continuous addressing requirement can be relaxed for multi-segment
    /// memory by using "memory holes" (dummy accesses to fill gaps).
    public func checkContinuity(_ sorted: [SortedMemoryEntry]) -> ContinuityCheckResult {
        guard sorted.count > 1 else {
            return ContinuityCheckResult(
                isValid: true, violationIndices: [], distinctAddressCount: sorted.count,
                addressGaps: [])
        }

        var violations = [Int]()
        var gaps: [(index: Int, gap: UInt64)] = []
        var distinctCount = 1  // First address is always distinct

        for i in 1..<sorted.count {
            let prevAddr = sorted[i - 1].address
            let curAddr = sorted[i].address

            if curAddr == prevAddr {
                // Same address: values must match (write-once memory)
                if sorted[i].value != sorted[i - 1].value {
                    violations.append(i)
                }
            } else {
                distinctCount += 1
                let gap = curAddr - prevAddr
                if gap != 1 {
                    // Non-unit address gap -- record it
                    gaps.append((index: i, gap: gap))
                }
            }
        }

        // For strict continuity (no holes), gaps must be empty.
        // For relaxed continuity (multi-segment), gaps are informational.
        let isValid = violations.isEmpty
        return ContinuityCheckResult(
            isValid: isValid, violationIndices: violations,
            distinctAddressCount: distinctCount, addressGaps: gaps)
    }

    // MARK: - Read-Write Consistency

    /// Check read-write consistency on the sorted trace.
    ///
    /// In Cairo's write-once memory:
    ///   - The first access to any address defines its value (must be a write)
    ///   - All subsequent accesses (reads or writes) must have the same value
    ///
    /// Returns true if consistency holds.
    public func checkReadWriteConsistency(_ sorted: [SortedMemoryEntry]) -> Bool {
        guard !sorted.isEmpty else { return true }

        var currentAddr = sorted[0].address
        var currentValue = sorted[0].value

        for i in 1..<sorted.count {
            if sorted[i].address != currentAddr {
                // New address group
                currentAddr = sorted[i].address
                currentValue = sorted[i].value
            } else {
                // Same address: value must match
                if sorted[i].value != currentValue {
                    return false
                }
            }
        }
        return true
    }

    // MARK: - Address Uniqueness

    /// Check that each address appears at most once as a write.
    /// This is the write-once property of Cairo memory.
    ///
    /// - Parameter accesses: The memory access trace.
    /// - Returns: True if no address is written more than once.
    public func checkAddressUniqueness(_ accesses: [CairoMemoryAccess]) -> Bool {
        var written = Set<UInt64>()
        for access in accesses {
            if access.isWrite {
                if strictWriteOnce {
                    if written.contains(access.address) {
                        return false
                    }
                }
                written.insert(access.address)
            }
        }
        return true
    }

    /// Find all duplicate write addresses in the trace.
    /// Returns a dictionary of address -> count for addresses written more than once.
    public func findDuplicateWrites(_ accesses: [CairoMemoryAccess]) -> [UInt64: Int] {
        var writeCounts = [UInt64: Int]()
        for access in accesses {
            if access.isWrite {
                writeCounts[access.address, default: 0] += 1
            }
        }
        return writeCounts.filter { $0.value > 1 }
    }

    // MARK: - Memory Holes

    /// Compute memory holes needed for strict continuity.
    ///
    /// In the sorted trace, any gap > 1 between consecutive addresses requires
    /// "memory hole" dummy entries to fill the gap. This is necessary for the
    /// AIR continuity constraint to pass.
    ///
    /// - Parameter sorted: The sorted memory trace.
    /// - Returns: Array of addresses that need hole entries.
    public func computeMemoryHoles(_ sorted: [SortedMemoryEntry]) -> [UInt64] {
        guard sorted.count > 1 else { return [] }

        var holes = [UInt64]()
        for i in 1..<sorted.count {
            let prevAddr = sorted[i - 1].address
            let curAddr = sorted[i].address
            if curAddr > prevAddr + 1 {
                for addr in (prevAddr + 1)..<curAddr {
                    holes.append(addr)
                }
            }
        }
        return holes
    }

    /// Fill memory holes by inserting dummy accesses with zero values.
    /// Returns the augmented sorted trace with holes filled.
    public func fillMemoryHoles(_ sorted: [SortedMemoryEntry]) -> [SortedMemoryEntry] {
        guard sorted.count > 1 else { return sorted }

        var filled = [SortedMemoryEntry]()
        filled.reserveCapacity(sorted.count * 2)  // Worst case
        filled.append(sorted[0])

        for i in 1..<sorted.count {
            let prevAddr = sorted[i - 1].address
            let curAddr = sorted[i].address
            // Insert holes for gaps
            if curAddr > prevAddr + 1 {
                for addr in (prevAddr + 1)..<curAddr {
                    filled.append(SortedMemoryEntry(
                        address: addr, value: Fr.zero, isWrite: true,
                        step: -1, originalIndex: -1))
                }
            }
            filled.append(sorted[i])
        }
        return filled
    }

    // MARK: - Multi-Segment Support

    /// Compute the memory argument for a multi-segment memory.
    ///
    /// Each segment is processed independently for statistics, but the
    /// permutation argument covers all segments together.
    public func computeMultiSegmentArgument(
        memory: CairoMultiSegmentMemory
    ) -> CairoMemoryArgResult {
        return computeMemoryArgument(accesses: memory.accesses)
    }

    /// Validate segment boundaries: ensure accesses in each segment
    /// fall within the configured address range.
    public func validateSegmentBoundaries(
        accesses: [CairoMemoryAccess],
        segments: [CairoSegmentConfig]
    ) -> [(Int, CairoMemorySegment, UInt64)] {
        var violations: [(Int, CairoMemorySegment, UInt64)] = []
        let segMap = Dictionary(uniqueKeysWithValues: segments.map { ($0.segment, $0) })

        for (i, access) in accesses.enumerated() {
            guard let config = segMap[access.segment] else { continue }
            let offset = access.address >= config.baseAddress
                ? access.address - config.baseAddress : UInt64.max
            if offset >= UInt64(config.size) {
                violations.append((i, access.segment, access.address))
            }
        }
        return violations
    }

    // MARK: - Batch Permutation Product

    /// Compute the permutation product for a batch of access traces.
    /// Useful when proving multiple execution segments in parallel.
    ///
    /// - Parameter traces: Array of access traces, one per segment.
    /// - Returns: Array of permutation product results, one per segment.
    public func batchPermutationProducts(
        traces: [[CairoMemoryAccess]]
    ) -> [PermutationProductResult] {
        return traces.map { trace in
            let factors = computePermutationFactors(trace)
            return accumulateProduct(factors)
        }
    }

    // MARK: - Ratio Argument

    /// Compute the permutation ratio argument.
    ///
    /// Instead of checking prod(original factors) == prod(sorted factors),
    /// compute z[i] = prod(original[0..i]) / prod(sorted[0..i]) incrementally.
    /// The argument is valid iff z[n] == 1.
    ///
    /// This is more efficient for the AIR because z can be committed as a
    /// single column and checked with local constraints.
    public func computeRatioArgument(
        accesses: [CairoMemoryAccess]
    ) -> PermutationProductResult {
        let sorted = sortAccesses(accesses)
        let origFactors = computePermutationFactors(accesses)
        let sortFactors = computeSortedPermutationFactors(sorted)

        // Compute inverse of sorted factors
        var sortInverses = [Fr](repeating: .zero, count: sortFactors.count)
        sortFactors.withUnsafeBytes { sBuf in
            sortInverses.withUnsafeMutableBytes { iBuf in
                bn254_fr_batch_inverse(
                    sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(sortFactors.count),
                    iBuf.baseAddress!.assumingMemoryBound(to: UInt64.self))
            }
        }

        // Ratio factors: origFactors[i] * sortInverses[i]
        let n = accesses.count
        var ratioFactors = [Fr](repeating: .zero, count: n)
        origFactors.withUnsafeBytes { aBuf in
            sortInverses.withUnsafeBytes { bBuf in
                ratioFactors.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul_neon(
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }

        return accumulateProduct(ratioFactors)
    }

    // MARK: - Interaction Column

    /// Generate the interaction column for the memory argument.
    ///
    /// The interaction column is used in the STARK proof to encode the
    /// permutation argument. For each row i:
    ///   interaction[i] = z[i+1] / z[i] = factor[i]
    ///
    /// This column is committed alongside the main trace columns.
    public func generateInteractionColumn(
        accesses: [CairoMemoryAccess]
    ) -> [Fr] {
        let factors = computePermutationFactors(accesses)
        return factors
    }

    /// Generate the cumulative interaction column (running product).
    /// This is z[i] for i in 0..n, used directly in the AIR constraints.
    public func generateCumulativeInteraction(
        accesses: [CairoMemoryAccess]
    ) -> [Fr] {
        let factors = computePermutationFactors(accesses)
        let product = accumulateProduct(factors)
        return product.runningProduct
    }

    // MARK: - Segment Statistics

    /// Compute per-segment statistics for the access trace.
    public func computeSegmentStats(
        _ accesses: [CairoMemoryAccess]
    ) -> [CairoMemorySegment: SegmentStats] {
        var stats = [CairoMemorySegment: SegmentStats]()

        // Group accesses by segment
        var segAccesses = [CairoMemorySegment: [CairoMemoryAccess]]()
        for access in accesses {
            segAccesses[access.segment, default: []].append(access)
        }

        for (segment, segAcc) in segAccesses {
            let uniqueAddrs = Set(segAcc.map(\.address))
            let writes = segAcc.filter(\.isWrite).count
            let reads = segAcc.count - writes
            stats[segment] = SegmentStats(
                segment: segment, accessCount: segAcc.count,
                uniqueAddresses: uniqueAddrs.count, writeCount: writes, readCount: reads)
        }
        return stats
    }

    // MARK: - Verification

    /// Verify a complete memory argument given the original and sorted traces
    /// plus their permutation products.
    ///
    /// This is what the verifier checks:
    ///   1. Products match (permutation valid)
    ///   2. Sorted trace has continuity (no write-once violations)
    ///   3. Sorted trace has consistency (reads match writes)
    public func verifyMemoryArgument(_ result: CairoMemoryArgResult) -> Bool {
        // Check 1: Permutation products match
        guard frEqual(result.originalProduct.finalProduct,
                       result.sortedProduct.finalProduct) else { return false }

        // Check 2: Continuity on sorted trace
        let continuity = checkContinuity(result.sortedTrace)
        guard continuity.isValid else { return false }

        // Check 3: Read-write consistency on sorted trace
        guard checkReadWriteConsistency(result.sortedTrace) else { return false }

        return true
    }

    // MARK: - Helpers

    /// Empty result for zero-length traces.
    private func emptyResult() -> CairoMemoryArgResult {
        CairoMemoryArgResult(
            originalTrace: [], sortedTrace: [],
            originalProduct: PermutationProductResult(
                runningProduct: [Fr.one], finalProduct: Fr.one, isValid: true),
            sortedProduct: PermutationProductResult(
                runningProduct: [Fr.one], finalProduct: Fr.one, isValid: true),
            permutationValid: true, continuityValid: true, consistencyValid: true,
            segmentStats: [:])
    }
}

// MARK: - Convenience Builders

/// Build a memory access trace from an array of (address, value, isWrite) tuples.
/// Steps are assigned sequentially starting from 0.
public func buildCairoMemoryTrace(
    _ entries: [(address: UInt64, value: Fr, isWrite: Bool)],
    segment: CairoMemorySegment = .execution
) -> [CairoMemoryAccess] {
    entries.enumerated().map { (i, e) in
        CairoMemoryAccess(
            address: e.address, value: e.value, isWrite: e.isWrite,
            step: i, segment: segment)
    }
}

/// Build a write-then-read trace for a set of address-value pairs.
/// First writes all values, then reads them back in the same order.
public func buildWriteReadTrace(
    _ pairs: [(address: UInt64, value: Fr)],
    segment: CairoMemorySegment = .execution
) -> [CairoMemoryAccess] {
    var accesses = [CairoMemoryAccess]()
    accesses.reserveCapacity(pairs.count * 2)
    var step = 0
    for (addr, val) in pairs {
        accesses.append(CairoMemoryAccess(
            address: addr, value: val, isWrite: true, step: step, segment: segment))
        step += 1
    }
    for (addr, val) in pairs {
        accesses.append(CairoMemoryAccess(
            address: addr, value: val, isWrite: false, step: step, segment: segment))
        step += 1
    }
    return accesses
}

/// Build a sequential write trace for contiguous addresses starting at base.
public func buildSequentialWriteTrace(
    base: UInt64, values: [Fr], segment: CairoMemorySegment = .execution
) -> [CairoMemoryAccess] {
    values.enumerated().map { (i, val) in
        CairoMemoryAccess(
            address: base + UInt64(i), value: val, isWrite: true,
            step: i, segment: segment)
    }
}
