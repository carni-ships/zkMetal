// GPUCairoMemoryArgTests -- Tests for GPU-accelerated Cairo memory argument engine
//
// Validates the continuous memory argument: sorting, permutation products,
// continuity checks, read-write consistency, multi-segment memory,
// memory holes, and the complete argument pipeline.

import zkMetal
import Foundation

public func runGPUCairoMemoryArgTests() {
    suite("CairoMemoryArg - Basic Construction")
    testEmptyTrace()
    testSingleWrite()
    testSingleReadAfterWrite()
    testMultipleWrites()

    suite("CairoMemoryArg - Sorting")
    testSortByAddress()
    testSortStableTieBreaking()
    testSortMixedAccesses()
    testSortAlreadySorted()

    suite("CairoMemoryArg - Permutation Product")
    testPermutationProductEmpty()
    testPermutationProductSingle()
    testPermutationProductMatches()
    testPermutationProductDeterministic()
    testPermutationProductRunningLength()

    suite("CairoMemoryArg - Continuity Check")
    testContinuityContiguousAddresses()
    testContinuitySameAddress()
    testContinuityGapDetection()
    testContinuityWriteOnceViolation()
    testContinuitySingleEntry()

    suite("CairoMemoryArg - Read-Write Consistency")
    testConsistencySimple()
    testConsistencyMultipleReads()
    testConsistencyInconsistentRead()
    testConsistencyWriteAfterWrite()
    testConsistencyNoWrites()

    suite("CairoMemoryArg - Address Uniqueness")
    testUniquenessNoDuplicates()
    testUniquenessDuplicateWrite()
    testUniquenessReadsDontCount()
    testUniquenessStrictMode()

    suite("CairoMemoryArg - Memory Holes")
    testNoHolesContiguous()
    testHolesDetected()
    testHolesFilled()
    testHolesLargeGap()

    suite("CairoMemoryArg - Multi-Segment Memory")
    testMultiSegmentBasic()
    testMultiSegmentWriteRead()
    testMultiSegmentPeek()
    testMultiSegmentArgument()
    testSegmentBoundaryValidation()

    suite("CairoMemoryArg - Full Argument Pipeline")
    testFullArgumentValid()
    testFullArgumentInvalid()
    testFullArgumentWriteReadPattern()
    testFullArgumentSequentialWrites()
    testFullArgumentVerification()

    suite("CairoMemoryArg - Ratio Argument")
    testRatioArgumentValid()
    testRatioArgumentRunningProduct()

    suite("CairoMemoryArg - Interaction Columns")
    testInteractionColumnLength()
    testCumulativeInteractionLength()
    testCumulativeInteractionStartsAtOne()

    suite("CairoMemoryArg - Convenience Builders")
    testBuildMemoryTrace()
    testBuildWriteReadTrace()
    testBuildSequentialWriteTrace()

    suite("CairoMemoryArg - Segment Statistics")
    testSegmentStatsBasic()
    testSegmentStatsMultipleSegments()
    testSegmentStatsCounts()

    suite("CairoMemoryArg - Batch Products")
    testBatchPermutationProductsEmpty()
    testBatchPermutationProductsMultiple()

    suite("CairoMemoryArg - Edge Cases")
    testLargeTrace()
    testRepeatedReadsOneAddress()
    testMaxAddressRange()
    testDuplicateWriteFinder()
}

// MARK: - Basic Construction

private func testEmptyTrace() {
    let engine = GPUCairoMemoryArgEngine()
    let result = engine.computeMemoryArgument(accesses: [])
    expect(result.isValid, "empty trace is valid")
    expect(result.permutationValid, "empty permutation valid")
    expect(result.continuityValid, "empty continuity valid")
    expect(result.consistencyValid, "empty consistency valid")
    expectEqual(result.originalTrace.count, 0, "no original accesses")
    expectEqual(result.sortedTrace.count, 0, "no sorted accesses")
}

private func testSingleWrite() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = [CairoMemoryAccess(
        address: 100, value: frFromInt(42), isWrite: true, step: 0)]
    let result = engine.computeMemoryArgument(accesses: accesses)
    expect(result.isValid, "single write is valid")
    expectEqual(result.sortedTrace.count, 1, "one sorted entry")
    expectEqual(result.sortedTrace[0].address, 100, "address preserved")
}

private func testSingleReadAfterWrite() {
    let engine = GPUCairoMemoryArgEngine()
    let val = frFromInt(77)
    let accesses = [
        CairoMemoryAccess(address: 10, value: val, isWrite: true, step: 0),
        CairoMemoryAccess(address: 10, value: val, isWrite: false, step: 1),
    ]
    let result = engine.computeMemoryArgument(accesses: accesses)
    expect(result.isValid, "write then read is valid")
    expect(result.consistencyValid, "consistency holds")
}

private func testMultipleWrites() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = [
        CairoMemoryAccess(address: 1, value: frFromInt(10), isWrite: true, step: 0),
        CairoMemoryAccess(address: 2, value: frFromInt(20), isWrite: true, step: 1),
        CairoMemoryAccess(address: 3, value: frFromInt(30), isWrite: true, step: 2),
    ]
    let result = engine.computeMemoryArgument(accesses: accesses)
    expect(result.isValid, "multiple contiguous writes valid")
    expectEqual(result.sortedTrace.count, 3, "3 sorted entries")
}

// MARK: - Sorting

private func testSortByAddress() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = [
        CairoMemoryAccess(address: 30, value: frFromInt(3), isWrite: true, step: 0),
        CairoMemoryAccess(address: 10, value: frFromInt(1), isWrite: true, step: 1),
        CairoMemoryAccess(address: 20, value: frFromInt(2), isWrite: true, step: 2),
    ]
    let sorted = engine.sortAccesses(accesses)
    expectEqual(sorted[0].address, 10, "first is addr 10")
    expectEqual(sorted[1].address, 20, "second is addr 20")
    expectEqual(sorted[2].address, 30, "third is addr 30")
}

private func testSortStableTieBreaking() {
    let engine = GPUCairoMemoryArgEngine()
    let val = frFromInt(99)
    let accesses = [
        CairoMemoryAccess(address: 5, value: val, isWrite: true, step: 3),
        CairoMemoryAccess(address: 5, value: val, isWrite: false, step: 1),
        CairoMemoryAccess(address: 5, value: val, isWrite: false, step: 7),
    ]
    let sorted = engine.sortAccesses(accesses)
    expectEqual(sorted[0].step, 1, "lowest step first")
    expectEqual(sorted[1].step, 3, "middle step second")
    expectEqual(sorted[2].step, 7, "highest step last")
}

private func testSortMixedAccesses() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = [
        CairoMemoryAccess(address: 50, value: frFromInt(5), isWrite: true, step: 0),
        CairoMemoryAccess(address: 10, value: frFromInt(1), isWrite: true, step: 1),
        CairoMemoryAccess(address: 50, value: frFromInt(5), isWrite: false, step: 2),
        CairoMemoryAccess(address: 10, value: frFromInt(1), isWrite: false, step: 3),
    ]
    let sorted = engine.sortAccesses(accesses)
    expectEqual(sorted[0].address, 10, "addr 10 first")
    expectEqual(sorted[1].address, 10, "addr 10 second")
    expectEqual(sorted[2].address, 50, "addr 50 third")
    expectEqual(sorted[3].address, 50, "addr 50 fourth")
}

private func testSortAlreadySorted() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = [
        CairoMemoryAccess(address: 1, value: frFromInt(1), isWrite: true, step: 0),
        CairoMemoryAccess(address: 2, value: frFromInt(2), isWrite: true, step: 1),
        CairoMemoryAccess(address: 3, value: frFromInt(3), isWrite: true, step: 2),
    ]
    let sorted = engine.sortAccesses(accesses)
    for i in 0..<3 {
        expectEqual(sorted[i].address, UInt64(i + 1), "addr \(i+1) in place")
    }
}

// MARK: - Permutation Product

private func testPermutationProductEmpty() {
    let engine = GPUCairoMemoryArgEngine()
    let result = engine.accumulateProduct([])
    expect(frEqual(result.finalProduct, Fr.one), "empty product is 1")
    expectEqual(result.runningProduct.count, 1, "running has just z[0]=1")
}

private func testPermutationProductSingle() {
    let engine = GPUCairoMemoryArgEngine()
    let factors = [frFromInt(42)]
    let result = engine.accumulateProduct(factors)
    expect(frEqual(result.finalProduct, frFromInt(42)), "single factor product")
    expectEqual(result.runningProduct.count, 2, "running has z[0] and z[1]")
    expect(frEqual(result.runningProduct[0], Fr.one), "z[0] = 1")
}

private func testPermutationProductMatches() {
    let engine = GPUCairoMemoryArgEngine()
    // Build a simple trace and check that original and sorted give same product
    let val = frFromInt(55)
    let accesses = [
        CairoMemoryAccess(address: 3, value: val, isWrite: true, step: 0),
        CairoMemoryAccess(address: 1, value: frFromInt(11), isWrite: true, step: 1),
        CairoMemoryAccess(address: 2, value: frFromInt(22), isWrite: true, step: 2),
    ]
    let sorted = engine.sortAccesses(accesses)
    let origFactors = engine.computePermutationFactors(accesses)
    let sortFactors = engine.computeSortedPermutationFactors(sorted)
    let origProd = engine.accumulateProduct(origFactors)
    let sortProd = engine.accumulateProduct(sortFactors)
    expect(frEqual(origProd.finalProduct, sortProd.finalProduct),
           "permutation products must match")
}

private func testPermutationProductDeterministic() {
    let engine = GPUCairoMemoryArgEngine()
    let factors = [frFromInt(3), frFromInt(5), frFromInt(7)]
    let r1 = engine.accumulateProduct(factors)
    let r2 = engine.accumulateProduct(factors)
    expect(frEqual(r1.finalProduct, r2.finalProduct), "deterministic product")
}

private func testPermutationProductRunningLength() {
    let engine = GPUCairoMemoryArgEngine()
    let n = 10
    let factors = (1...n).map { frFromInt(UInt64($0)) }
    let result = engine.accumulateProduct(factors)
    expectEqual(result.runningProduct.count, n + 1, "running product has n+1 entries")
    expect(frEqual(result.runningProduct[0], Fr.one), "starts at 1")
}

// MARK: - Continuity Check

private func testContinuityContiguousAddresses() {
    let engine = GPUCairoMemoryArgEngine()
    let sorted = [
        SortedMemoryEntry(address: 1, value: frFromInt(10), isWrite: true, step: 0, originalIndex: 0),
        SortedMemoryEntry(address: 2, value: frFromInt(20), isWrite: true, step: 1, originalIndex: 1),
        SortedMemoryEntry(address: 3, value: frFromInt(30), isWrite: true, step: 2, originalIndex: 2),
    ]
    let result = engine.checkContinuity(sorted)
    expect(result.isValid, "contiguous addresses pass continuity")
    expectEqual(result.distinctAddressCount, 3, "3 distinct addresses")
    expectEqual(result.addressGaps.count, 0, "no gaps")
}

private func testContinuitySameAddress() {
    let engine = GPUCairoMemoryArgEngine()
    let val = frFromInt(42)
    let sorted = [
        SortedMemoryEntry(address: 5, value: val, isWrite: true, step: 0, originalIndex: 0),
        SortedMemoryEntry(address: 5, value: val, isWrite: false, step: 1, originalIndex: 1),
        SortedMemoryEntry(address: 5, value: val, isWrite: false, step: 2, originalIndex: 2),
    ]
    let result = engine.checkContinuity(sorted)
    expect(result.isValid, "same address with same value passes")
    expectEqual(result.distinctAddressCount, 1, "1 distinct address")
}

private func testContinuityGapDetection() {
    let engine = GPUCairoMemoryArgEngine()
    let sorted = [
        SortedMemoryEntry(address: 1, value: frFromInt(10), isWrite: true, step: 0, originalIndex: 0),
        SortedMemoryEntry(address: 5, value: frFromInt(50), isWrite: true, step: 1, originalIndex: 1),
    ]
    let result = engine.checkContinuity(sorted)
    expect(result.isValid, "gaps are noted but don't cause invalidity alone")
    expectEqual(result.addressGaps.count, 1, "one gap detected")
    expectEqual(result.addressGaps[0].gap, 4, "gap of 4")
}

private func testContinuityWriteOnceViolation() {
    let engine = GPUCairoMemoryArgEngine()
    let sorted = [
        SortedMemoryEntry(address: 10, value: frFromInt(1), isWrite: true, step: 0, originalIndex: 0),
        SortedMemoryEntry(address: 10, value: frFromInt(2), isWrite: true, step: 1, originalIndex: 1),
    ]
    let result = engine.checkContinuity(sorted)
    expect(!result.isValid, "value mismatch at same address is invalid")
    expectEqual(result.violationIndices.count, 1, "one violation")
}

private func testContinuitySingleEntry() {
    let engine = GPUCairoMemoryArgEngine()
    let sorted = [
        SortedMemoryEntry(address: 42, value: frFromInt(1), isWrite: true, step: 0, originalIndex: 0),
    ]
    let result = engine.checkContinuity(sorted)
    expect(result.isValid, "single entry always valid")
    expectEqual(result.distinctAddressCount, 1, "1 distinct")
}

// MARK: - Read-Write Consistency

private func testConsistencySimple() {
    let engine = GPUCairoMemoryArgEngine()
    let val = frFromInt(100)
    let sorted = [
        SortedMemoryEntry(address: 1, value: val, isWrite: true, step: 0, originalIndex: 0),
        SortedMemoryEntry(address: 1, value: val, isWrite: false, step: 1, originalIndex: 1),
    ]
    expect(engine.checkReadWriteConsistency(sorted), "read matches write")
}

private func testConsistencyMultipleReads() {
    let engine = GPUCairoMemoryArgEngine()
    let val = frFromInt(200)
    let sorted = [
        SortedMemoryEntry(address: 7, value: val, isWrite: true, step: 0, originalIndex: 0),
        SortedMemoryEntry(address: 7, value: val, isWrite: false, step: 1, originalIndex: 1),
        SortedMemoryEntry(address: 7, value: val, isWrite: false, step: 2, originalIndex: 2),
        SortedMemoryEntry(address: 7, value: val, isWrite: false, step: 3, originalIndex: 3),
    ]
    expect(engine.checkReadWriteConsistency(sorted), "multiple reads match single write")
}

private func testConsistencyInconsistentRead() {
    let engine = GPUCairoMemoryArgEngine()
    let sorted = [
        SortedMemoryEntry(address: 1, value: frFromInt(10), isWrite: true, step: 0, originalIndex: 0),
        SortedMemoryEntry(address: 1, value: frFromInt(99), isWrite: false, step: 1, originalIndex: 1),
    ]
    expect(!engine.checkReadWriteConsistency(sorted), "mismatched read fails consistency")
}

private func testConsistencyWriteAfterWrite() {
    let engine = GPUCairoMemoryArgEngine()
    let val = frFromInt(50)
    // Same value at same address: allowed (idempotent write)
    let sorted = [
        SortedMemoryEntry(address: 3, value: val, isWrite: true, step: 0, originalIndex: 0),
        SortedMemoryEntry(address: 3, value: val, isWrite: true, step: 1, originalIndex: 1),
    ]
    expect(engine.checkReadWriteConsistency(sorted), "same-value double write is consistent")
}

private func testConsistencyNoWrites() {
    let engine = GPUCairoMemoryArgEngine()
    expect(engine.checkReadWriteConsistency([]), "empty sorted trace is consistent")
}

// MARK: - Address Uniqueness

private func testUniquenessNoDuplicates() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = [
        CairoMemoryAccess(address: 1, value: frFromInt(10), isWrite: true, step: 0),
        CairoMemoryAccess(address: 2, value: frFromInt(20), isWrite: true, step: 1),
        CairoMemoryAccess(address: 3, value: frFromInt(30), isWrite: true, step: 2),
    ]
    expect(engine.checkAddressUniqueness(accesses), "unique writes pass")
}

private func testUniquenessDuplicateWrite() {
    var engine = GPUCairoMemoryArgEngine()
    engine.strictWriteOnce = true
    let accesses = [
        CairoMemoryAccess(address: 1, value: frFromInt(10), isWrite: true, step: 0),
        CairoMemoryAccess(address: 1, value: frFromInt(10), isWrite: true, step: 1),
    ]
    expect(!engine.checkAddressUniqueness(accesses), "strict mode catches duplicate write")
}

private func testUniquenessReadsDontCount() {
    var engine = GPUCairoMemoryArgEngine()
    engine.strictWriteOnce = true
    let accesses = [
        CairoMemoryAccess(address: 1, value: frFromInt(10), isWrite: true, step: 0),
        CairoMemoryAccess(address: 1, value: frFromInt(10), isWrite: false, step: 1),
        CairoMemoryAccess(address: 1, value: frFromInt(10), isWrite: false, step: 2),
    ]
    expect(engine.checkAddressUniqueness(accesses), "reads don't trigger uniqueness violation")
}

private func testUniquenessStrictMode() {
    // Non-strict mode allows duplicate writes (default)
    let engine = GPUCairoMemoryArgEngine()
    let accesses = [
        CairoMemoryAccess(address: 5, value: frFromInt(10), isWrite: true, step: 0),
        CairoMemoryAccess(address: 5, value: frFromInt(10), isWrite: true, step: 1),
    ]
    expect(engine.checkAddressUniqueness(accesses), "non-strict allows duplicate writes")
}

// MARK: - Memory Holes

private func testNoHolesContiguous() {
    let engine = GPUCairoMemoryArgEngine()
    let sorted = [
        SortedMemoryEntry(address: 1, value: frFromInt(1), isWrite: true, step: 0, originalIndex: 0),
        SortedMemoryEntry(address: 2, value: frFromInt(2), isWrite: true, step: 1, originalIndex: 1),
        SortedMemoryEntry(address: 3, value: frFromInt(3), isWrite: true, step: 2, originalIndex: 2),
    ]
    let holes = engine.computeMemoryHoles(sorted)
    expectEqual(holes.count, 0, "no holes for contiguous addresses")
}

private func testHolesDetected() {
    let engine = GPUCairoMemoryArgEngine()
    let sorted = [
        SortedMemoryEntry(address: 1, value: frFromInt(1), isWrite: true, step: 0, originalIndex: 0),
        SortedMemoryEntry(address: 4, value: frFromInt(4), isWrite: true, step: 1, originalIndex: 1),
    ]
    let holes = engine.computeMemoryHoles(sorted)
    expectEqual(holes.count, 2, "2 holes: addresses 2, 3")
    expectEqual(holes[0], 2, "first hole at 2")
    expectEqual(holes[1], 3, "second hole at 3")
}

private func testHolesFilled() {
    let engine = GPUCairoMemoryArgEngine()
    let sorted = [
        SortedMemoryEntry(address: 10, value: frFromInt(1), isWrite: true, step: 0, originalIndex: 0),
        SortedMemoryEntry(address: 13, value: frFromInt(4), isWrite: true, step: 1, originalIndex: 1),
    ]
    let filled = engine.fillMemoryHoles(sorted)
    expectEqual(filled.count, 4, "2 original + 2 holes filled")
    expectEqual(filled[0].address, 10, "original addr 10")
    expectEqual(filled[1].address, 11, "hole at 11")
    expectEqual(filled[2].address, 12, "hole at 12")
    expectEqual(filled[3].address, 13, "original addr 13")
    // Holes have zero value
    expect(frEqual(filled[1].value, Fr.zero), "hole value is zero")
}

private func testHolesLargeGap() {
    let engine = GPUCairoMemoryArgEngine()
    let sorted = [
        SortedMemoryEntry(address: 0, value: frFromInt(1), isWrite: true, step: 0, originalIndex: 0),
        SortedMemoryEntry(address: 10, value: frFromInt(2), isWrite: true, step: 1, originalIndex: 1),
    ]
    let holes = engine.computeMemoryHoles(sorted)
    expectEqual(holes.count, 9, "9 holes for gap of 10")
    for (i, hole) in holes.enumerated() {
        expectEqual(hole, UInt64(i + 1), "hole at addr \(i+1)")
    }
}

// MARK: - Multi-Segment Memory

private func testMultiSegmentBasic() {
    let segments = [
        CairoSegmentConfig(segment: .program, baseAddress: 0, size: 100),
        CairoSegmentConfig(segment: .execution, baseAddress: 100, size: 200),
    ]
    let mem = CairoMultiSegmentMemory(segments: segments)
    expectEqual(mem.totalAccessCount, 0, "no accesses initially")
    expectEqual(mem.distinctAddressCount, 0, "no addresses initially")
}

private func testMultiSegmentWriteRead() {
    let segments = [
        CairoSegmentConfig(segment: .program, baseAddress: 0, size: 100),
        CairoSegmentConfig(segment: .execution, baseAddress: 100, size: 200),
    ]
    var mem = CairoMultiSegmentMemory(segments: segments)

    let ok = mem.write(segment: .program, offset: 5, value: frFromInt(42), step: 0)
    expect(ok, "write to program segment succeeds")

    let val = mem.read(segment: .program, offset: 5, step: 1)
    expect(frEqual(val, frFromInt(42)), "read back correct value")
    expectEqual(mem.totalAccessCount, 2, "one write + one read")
}

private func testMultiSegmentPeek() {
    let segments = [
        CairoSegmentConfig(segment: .execution, baseAddress: 0, size: 50),
    ]
    var mem = CairoMultiSegmentMemory(segments: segments)
    _ = mem.write(segment: .execution, offset: 3, value: frFromInt(77), step: 0)

    let peeked = mem.peek(segment: .execution, offset: 3)
    expect(peeked != nil, "peek finds written value")
    expect(frEqual(peeked!, frFromInt(77)), "peek returns correct value")

    let missing = mem.peek(segment: .execution, offset: 99)
    expect(missing == nil, "peek returns nil for unwritten address")
}

private func testMultiSegmentArgument() {
    let segments = [
        CairoSegmentConfig(segment: .program, baseAddress: 0, size: 10),
        CairoSegmentConfig(segment: .execution, baseAddress: 10, size: 10),
    ]
    var mem = CairoMultiSegmentMemory(segments: segments)

    _ = mem.write(segment: .program, offset: 0, value: frFromInt(1), step: 0)
    _ = mem.write(segment: .program, offset: 1, value: frFromInt(2), step: 1)
    _ = mem.write(segment: .execution, offset: 0, value: frFromInt(10), step: 2)
    _ = mem.read(segment: .program, offset: 0, step: 3)
    _ = mem.read(segment: .execution, offset: 0, step: 4)

    let engine = GPUCairoMemoryArgEngine()
    let result = engine.computeMultiSegmentArgument(memory: mem)
    expect(result.permutationValid, "multi-segment permutation valid")
    expect(result.consistencyValid, "multi-segment consistency valid")
}

private func testSegmentBoundaryValidation() {
    let segments = [
        CairoSegmentConfig(segment: .program, baseAddress: 0, size: 5),
    ]
    let accesses = [
        CairoMemoryAccess(address: 3, value: frFromInt(1), isWrite: true, step: 0, segment: .program),
        CairoMemoryAccess(address: 10, value: frFromInt(2), isWrite: true, step: 1, segment: .program),
    ]
    let engine = GPUCairoMemoryArgEngine()
    let violations = engine.validateSegmentBoundaries(accesses: accesses, segments: segments)
    expectEqual(violations.count, 1, "one out-of-bounds violation")
    expectEqual(violations[0].2, 10, "violation at address 10")
}

// MARK: - Full Argument Pipeline

private func testFullArgumentValid() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = buildWriteReadTrace([
        (address: 1, value: frFromInt(100)),
        (address: 2, value: frFromInt(200)),
        (address: 3, value: frFromInt(300)),
    ])
    let result = engine.computeMemoryArgument(accesses: accesses)
    expect(result.permutationValid, "permutation valid for write-read pattern")
    expect(result.consistencyValid, "consistency valid")
    expect(result.isValid, "overall argument valid")
}

private func testFullArgumentInvalid() {
    let engine = GPUCairoMemoryArgEngine()
    // Manually construct an inconsistent read: write 10 then "read" a different value
    let accesses = [
        CairoMemoryAccess(address: 1, value: frFromInt(10), isWrite: true, step: 0),
        CairoMemoryAccess(address: 1, value: frFromInt(99), isWrite: false, step: 1),
    ]
    let result = engine.computeMemoryArgument(accesses: accesses)
    // The consistency check should fail because the read value doesn't match the write
    expect(!result.consistencyValid, "inconsistent read fails")
    expect(!result.isValid, "overall argument invalid")
}

private func testFullArgumentWriteReadPattern() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = buildWriteReadTrace([
        (address: 10, value: frFromInt(1)),
        (address: 20, value: frFromInt(2)),
        (address: 30, value: frFromInt(3)),
        (address: 40, value: frFromInt(4)),
        (address: 50, value: frFromInt(5)),
    ])
    let result = engine.computeMemoryArgument(accesses: accesses)
    expect(result.isValid, "5-pair write-read pattern valid")
    expectEqual(result.originalTrace.count, 10, "10 accesses total")
}

private func testFullArgumentSequentialWrites() {
    let engine = GPUCairoMemoryArgEngine()
    let values = (1...8).map { frFromInt(UInt64($0) * 11) }
    let accesses = buildSequentialWriteTrace(base: 100, values: values)
    let result = engine.computeMemoryArgument(accesses: accesses)
    expect(result.permutationValid, "sequential writes: permutation valid")
    expect(result.consistencyValid, "sequential writes: consistency valid")
    expectEqual(result.sortedTrace.count, 8, "8 sorted entries")
}

private func testFullArgumentVerification() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = buildWriteReadTrace([
        (address: 5, value: frFromInt(55)),
        (address: 6, value: frFromInt(66)),
    ])
    let result = engine.computeMemoryArgument(accesses: accesses)
    let verified = engine.verifyMemoryArgument(result)
    expect(verified, "verification passes for valid argument")
}

// MARK: - Ratio Argument

private func testRatioArgumentValid() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = buildWriteReadTrace([
        (address: 1, value: frFromInt(10)),
        (address: 2, value: frFromInt(20)),
    ])
    let ratio = engine.computeRatioArgument(accesses: accesses)
    // The ratio product should equal 1 for a valid permutation
    expect(frEqual(ratio.finalProduct, Fr.one), "ratio argument product is 1")
}

private func testRatioArgumentRunningProduct() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = buildWriteReadTrace([
        (address: 1, value: frFromInt(7)),
    ])
    let ratio = engine.computeRatioArgument(accesses: accesses)
    expectEqual(ratio.runningProduct.count, 3, "running product has n+1 entries")
    expect(frEqual(ratio.runningProduct[0], Fr.one), "starts at 1")
}

// MARK: - Interaction Columns

private func testInteractionColumnLength() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = [
        CairoMemoryAccess(address: 1, value: frFromInt(10), isWrite: true, step: 0),
        CairoMemoryAccess(address: 2, value: frFromInt(20), isWrite: true, step: 1),
    ]
    let col = engine.generateInteractionColumn(accesses: accesses)
    expectEqual(col.count, 2, "interaction column same length as trace")
}

private func testCumulativeInteractionLength() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = [
        CairoMemoryAccess(address: 1, value: frFromInt(10), isWrite: true, step: 0),
        CairoMemoryAccess(address: 2, value: frFromInt(20), isWrite: true, step: 1),
        CairoMemoryAccess(address: 3, value: frFromInt(30), isWrite: true, step: 2),
    ]
    let cum = engine.generateCumulativeInteraction(accesses: accesses)
    expectEqual(cum.count, 4, "cumulative has n+1 entries")
}

private func testCumulativeInteractionStartsAtOne() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = [
        CairoMemoryAccess(address: 1, value: frFromInt(10), isWrite: true, step: 0),
    ]
    let cum = engine.generateCumulativeInteraction(accesses: accesses)
    expect(frEqual(cum[0], Fr.one), "cumulative starts at 1")
}

// MARK: - Convenience Builders

private func testBuildMemoryTrace() {
    let entries: [(address: UInt64, value: Fr, isWrite: Bool)] = [
        (1, frFromInt(10), true),
        (2, frFromInt(20), true),
        (1, frFromInt(10), false),
    ]
    let trace = buildCairoMemoryTrace(entries, segment: .execution)
    expectEqual(trace.count, 3, "3 entries")
    expectEqual(trace[0].step, 0, "step 0")
    expectEqual(trace[1].step, 1, "step 1")
    expectEqual(trace[2].step, 2, "step 2")
    expectEqual(trace[0].address, 1, "addr 1")
    expect(trace[0].isWrite, "first is write")
    expect(!trace[2].isWrite, "third is read")
}

private func testBuildWriteReadTrace() {
    let pairs: [(address: UInt64, value: Fr)] = [
        (10, frFromInt(100)),
        (20, frFromInt(200)),
    ]
    let trace = buildWriteReadTrace(pairs)
    expectEqual(trace.count, 4, "2 writes + 2 reads = 4")
    expect(trace[0].isWrite, "first is write")
    expect(trace[1].isWrite, "second is write")
    expect(!trace[2].isWrite, "third is read")
    expect(!trace[3].isWrite, "fourth is read")
    expectEqual(trace[0].address, 10, "write addr 10")
    expectEqual(trace[2].address, 10, "read addr 10")
}

private func testBuildSequentialWriteTrace() {
    let values = [frFromInt(1), frFromInt(2), frFromInt(3)]
    let trace = buildSequentialWriteTrace(base: 50, values: values)
    expectEqual(trace.count, 3, "3 sequential writes")
    expectEqual(trace[0].address, 50, "base addr")
    expectEqual(trace[1].address, 51, "base + 1")
    expectEqual(trace[2].address, 52, "base + 2")
    for a in trace { expect(a.isWrite, "all are writes") }
}

// MARK: - Segment Statistics

private func testSegmentStatsBasic() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = [
        CairoMemoryAccess(address: 1, value: frFromInt(1), isWrite: true, step: 0, segment: .program),
    ]
    let stats = engine.computeSegmentStats(accesses)
    expectEqual(stats.count, 1, "one segment")
    expectEqual(stats[.program]?.accessCount, 1, "1 access in program segment")
    expectEqual(stats[.program]?.writeCount, 1, "1 write")
    expectEqual(stats[.program]?.readCount, 0, "0 reads")
}

private func testSegmentStatsMultipleSegments() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = [
        CairoMemoryAccess(address: 1, value: frFromInt(1), isWrite: true, step: 0, segment: .program),
        CairoMemoryAccess(address: 100, value: frFromInt(2), isWrite: true, step: 1, segment: .execution),
        CairoMemoryAccess(address: 200, value: frFromInt(3), isWrite: true, step: 2, segment: .output),
    ]
    let stats = engine.computeSegmentStats(accesses)
    expectEqual(stats.count, 3, "three segments")
    expect(stats[.program] != nil, "program segment present")
    expect(stats[.execution] != nil, "execution segment present")
    expect(stats[.output] != nil, "output segment present")
}

private func testSegmentStatsCounts() {
    let engine = GPUCairoMemoryArgEngine()
    let val = frFromInt(42)
    let accesses = [
        CairoMemoryAccess(address: 1, value: val, isWrite: true, step: 0, segment: .execution),
        CairoMemoryAccess(address: 2, value: val, isWrite: true, step: 1, segment: .execution),
        CairoMemoryAccess(address: 1, value: val, isWrite: false, step: 2, segment: .execution),
        CairoMemoryAccess(address: 2, value: val, isWrite: false, step: 3, segment: .execution),
        CairoMemoryAccess(address: 1, value: val, isWrite: false, step: 4, segment: .execution),
    ]
    let stats = engine.computeSegmentStats(accesses)
    let s = stats[.execution]!
    expectEqual(s.accessCount, 5, "5 total accesses")
    expectEqual(s.writeCount, 2, "2 writes")
    expectEqual(s.readCount, 3, "3 reads")
    expectEqual(s.uniqueAddresses, 2, "2 unique addresses")
}

// MARK: - Batch Products

private func testBatchPermutationProductsEmpty() {
    let engine = GPUCairoMemoryArgEngine()
    let results = engine.batchPermutationProducts(traces: [])
    expectEqual(results.count, 0, "empty batch")
}

private func testBatchPermutationProductsMultiple() {
    let engine = GPUCairoMemoryArgEngine()
    let trace1 = [
        CairoMemoryAccess(address: 1, value: frFromInt(10), isWrite: true, step: 0),
    ]
    let trace2 = [
        CairoMemoryAccess(address: 2, value: frFromInt(20), isWrite: true, step: 0),
        CairoMemoryAccess(address: 3, value: frFromInt(30), isWrite: true, step: 1),
    ]
    let results = engine.batchPermutationProducts(traces: [trace1, trace2])
    expectEqual(results.count, 2, "two batch results")
    expectEqual(results[0].runningProduct.count, 2, "first has 1+1 entries")
    expectEqual(results[1].runningProduct.count, 3, "second has 2+1 entries")
}

// MARK: - Edge Cases

private func testLargeTrace() {
    let engine = GPUCairoMemoryArgEngine()
    let n = 1000
    var accesses = [CairoMemoryAccess]()
    accesses.reserveCapacity(n * 2)
    // Write n addresses then read them all back
    for i in 0..<n {
        accesses.append(CairoMemoryAccess(
            address: UInt64(i), value: frFromInt(UInt64(i * 7 + 3)),
            isWrite: true, step: i))
    }
    for i in 0..<n {
        accesses.append(CairoMemoryAccess(
            address: UInt64(i), value: frFromInt(UInt64(i * 7 + 3)),
            isWrite: false, step: n + i))
    }
    let result = engine.computeMemoryArgument(accesses: accesses)
    expect(result.permutationValid, "large trace: permutation valid")
    expect(result.consistencyValid, "large trace: consistency valid")
    expect(result.isValid, "large trace: overall valid")
    expectEqual(result.originalTrace.count, n * 2, "2n accesses")
}

private func testRepeatedReadsOneAddress() {
    let engine = GPUCairoMemoryArgEngine()
    let val = frFromInt(12345)
    var accesses = [CairoMemoryAccess]()
    accesses.append(CairoMemoryAccess(address: 1, value: val, isWrite: true, step: 0))
    for i in 1...20 {
        accesses.append(CairoMemoryAccess(address: 1, value: val, isWrite: false, step: i))
    }
    let result = engine.computeMemoryArgument(accesses: accesses)
    expect(result.isValid, "20 reads of same address valid")
    expectEqual(result.sortedTrace.count, 21, "21 sorted entries")
}

private func testMaxAddressRange() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = [
        CairoMemoryAccess(address: 0, value: frFromInt(1), isWrite: true, step: 0),
        CairoMemoryAccess(address: 1000000, value: frFromInt(2), isWrite: true, step: 1),
    ]
    let result = engine.computeMemoryArgument(accesses: accesses)
    expect(result.permutationValid, "wide address range: permutation valid")
    let sorted = result.sortedTrace
    expectEqual(sorted[0].address, 0, "smallest address first")
    expectEqual(sorted[1].address, 1000000, "largest address second")
}

private func testDuplicateWriteFinder() {
    let engine = GPUCairoMemoryArgEngine()
    let accesses = [
        CairoMemoryAccess(address: 1, value: frFromInt(10), isWrite: true, step: 0),
        CairoMemoryAccess(address: 2, value: frFromInt(20), isWrite: true, step: 1),
        CairoMemoryAccess(address: 1, value: frFromInt(10), isWrite: true, step: 2),
        CairoMemoryAccess(address: 1, value: frFromInt(10), isWrite: true, step: 3),
        CairoMemoryAccess(address: 3, value: frFromInt(30), isWrite: true, step: 4),
    ]
    let dupes = engine.findDuplicateWrites(accesses)
    expectEqual(dupes.count, 1, "one duplicate address")
    expectEqual(dupes[1], 3, "address 1 written 3 times")
}
