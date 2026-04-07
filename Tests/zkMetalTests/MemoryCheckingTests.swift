import zkMetal
import Foundation

public func runMemoryCheckingTests() {
    suite("Memory Checking Engine")

    // Helper: compare two Fr values
    func frEqual(_ a: Fr, _ b: Fr) -> Bool {
        return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
               a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
    }

    // =========================================================================
    // SECTION 1: MemCheckOp enum basics
    // =========================================================================

    do {
        let readOp = MemCheckOp.read(addr: 5, value: frFromInt(42), timestamp: 1)
        let writeOp = MemCheckOp.write(addr: 5, value: frFromInt(42), timestamp: 0)

        expectEqual(readOp.addr, 5, "Read op addr")
        expect(frEqual(readOp.value, frFromInt(42)), "Read op value")
        expectEqual(readOp.timestamp, 1, "Read op timestamp")
        expect(!readOp.isWrite, "Read op is not write")

        expectEqual(writeOp.addr, 5, "Write op addr")
        expect(frEqual(writeOp.value, frFromInt(42)), "Write op value")
        expectEqual(writeOp.timestamp, 0, "Write op timestamp")
        expect(writeOp.isWrite, "Write op is write")
    }

    // =========================================================================
    // SECTION 2: Simple read-after-write
    // =========================================================================

    // Test: write then read same address -- should succeed
    do {
        let checker = OfflineMemoryChecker()
        checker.write(addr: 0, value: frFromInt(42))
        checker.read(addr: 0, value: frFromInt(42))
        let result = checker.check()
        expect(result.success, "Simple write-then-read succeeds")
        expectEqual(result.numAddresses, 1, "One address accessed")
        expectEqual(result.numOriginalOps, 2, "Two original ops")
        expect(result.failureReason == nil, "No failure reason")
    }

    // Test: write then read with readCorrect helper
    do {
        let checker = OfflineMemoryChecker()
        checker.write(addr: 10, value: frFromInt(99))
        let val = checker.readCorrect(addr: 10)
        expect(frEqual(val, frFromInt(99)), "readCorrect returns written value")
        let result = checker.check()
        expect(result.success, "readCorrect trace is valid")
    }

    // =========================================================================
    // SECTION 3: Multiple addresses interleaved
    // =========================================================================

    do {
        let checker = OfflineMemoryChecker()
        // Write to addresses 0, 1, 2
        checker.write(addr: 0, value: frFromInt(10))
        checker.write(addr: 1, value: frFromInt(20))
        checker.write(addr: 2, value: frFromInt(30))
        // Read interleaved
        checker.read(addr: 1, value: frFromInt(20))
        checker.read(addr: 0, value: frFromInt(10))
        checker.read(addr: 2, value: frFromInt(30))
        // Read again
        checker.read(addr: 0, value: frFromInt(10))

        let result = checker.check()
        expect(result.success, "Interleaved multi-address trace is valid")
        expectEqual(result.numAddresses, 3, "Three addresses accessed")
        expectEqual(result.numOriginalOps, 7, "Seven original ops")
    }

    // =========================================================================
    // SECTION 4: Overwrite and re-read
    // =========================================================================

    // Write a, write a again with different value, read a -> should get second value
    do {
        let checker = OfflineMemoryChecker()
        checker.write(addr: 5, value: frFromInt(100))
        checker.write(addr: 5, value: frFromInt(200))
        checker.read(addr: 5, value: frFromInt(200))

        let result = checker.check()
        expect(result.success, "Overwrite-then-read returns second value")
    }

    // Multiple overwrites
    do {
        let checker = OfflineMemoryChecker()
        checker.write(addr: 0, value: frFromInt(1))
        checker.write(addr: 0, value: frFromInt(2))
        checker.write(addr: 0, value: frFromInt(3))
        checker.read(addr: 0, value: frFromInt(3))
        checker.write(addr: 0, value: frFromInt(4))
        checker.read(addr: 0, value: frFromInt(4))

        let result = checker.check()
        expect(result.success, "Multiple overwrites trace is valid")
    }

    // =========================================================================
    // SECTION 5: Invalid trace detection
    // =========================================================================

    // Read returns wrong value -> should fail
    do {
        let checker = OfflineMemoryChecker()
        checker.write(addr: 0, value: frFromInt(42))
        // Intentionally read wrong value
        checker.read(addr: 0, value: frFromInt(99))

        let result = checker.check()
        expect(!result.success, "Wrong read value detected as invalid")
        expect(result.failureReason != nil, "Failure reason is set")
    }

    // Read from unwritten address with wrong value (should be 0)
    do {
        let checker = OfflineMemoryChecker()
        checker.write(addr: 1, value: frFromInt(10))
        // Address 0 was never written, init is 0, but we claim to read 5
        checker.read(addr: 0, value: frFromInt(5))

        let result = checker.check()
        expect(!result.success, "Reading non-zero from uninitialized address is invalid")
    }

    // Read from unwritten address with value 0 (should succeed since init is 0)
    do {
        let checker = OfflineMemoryChecker()
        checker.write(addr: 1, value: frFromInt(10))
        checker.read(addr: 0, value: Fr.zero)

        let result = checker.check()
        expect(result.success, "Reading zero from uninitialized address is valid")
    }

    // Read stale value after overwrite
    do {
        let checker = OfflineMemoryChecker()
        checker.write(addr: 0, value: frFromInt(10))
        checker.write(addr: 0, value: frFromInt(20))
        // Read the OLD value (should fail)
        checker.read(addr: 0, value: frFromInt(10))

        let result = checker.check()
        expect(!result.success, "Reading stale value after overwrite is invalid")
    }

    // =========================================================================
    // SECTION 6: Engine direct API
    // =========================================================================

    do {
        let engine = MemoryCheckingEngine()
        let ops: [MemCheckOp] = [
            .write(addr: 0, value: Fr.zero, timestamp: 0),    // init
            .write(addr: 0, value: frFromInt(7), timestamp: 1),
            .read(addr: 0, value: frFromInt(7), timestamp: 2),
            .read(addr: 0, value: frFromInt(7), timestamp: 3), // final read
        ]
        let transcript = Transcript(label: "test-engine-direct")
        let result = engine.verify(ops: ops, transcript: transcript)
        expect(result.isValid, "Engine direct: valid trace")
        expect(frEqual(result.readFingerprint, result.writeFingerprint),
               "Engine direct: fingerprints match")
    }

    // Engine with auto init/finalize
    do {
        let engine = MemoryCheckingEngine()
        let ops: [MemCheckOp] = [
            .write(addr: 0, value: frFromInt(7), timestamp: 1),
            .read(addr: 0, value: frFromInt(7), timestamp: 2),
        ]
        let transcript = Transcript(label: "test-engine-auto")
        let result = engine.verifyWithInitFinalize(ops: ops, transcript: transcript)
        expect(result.isValid, "Engine auto-init: valid trace")
    }

    // =========================================================================
    // SECTION 7: Configurable memory size and address width
    // =========================================================================

    do {
        let smallConfig = MemoryCheckingConfig(memorySize: 16, addressWidth: 4)
        let engine = MemoryCheckingEngine(config: smallConfig)

        // Address within range
        let ops1: [MemCheckOp] = [
            .write(addr: 0, value: Fr.zero, timestamp: 0),
            .write(addr: 15, value: frFromInt(1), timestamp: 1),
            .read(addr: 15, value: frFromInt(1), timestamp: 2),
            .read(addr: 0, value: Fr.zero, timestamp: 3),
        ]
        let t1 = Transcript(label: "test-addr-ok")
        let r1 = engine.verify(ops: ops1, transcript: t1)
        expect(r1.isValid, "Address within 4-bit range succeeds")

        // Address out of range
        let ops2: [MemCheckOp] = [
            .write(addr: 16, value: frFromInt(1), timestamp: 1),
        ]
        let t2 = Transcript(label: "test-addr-oob")
        let r2 = engine.verify(ops: ops2, transcript: t2)
        expect(!r2.isValid, "Address exceeding 4-bit range fails")
    }

    // =========================================================================
    // SECTION 8: Batch memory checker
    // =========================================================================

    do {
        let batch = BatchOfflineMemoryChecker()

        let regs = batch.space("registers")
        regs.write(addr: 0, value: frFromInt(100))
        regs.read(addr: 0, value: frFromInt(100))

        let mem = batch.space("main_memory")
        mem.write(addr: 0, value: frFromInt(200))
        mem.read(addr: 0, value: frFromInt(200))

        let (allValid, results) = batch.checkAll()
        expect(allValid, "Batch checker: all spaces valid")
        expectEqual(results.count, 2, "Batch checker: two spaces")
    }

    // =========================================================================
    // SECTION 9: Stress test with 10K random operations
    // =========================================================================

    do {
        let t0 = CFAbsoluteTimeGetCurrent()
        let numOps = 10_000
        let numAddrs: UInt64 = 64

        // Deterministic PRNG
        var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
        func nextRng() -> UInt64 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return rng >> 32
        }

        let checker = OfflineMemoryChecker(config: MemoryCheckingConfig(memorySize: 256, addressWidth: 8))
        var localMemory = [UInt64: Fr]()

        for _ in 0..<numOps {
            let addr = nextRng() % numAddrs
            let isWrite = (nextRng() % 3) != 0  // ~67% writes to ensure reads have values

            if isWrite {
                let val = frFromInt(nextRng() % 10000)
                checker.write(addr: addr, value: val)
                localMemory[addr] = val
            } else {
                let expected = localMemory[addr] ?? Fr.zero
                checker.read(addr: addr, value: expected)
            }
        }

        let result = checker.check()
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        expect(result.success, "Stress test 10K ops valid")
        expect(result.numOriginalOps == numOps, "Stress test: correct op count")
        print("  10K ops memory check: \(String(format: "%.1f", elapsed * 1000))ms")
    }

    // =========================================================================
    // SECTION 10: Stress test with invalid trace (10K ops, one bad read)
    // =========================================================================

    do {
        let numOps = 10_000
        let numAddrs: UInt64 = 64

        var rng: UInt64 = 0xBAAD_F00D_1234_5678
        func nextRng() -> UInt64 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return rng >> 32
        }

        let checker = OfflineMemoryChecker(config: MemoryCheckingConfig(memorySize: 256, addressWidth: 8))
        var localMemory = [UInt64: Fr]()

        // Generate valid ops
        for _ in 0..<numOps {
            let addr = nextRng() % numAddrs
            let isWrite = (nextRng() % 3) != 0

            if isWrite {
                let val = frFromInt(nextRng() % 10000)
                checker.write(addr: addr, value: val)
                localMemory[addr] = val
            } else {
                let expected = localMemory[addr] ?? Fr.zero
                checker.read(addr: addr, value: expected)
            }
        }

        // Inject one bad read after all valid ops
        let badAddr = nextRng() % numAddrs
        let expected = localMemory[badAddr] ?? Fr.zero
        checker.read(addr: badAddr, value: frAdd(expected, Fr.one))

        let result = checker.check()
        expect(!result.success, "Stress test with bad read detected as invalid")
    }

    // =========================================================================
    // SECTION 11: Version check
    // =========================================================================

    do {
        expect(MemoryCheckingEngine.version.version == "1.1.0", "MemoryCheckingEngine version is 1.1.0")
        expect(OfflineMemoryChecker.version.version == "1.1.0", "OfflineMemoryChecker version is 1.1.0")
    }
}
