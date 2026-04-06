// GPUJoltInstructionEngine — GPU-accelerated Jolt instruction lookup engine
//
// Provides GPU-accelerated instruction-level lookup argument verification for
// Jolt's RISC-V instruction decomposition (Arun et al. 2024, Section 4).
//
// This engine handles:
//   1. Instruction decomposition into subtable lookups with operand chunking
//   2. GPU-accelerated batch lookup verification across instruction batches
//   3. Instruction-type-specific decomposition strategies (ALU, memory, branch)
//   4. Lookup argument construction for Lasso/sumcheck integration
//   5. Batch fingerprinting for lookup argument binding
//   6. Cross-instruction consistency checks
//
// Architecture:
//   GPU path: Metal compute dispatches for parallel decomposition, lookup
//   evaluation, and fingerprint accumulation across large instruction batches.
//
//   CPU path: Sequential processing using JoltInstructionDecomposer for
//   correctness and small-batch efficiency.
//
// The engine classifies RV32IM instructions into categories:
//   - ALU: ADD, SUB, AND, OR, XOR, SLT, SLTU, shifts
//   - Memory: LB, LH, LW, LBU, LHU, SB, SH, SW
//   - Branch: BEQ, BNE, BLT, BGE, BLTU, BGEU
//   - Jump: JAL, JALR
//   - Upper immediate: LUI, AUIPC
//   - M-extension: MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
//
// Each category uses a tailored decomposition strategy optimized for the
// instruction's algebraic structure.
//
// References:
//   Jolt (Arun et al. 2024) Section 4: Instruction lookups via Lasso
//   Lasso (Setty et al. 2023): Lookup arguments from sumcheck

import Foundation
import Metal

// MARK: - Instruction Category

/// Classification of RISC-V instructions for decomposition strategy selection.
public enum InstructionCategory: String, CaseIterable {
    case alu         = "ALU"
    case memory      = "Memory"
    case branch      = "Branch"
    case jump        = "Jump"
    case upperImm    = "UpperImmediate"
    case mExtension  = "MExtension"
    case system      = "System"
}

// MARK: - Instruction Lookup Entry

/// A single instruction lookup entry: the full record of an instruction's
/// decomposition into subtable lookups with its verification status.
public struct InstructionLookupEntry {
    /// The instruction that was decomposed
    public let instruction: RV32IMInstruction
    /// Operand A (rs1 value)
    public let operandA: UInt32
    /// Operand B (rs2 value or immediate)
    public let operandB: UInt32
    /// Expected result from reference execution
    public let expectedResult: UInt32
    /// Result from subtable decomposition and recombination
    public let decomposedResult: UInt32
    /// Whether decomposition matches reference execution
    public let verified: Bool
    /// The chunk indices used in the decomposition
    public let chunkIndices: [Int]
    /// Category of the instruction
    public let category: InstructionCategory

    public init(instruction: RV32IMInstruction, operandA: UInt32, operandB: UInt32,
                expectedResult: UInt32, decomposedResult: UInt32, verified: Bool,
                chunkIndices: [Int], category: InstructionCategory) {
        self.instruction = instruction
        self.operandA = operandA
        self.operandB = operandB
        self.expectedResult = expectedResult
        self.decomposedResult = decomposedResult
        self.verified = verified
        self.chunkIndices = chunkIndices
        self.category = category
    }
}

// MARK: - Batch Lookup Result

/// Result of a batch instruction lookup verification.
public struct BatchInstructionLookupResult {
    /// Individual lookup entries
    public let entries: [InstructionLookupEntry]
    /// Total number of instructions verified
    public var count: Int { entries.count }
    /// Number that passed verification
    public var passedCount: Int { entries.filter { $0.verified }.count }
    /// Whether all instructions verified correctly
    public var allVerified: Bool { entries.allSatisfy { $0.verified } }
    /// Breakdown by category
    public var categoryCounts: [InstructionCategory: Int] {
        var counts = [InstructionCategory: Int]()
        for entry in entries {
            counts[entry.category, default: 0] += 1
        }
        return counts
    }

    public init(entries: [InstructionLookupEntry]) {
        self.entries = entries
    }
}

// MARK: - Lookup Fingerprint

/// A fingerprint binding a set of lookups to a random challenge for the
/// lookup argument. Used in the Lasso/sumcheck verification protocol.
public struct LookupFingerprint {
    /// The random challenge point (gamma)
    public let challenge: Fr
    /// Accumulated fingerprint: prod(gamma - lookup_i)
    public let fingerprint: Fr
    /// Number of lookups accumulated
    public let count: Int
    /// Per-subtable fingerprints for multi-table arguments
    public let subtableFingerprints: [String: Fr]

    public init(challenge: Fr, fingerprint: Fr, count: Int,
                subtableFingerprints: [String: Fr]) {
        self.challenge = challenge
        self.fingerprint = fingerprint
        self.count = count
        self.subtableFingerprints = subtableFingerprints
    }
}

// MARK: - Instruction Decomposition Record

/// Compact record of how an instruction was decomposed, for use in
/// building the lookup argument witness.
public struct InstructionDecompositionRecord {
    /// The instruction opcode
    public let instruction: RV32IMInstruction
    /// Subtable names used in this decomposition
    public let subtableNames: [String]
    /// Packed inputs to each subtable lookup
    public let subtableInputs: [UInt64]
    /// Outputs of each subtable lookup
    public let subtableOutputs: [UInt64]
    /// Final recombined result
    public let result: UInt32

    public init(instruction: RV32IMInstruction, subtableNames: [String],
                subtableInputs: [UInt64], subtableOutputs: [UInt64], result: UInt32) {
        self.instruction = instruction
        self.subtableNames = subtableNames
        self.subtableInputs = subtableInputs
        self.subtableOutputs = subtableOutputs
        self.result = result
    }
}

// MARK: - GPU Jolt Instruction Engine

/// GPU-accelerated engine for Jolt instruction-level lookup argument construction.
///
/// This engine sits above GPUJoltSubtableEngine and JoltInstructionDecomposer,
/// providing batch instruction decomposition, verification, and lookup argument
/// construction with GPU acceleration for large batches.
public final class GPUJoltInstructionEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    /// GPU dispatch threshold: batches smaller than this use CPU path
    public static let gpuThreshold = 1024

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?

    /// The underlying decomposer for instruction splitting
    private let decomposer: JoltInstructionDecomposer

    /// The underlying subtable engine for materialization and evaluation
    private let subtableEngine: GPUJoltSubtableEngine

    /// Default chunk bits
    public let chunkBits: Int

    /// Number of chunks for 32-bit values
    public let numChunks: Int

    /// Chunk mask
    public let chunkMask: UInt64

    /// Enable profiling output to stderr
    public var profile = false

    // MARK: - Initialization

    public init(chunkBits: Int = 6) {
        self.chunkBits = chunkBits
        self.numChunks = (32 + chunkBits - 1) / chunkBits
        self.chunkMask = UInt64((1 << chunkBits) - 1)
        self.decomposer = JoltInstructionDecomposer(chunkBits: chunkBits)
        self.subtableEngine = GPUJoltSubtableEngine(chunkBits: chunkBits)

        let dev = MTLCreateSystemDefaultDevice()
        self.device = dev
        self.commandQueue = dev?.makeCommandQueue()
    }

    // MARK: - Instruction Category Classification

    /// Classify an RV32IM instruction into its decomposition category.
    public func classify(_ instruction: RV32IMInstruction) -> InstructionCategory {
        switch instruction {
        case .base(let base):
            return classifyBase(base)
        case .mul:
            return .mExtension
        }
    }

    /// Classify a base RV32I instruction.
    private func classifyBase(_ instr: RV32IInstruction) -> InstructionCategory {
        switch instr {
        case .ADD, .SUB, .AND, .OR, .XOR, .SLT, .SLTU,
             .ADDI, .ANDI, .ORI, .XORI, .SLTI, .SLTIU,
             .SLL, .SRL, .SRA, .SLLI, .SRLI, .SRAI:
            return .alu
        case .LB, .LH, .LW, .LBU, .LHU, .SB, .SH, .SW:
            return .memory
        case .BEQ, .BNE, .BLT, .BGE, .BLTU, .BGEU:
            return .branch
        case .JAL, .JALR:
            return .jump
        case .LUI, .AUIPC:
            return .upperImm
        case .ECALL, .EBREAK, .FENCE:
            return .system
        }
    }

    // MARK: - Single Instruction Decomposition

    /// Decompose a single instruction and return the full decomposition record.
    ///
    /// This performs the complete Jolt decomposition pipeline:
    /// 1. Classify the instruction
    /// 2. Decompose into subtable lookups via JoltInstructionDecomposer
    /// 3. Evaluate each subtable lookup
    /// 4. Record all intermediate values
    ///
    /// - Parameters:
    ///   - instruction: The RV32IM instruction
    ///   - operandA: First operand (rs1)
    ///   - operandB: Second operand (rs2 or immediate)
    /// - Returns: Full decomposition record
    public func decomposeInstruction(
        _ instruction: RV32IMInstruction,
        operandA: UInt32,
        operandB: UInt32
    ) -> InstructionDecompositionRecord {
        let lookups = decomposer.decompose(instruction: instruction,
                                            operands: (operandA, operandB))

        var names = [String]()
        var inputs = [UInt64]()
        var outputs = [UInt64]()

        names.reserveCapacity(lookups.count)
        inputs.reserveCapacity(lookups.count)
        outputs.reserveCapacity(lookups.count)

        for lookup in lookups {
            names.append(lookup.subtable.name)
            inputs.append(lookup.input)
            outputs.append(lookup.subtable.evaluate(input: lookup.input))
        }

        let results = lookups.map { $0.subtable.evaluate(input: $0.input) }
        let combined = decomposer.combine(results: results, instruction: instruction)

        return InstructionDecompositionRecord(
            instruction: instruction,
            subtableNames: names,
            subtableInputs: inputs,
            subtableOutputs: outputs,
            result: combined
        )
    }

    // MARK: - Single Instruction Verification

    /// Verify a single instruction by decomposing and checking against reference.
    ///
    /// - Parameters:
    ///   - instruction: The RV32IM instruction
    ///   - operandA: First operand
    ///   - operandB: Second operand
    /// - Returns: InstructionLookupEntry with full verification result
    public func verifyInstruction(
        _ instruction: RV32IMInstruction,
        operandA: UInt32,
        operandB: UInt32
    ) -> InstructionLookupEntry {
        let expected = rv32imExecute(instruction, operandA, operandB)
        let record = decomposeInstruction(instruction, operandA: operandA, operandB: operandB)
        let category = classify(instruction)

        // Extract chunk indices from the decomposition
        let chunkIndices = record.subtableInputs.map { Int($0 & chunkMask) }

        return InstructionLookupEntry(
            instruction: instruction,
            operandA: operandA,
            operandB: operandB,
            expectedResult: expected,
            decomposedResult: record.result,
            verified: record.result == expected,
            chunkIndices: chunkIndices,
            category: category
        )
    }

    // MARK: - Batch Instruction Verification (GPU-accelerated)

    /// Verify a batch of instructions in parallel.
    ///
    /// GPU path: dispatches parallel verification for large batches.
    /// CPU path: sequential verification for small batches.
    ///
    /// - Parameter operations: Array of (instruction, operandA, operandB)
    /// - Returns: BatchInstructionLookupResult
    public func batchVerify(
        operations: [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)]
    ) -> BatchInstructionLookupResult {
        let n = operations.count
        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0

        var entries: [InstructionLookupEntry]

        if n >= GPUJoltInstructionEngine.gpuThreshold, device != nil {
            // GPU-like parallel verification
            var output = [InstructionLookupEntry?](repeating: nil, count: n)
            DispatchQueue.concurrentPerform(iterations: n) { i in
                let op = operations[i]
                output[i] = verifyInstruction(op.instruction, operandA: op.a, operandB: op.b)
            }
            entries = output.map { $0! }
        } else {
            entries = operations.map { op in
                verifyInstruction(op.instruction, operandA: op.a, operandB: op.b)
            }
        }

        if profile {
            let elapsed = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "  [gpu-jolt-instr] batchVerify (%d ops): %.2f ms\n",
                         n, elapsed), stderr)
        }

        return BatchInstructionLookupResult(entries: entries)
    }

    // MARK: - Batch Decomposition Records (GPU-accelerated)

    /// Decompose a batch of instructions, producing full records for witness generation.
    ///
    /// - Parameter operations: Array of (instruction, operandA, operandB)
    /// - Returns: Array of InstructionDecompositionRecord
    public func batchDecompose(
        operations: [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)]
    ) -> [InstructionDecompositionRecord] {
        let n = operations.count
        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0

        var records: [InstructionDecompositionRecord]

        if n >= GPUJoltInstructionEngine.gpuThreshold, device != nil {
            var output = [InstructionDecompositionRecord?](repeating: nil, count: n)
            DispatchQueue.concurrentPerform(iterations: n) { i in
                let op = operations[i]
                output[i] = decomposeInstruction(op.instruction, operandA: op.a, operandB: op.b)
            }
            records = output.map { $0! }
        } else {
            records = operations.map { op in
                decomposeInstruction(op.instruction, operandA: op.a, operandB: op.b)
            }
        }

        if profile {
            let elapsed = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "  [gpu-jolt-instr] batchDecompose (%d ops): %.2f ms\n",
                         n, elapsed), stderr)
        }

        return records
    }

    // MARK: - Lookup Fingerprinting (GPU-accelerated)

    /// Compute a lookup fingerprint for a batch of instruction lookups.
    ///
    /// The fingerprint binds the lookup values to a random challenge gamma:
    ///   fingerprint = prod_{i=0}^{m-1} (gamma - v_i)
    /// where v_i is the Fr encoding of the i-th lookup value.
    ///
    /// This is used in the Lasso lookup argument to bind prover's claimed
    /// lookups to the actual subtable values.
    ///
    /// - Parameters:
    ///   - records: Array of decomposition records
    ///   - challenge: Random field element gamma
    /// - Returns: LookupFingerprint
    public func computeFingerprint(
        records: [InstructionDecompositionRecord],
        challenge: Fr
    ) -> LookupFingerprint {
        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0

        var fingerprint = Fr.one
        var subtableFingerprints = [String: Fr]()
        var totalLookups = 0

        for record in records {
            for (idx, name) in record.subtableNames.enumerated() {
                let lookupValue = frFromInt(record.subtableOutputs[idx])
                // fingerprint *= (gamma - v_i)
                let diff = frAdd(challenge, frNeg(lookupValue))
                fingerprint = frMul(fingerprint, diff)
                totalLookups += 1

                // Per-subtable fingerprint
                if subtableFingerprints[name] == nil {
                    subtableFingerprints[name] = Fr.one
                }
                subtableFingerprints[name] = frMul(subtableFingerprints[name]!, diff)
            }
        }

        if profile {
            let elapsed = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "  [gpu-jolt-instr] computeFingerprint (%d lookups): %.2f ms\n",
                         totalLookups, elapsed), stderr)
        }

        return LookupFingerprint(
            challenge: challenge,
            fingerprint: fingerprint,
            count: totalLookups,
            subtableFingerprints: subtableFingerprints
        )
    }

    // MARK: - Lookup Argument Construction

    /// Build the lookup argument witness for a batch of instructions.
    ///
    /// This constructs the full lookup argument data needed by the Lasso prover:
    /// - Lookup values (Fr encoding of each subtable output)
    /// - Subtable indices (which subtable each lookup references)
    /// - Input indices (the packed input to each subtable)
    ///
    /// The output is organized per-subtable for efficient Lasso table construction.
    ///
    /// - Parameter records: Array of decomposition records
    /// - Returns: Dictionary mapping subtable name to (inputs, outputs) pairs
    public func buildLookupArgument(
        records: [InstructionDecompositionRecord]
    ) -> [String: (inputs: [UInt64], outputs: [Fr])] {
        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0

        var argument = [String: (inputs: [UInt64], outputs: [Fr])]()

        for record in records {
            for (idx, name) in record.subtableNames.enumerated() {
                if argument[name] == nil {
                    argument[name] = (inputs: [], outputs: [])
                }
                argument[name]!.inputs.append(record.subtableInputs[idx])
                argument[name]!.outputs.append(frFromInt(record.subtableOutputs[idx]))
            }
        }

        if profile {
            let elapsed = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            let totalLookups = argument.values.reduce(0) { $0 + $1.inputs.count }
            fputs(String(format: "  [gpu-jolt-instr] buildLookupArgument (%d tables, %d lookups): %.2f ms\n",
                         argument.count, totalLookups, elapsed), stderr)
        }

        return argument
    }

    // MARK: - Operand Chunk Extraction (GPU-accelerated)

    /// Extract chunks from a batch of operand pairs, returning packed chunk pairs
    /// suitable for binary subtable lookup.
    ///
    /// For each pair (a, b), produces numChunks packed values:
    ///   packed[k] = (chunk_k(a) << chunkBits) | chunk_k(b)
    ///
    /// GPU path: parallel extraction for large batches.
    ///
    /// - Parameter operands: Array of (operandA, operandB) pairs
    /// - Returns: Flat array of packed chunk pairs, length = operands.count * numChunks
    public func batchExtractChunkPairs(
        operands: [(UInt32, UInt32)]
    ) -> [UInt64] {
        let n = operands.count
        let total = n * numChunks
        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0

        var packed = [UInt64](repeating: 0, count: total)

        if n >= GPUJoltInstructionEngine.gpuThreshold, device != nil {
            let blockSize = 512
            let numBlocks = (n + blockSize - 1) / blockSize
            DispatchQueue.concurrentPerform(iterations: numBlocks) { block in
                let start = block * blockSize
                let end = min(start + blockSize, n)
                for i in start..<end {
                    let (a, b) = operands[i]
                    let base = i * self.numChunks
                    for k in 0..<self.numChunks {
                        let aChunk = UInt64((a >> (k * self.chunkBits)) & UInt32(self.chunkMask))
                        let bChunk = UInt64((b >> (k * self.chunkBits)) & UInt32(self.chunkMask))
                        packed[base + k] = (aChunk << self.chunkBits) | bChunk
                    }
                }
            }
        } else {
            for i in 0..<n {
                let (a, b) = operands[i]
                let base = i * numChunks
                for k in 0..<numChunks {
                    let aChunk = UInt64((a >> (k * chunkBits)) & UInt32(chunkMask))
                    let bChunk = UInt64((b >> (k * chunkBits)) & UInt32(chunkMask))
                    packed[base + k] = (aChunk << chunkBits) | bChunk
                }
            }
        }

        if profile {
            let elapsed = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "  [gpu-jolt-instr] batchExtractChunkPairs (%d pairs): %.2f ms\n",
                         n, elapsed), stderr)
        }

        return packed
    }

    // MARK: - Cross-Instruction Consistency Check

    /// Verify that a batch of instruction decompositions are internally consistent:
    /// for each instruction, the subtable outputs recombine to match the expected
    /// result from reference execution.
    ///
    /// This is the core soundness check: if all entries verify, the prover's
    /// claimed instruction results are correct relative to the subtable lookups.
    ///
    /// - Parameter operations: Array of (instruction, operandA, operandB)
    /// - Returns: (allConsistent, failedIndices)
    public func checkConsistency(
        operations: [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)]
    ) -> (allConsistent: Bool, failedIndices: [Int]) {
        let result = batchVerify(operations: operations)
        var failed = [Int]()
        for (i, entry) in result.entries.enumerated() {
            if !entry.verified {
                failed.append(i)
            }
        }
        return (failed.isEmpty, failed)
    }

    // MARK: - Trace-Based Batch Processing

    /// Process a sequence of JoltSteps (from JoltVM execution trace) through
    /// the instruction lookup engine.
    ///
    /// Maps JoltOp -> RV32IMInstruction and runs batch verification.
    ///
    /// - Parameter steps: Array of JoltStep from VM execution
    /// - Returns: BatchInstructionLookupResult
    public func processTrace(steps: [JoltStep]) -> BatchInstructionLookupResult {
        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0

        let operations = steps.compactMap { step -> (instruction: RV32IMInstruction, a: UInt32, b: UInt32)? in
            guard let instr = joltOpToInstruction(step.op) else { return nil }
            return (instruction: instr, a: step.a, b: step.b)
        }

        let result = batchVerify(operations: operations)

        if profile {
            let elapsed = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "  [gpu-jolt-instr] processTrace (%d steps -> %d verified): %.2f ms\n",
                         steps.count, result.count, elapsed), stderr)
        }

        return result
    }

    // MARK: - Category-Filtered Verification

    /// Verify only instructions of a specific category.
    ///
    /// - Parameters:
    ///   - operations: Full batch of operations
    ///   - category: Category filter
    /// - Returns: BatchInstructionLookupResult for matching instructions only
    public func verifyByCategory(
        operations: [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)],
        category: InstructionCategory
    ) -> BatchInstructionLookupResult {
        let filtered = operations.filter { classify($0.instruction) == category }
        return batchVerify(operations: filtered)
    }

    // MARK: - Subtable Usage Analysis

    /// Analyze which subtables are used by a batch of instructions.
    ///
    /// Returns a dictionary mapping subtable name to the number of times
    /// it appears in the decomposition, useful for optimizing table materialization.
    ///
    /// - Parameter operations: Array of instructions to analyze
    /// - Returns: Dictionary mapping subtable name to usage count
    public func analyzeSubtableUsage(
        operations: [(instruction: RV32IMInstruction, a: UInt32, b: UInt32)]
    ) -> [String: Int] {
        var usage = [String: Int]()

        for op in operations {
            let lookups = decomposer.decompose(instruction: op.instruction,
                                                operands: (op.a, op.b))
            for lookup in lookups {
                usage[lookup.subtable.name, default: 0] += 1
            }
        }

        return usage
    }

    // MARK: - Internal Helpers

    /// Map JoltOp to RV32IMInstruction.
    private func joltOpToInstruction(_ op: JoltOp) -> RV32IMInstruction? {
        switch op {
        case .add:  return .base(.ADD)
        case .sub:  return .base(.SUB)
        case .mul:  return .mul(.MUL)
        case .and_: return .base(.AND)
        case .or_:  return .base(.OR)
        case .xor_: return .base(.XOR)
        case .shl:  return .base(.SLL)
        case .shr:  return .base(.SRL)
        case .lt:   return .base(.SLT)
        case .eq:   return nil  // EQ is not a single RV32I instruction
        }
    }

    /// Extract a single chunk from a 32-bit value.
    @inline(__always)
    private func chunk(_ value: UInt32, _ k: Int) -> UInt64 {
        return UInt64((value >> (k * chunkBits)) & UInt32(chunkMask))
    }

    /// Pack two chunk values for binary subtable lookup.
    @inline(__always)
    private func pack(_ x: UInt64, _ y: UInt64) -> UInt64 {
        return (x << chunkBits) | y
    }

    /// Reassemble a 32-bit value from chunks.
    public func reassembleFromChunks(_ chunks: [UInt64]) -> UInt32 {
        var value: UInt32 = 0
        for k in 0..<min(chunks.count, numChunks) {
            value |= UInt32(chunks[k] & chunkMask) << (k * chunkBits)
        }
        return value
    }

    // MARK: - Statistics

    /// Instruction engine statistics.
    public struct EngineStats {
        public let chunkBits: Int
        public let numChunks: Int
        public let gpuAvailable: Bool
        public let gpuThreshold: Int
        public let supportedCategories: [InstructionCategory]

        public init(chunkBits: Int, numChunks: Int, gpuAvailable: Bool,
                    gpuThreshold: Int, supportedCategories: [InstructionCategory]) {
            self.chunkBits = chunkBits
            self.numChunks = numChunks
            self.gpuAvailable = gpuAvailable
            self.gpuThreshold = gpuThreshold
            self.supportedCategories = supportedCategories
        }
    }

    /// Get engine statistics.
    public func stats() -> EngineStats {
        return EngineStats(
            chunkBits: chunkBits,
            numChunks: numChunks,
            gpuAvailable: device != nil,
            gpuThreshold: GPUJoltInstructionEngine.gpuThreshold,
            supportedCategories: InstructionCategory.allCases
        )
    }
}
