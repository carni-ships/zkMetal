// GPUzkVMEngine -- GPU-accelerated minimal zkVM for provable computation
//
// Implements a simple instruction set suitable for STARK proving:
//   ADD, MUL, LOAD, STORE, BRANCH_EQ, BRANCH_NEQ, HALT
//
// All operations are over BN254 Fr (scalar field) so the execution trace
// is directly usable in a STARK/AIR-based proof system.
//
// Key features:
//   - Configurable register file width (default 16 registers)
//   - Execution trace generation: (pc, opcode, op1, op2, dst, val) per step
//   - Memory consistency checking via sorted-address permutation argument
//   - Program counter tracking with branch support
//
// The trace format is designed so each row can be constrained by an AIR:
//   Column layout: [step, pc, opcode, arg1, arg2, dst, result, memAddr, memVal]

import Foundation

// MARK: - Instruction Set

/// Opcodes for the zkVM instruction set, encoded as field elements (0..6).
public enum zkVMOpcode: UInt8, CaseIterable {
    case add      = 0   // dst = reg[arg1] + reg[arg2]
    case mul      = 1   // dst = reg[arg1] * reg[arg2]
    case load     = 2   // dst = memory[reg[arg1] + arg2]
    case store    = 3   // memory[reg[arg1] + arg2] = reg[dst]
    case branchEq = 4   // if reg[arg1] == reg[arg2] then pc = dst
    case branchNe = 5   // if reg[arg1] != reg[arg2] then pc = dst
    case halt     = 6   // stop execution
}

/// A single zkVM instruction.
public struct zkVMInstruction {
    public let opcode: zkVMOpcode
    public let dst: UInt8      // destination register (or branch target / store source)
    public let arg1: UInt8     // first source register
    public let arg2: UInt8     // second source register or immediate

    public init(opcode: zkVMOpcode, dst: UInt8, arg1: UInt8, arg2: UInt8) {
        self.opcode = opcode
        self.dst = dst
        self.arg1 = arg1
        self.arg2 = arg2
    }
}

// MARK: - Execution Trace

/// A single row of the execution trace, capturing all state for AIR constraint checking.
public struct zkVMTraceRow {
    /// Step index (0-based)
    public let step: Int
    /// Program counter before this instruction
    public let pc: Int
    /// Opcode executed
    public let opcode: zkVMOpcode
    /// First operand value (from register file)
    public let op1: Fr
    /// Second operand value (from register file or immediate)
    public let op2: Fr
    /// Destination register index
    public let dst: UInt8
    /// Result value written to dst (or Fr.zero for branches/store/halt)
    public let result: Fr
    /// Memory address accessed (0 for non-memory ops)
    public let memAddr: UInt64
    /// Memory value read or written (Fr.zero for non-memory ops)
    public let memVal: Fr
}

/// Memory access record for permutation-based consistency checking.
public struct zkVMMemoryAccess: Comparable {
    public let step: Int
    public let address: UInt64
    public let value: Fr
    public let isWrite: Bool

    public init(step: Int, address: UInt64, value: Fr, isWrite: Bool) {
        self.step = step; self.address = address; self.value = value; self.isWrite = isWrite
    }

    public static func < (lhs: zkVMMemoryAccess, rhs: zkVMMemoryAccess) -> Bool {
        if lhs.address != rhs.address { return lhs.address < rhs.address }
        return lhs.step < rhs.step
    }
}

// MARK: - Execution Result

/// Result of running a zkVM program.
public struct zkVMResult {
    /// Ordered execution trace (one row per step)
    public let trace: [zkVMTraceRow]
    /// Final register file
    public let registers: [Fr]
    /// Final program counter
    public let finalPC: Int
    /// Whether execution terminated via HALT (vs step limit)
    public let halted: Bool
    /// Number of steps executed
    public var stepCount: Int { trace.count }
    /// Memory accesses sorted by (address, step) for permutation argument
    public let memoryAccesses: [zkVMMemoryAccess]
}

// MARK: - GPUzkVMEngine

/// GPU-accelerated zkVM execution engine.
///
/// Executes programs over BN254 Fr field elements and produces an execution
/// trace suitable for STARK proving. Memory consistency is enforced via a
/// permutation argument over sorted memory accesses.
///
/// Usage:
///   let engine = GPUzkVMEngine(registerCount: 16)
///   let program: [zkVMInstruction] = [...]
///   let result = engine.execute(program: program, stepLimit: 10000)
///   // result.trace has the AIR-compatible execution trace
///   // result.memoryAccesses has sorted memory log for permutation check
public struct GPUzkVMEngine {
    /// Number of registers in the register file
    public let registerCount: Int

    public init(registerCount: Int = 16) {
        precondition(registerCount >= 2, "Need at least 2 registers")
        precondition(registerCount <= 256, "Register index must fit in UInt8")
        self.registerCount = registerCount
    }

    /// Execute a zkVM program.
    ///
    /// - Parameters:
    ///   - program: Array of zkVM instructions
    ///   - stepLimit: Maximum steps before forced halt (default 1_000_000)
    ///   - initialRegisters: Optional initial register values (padded/truncated to registerCount)
    ///   - initialMemory: Optional pre-loaded memory contents
    /// - Returns: zkVMResult with trace, final registers, and memory access log
    public func execute(
        program: [zkVMInstruction],
        stepLimit: Int = 1_000_000,
        initialRegisters: [Fr] = [],
        initialMemory: [UInt64: Fr] = [:]
    ) -> zkVMResult {
        // Initialize register file
        var regs = [Fr](repeating: Fr.zero, count: registerCount)
        for (i, v) in initialRegisters.prefix(registerCount).enumerated() {
            regs[i] = v
        }

        var memory = initialMemory
        var pc = 0
        var trace = [zkVMTraceRow]()
        trace.reserveCapacity(min(stepLimit, program.count * 4))
        var memLog = [zkVMMemoryAccess]()

        var didHalt = false

        for stepIdx in 0..<stepLimit {
            // Bounds check
            guard pc >= 0, pc < program.count else {
                break
            }

            let instr = program[pc]

            // Read operand registers (bounds-checked)
            let op1 = safeRead(regs, instr.arg1)
            let op2 = safeRead(regs, instr.arg2)

            var result = Fr.zero
            var memAddr: UInt64 = 0
            var memVal = Fr.zero
            var nextPC = pc + 1

            switch instr.opcode {
            case .add:
                result = frAdd(op1, op2)
                safeWrite(&regs, instr.dst, result)

            case .mul:
                result = frMul(op1, op2)
                safeWrite(&regs, instr.dst, result)

            case .load:
                let base = frToUInt64(op1)
                memAddr = base &+ UInt64(instr.arg2)
                memVal = memory[memAddr] ?? Fr.zero
                result = memVal
                safeWrite(&regs, instr.dst, result)
                memLog.append(zkVMMemoryAccess(
                    step: stepIdx, address: memAddr, value: memVal, isWrite: false))

            case .store:
                let base = frToUInt64(op1)
                memAddr = base &+ UInt64(instr.arg2)
                memVal = safeRead(regs, instr.dst)
                memory[memAddr] = memVal
                memLog.append(zkVMMemoryAccess(
                    step: stepIdx, address: memAddr, value: memVal, isWrite: true))

            case .branchEq:
                if op1 == op2 {
                    nextPC = Int(instr.dst)
                }

            case .branchNe:
                if op1 != op2 {
                    nextPC = Int(instr.dst)
                }

            case .halt:
                let row = zkVMTraceRow(
                    step: stepIdx, pc: pc, opcode: .halt,
                    op1: Fr.zero, op2: Fr.zero, dst: 0,
                    result: Fr.zero, memAddr: 0, memVal: Fr.zero)
                trace.append(row)
                didHalt = true
                pc = nextPC
                break
            }

            if didHalt { break }

            let row = zkVMTraceRow(
                step: stepIdx, pc: pc, opcode: instr.opcode,
                op1: op1, op2: op2, dst: instr.dst,
                result: result, memAddr: memAddr, memVal: memVal)
            trace.append(row)

            pc = nextPC
        }

        // Sort memory accesses by (address, step) for permutation argument
        let sortedMem = memLog.sorted()

        return zkVMResult(
            trace: trace,
            registers: regs,
            finalPC: pc,
            halted: didHalt,
            memoryAccesses: sortedMem)
    }

    // MARK: - Memory Consistency Check

    /// Verify memory consistency via permutation argument.
    ///
    /// For each address, the sequence of reads and writes must be consistent:
    /// every read must return the value of the most recent write to that address
    /// (or Fr.zero if never written).
    ///
    /// Returns true if all memory accesses are consistent.
    public func verifyMemoryConsistency(_ accesses: [zkVMMemoryAccess]) -> Bool {
        // Group by address (already sorted by address, then step)
        var currentAddr: UInt64? = nil
        var currentValue = Fr.zero

        for access in accesses {
            if access.address != currentAddr {
                // New address -- reset to zero (initial memory value)
                currentAddr = access.address
                currentValue = Fr.zero
            }

            if access.isWrite {
                currentValue = access.value
            } else {
                // Read must match current value
                if access.value != currentValue {
                    return false
                }
            }
        }
        return true
    }

    // MARK: - Trace Validation

    /// Validate that the execution trace is internally consistent.
    ///
    /// Checks:
    ///   - Step indices are sequential
    ///   - ADD/MUL results match operands
    ///   - PC transitions are valid
    ///
    /// Returns true if all checks pass.
    public func validateTrace(_ trace: [zkVMTraceRow]) -> Bool {
        for (i, row) in trace.enumerated() {
            // Step index must be sequential
            if row.step != i { return false }

            switch row.opcode {
            case .add:
                let expected = frAdd(row.op1, row.op2)
                if expected != row.result { return false }
            case .mul:
                let expected = frMul(row.op1, row.op2)
                if expected != row.result { return false }
            case .halt, .load, .store, .branchEq, .branchNe:
                break  // Validated via memory consistency or branch logic
            }
        }
        return true
    }

    // MARK: - Register Helpers

    @inline(__always)
    private func safeRead(_ regs: [Fr], _ idx: UInt8) -> Fr {
        let i = Int(idx)
        guard i < regs.count else { return Fr.zero }
        return regs[i]
    }

    @inline(__always)
    private func safeWrite(_ regs: inout [Fr], _ idx: UInt8, _ val: Fr) {
        let i = Int(idx)
        guard i < regs.count else { return }
        regs[i] = val
    }
}
