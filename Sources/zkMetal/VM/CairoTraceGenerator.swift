// CairoTraceGenerator -- Cairo VM execution engine with full trace capture
//
// Implements the Cairo VM instruction set and produces execution traces
// suitable for STARK proving via AIR constraints. The VM uses Stark252
// field elements as its native word size (matching StarkNet's field).
//
// Cairo VM architecture:
//   - 3 registers: pc (program counter), ap (allocation pointer), fp (frame pointer)
//   - Word-addressable sparse memory over Stark252 field elements
//   - Instructions encoded as single field elements with flag decomposition
//   - Supports: assert_eq, call, ret, jmp, jnz, add_ap
//
// Reference: Cairo whitepaper (https://eprint.iacr.org/2021/1063)

import Foundation

// MARK: - Cairo Instruction Flags

/// Source for the second operand (op1).
public enum CairoOp1Src: UInt8 {
    case op0     = 0  // op1 = mem[op0 + off_op1]
    case pc      = 1  // op1 = mem[pc + off_op1]  (immediate value)
    case fp      = 2  // op1 = mem[fp + off_op1]
    case ap      = 4  // op1 = mem[ap + off_op1]
}

/// How the result is computed from op0 and op1.
public enum CairoResLogic: UInt8 {
    case op1     = 0  // res = op1
    case add     = 1  // res = op0 + op1
    case mul     = 2  // res = op0 * op1
}

/// How the PC is updated after the instruction.
public enum CairoPCUpdate: UInt8 {
    case regular = 0  // pc += instruction_size
    case jump    = 1  // pc = res (absolute jump)
    case jumpRel = 2  // pc = pc + res (relative jump)
    case jnz     = 4  // if dst != 0: pc += op1, else pc += instruction_size
}

/// How the AP is updated after the instruction.
public enum CairoAPUpdate: UInt8 {
    case regular = 0  // ap unchanged
    case add     = 1  // ap += res
    case addOne  = 2  // ap += 1
    case addTwo  = 4  // ap += 2
}

/// The opcode type (high-level instruction category).
public enum CairoOpcodeType: UInt8 {
    case jmp      = 0  // jump/jnz (no assertion, no call)
    case call     = 1  // push return address and fp, jump
    case ret      = 2  // restore fp and jump to return address
    case assertEq = 4  // assert dst == res
}

// MARK: - Cairo Instruction

/// A decoded Cairo instruction with all flag fields and signed offsets.
public struct CairoInstruction {
    /// Signed offset for dst operand (biased by 2^15)
    public let offDst: Int16
    /// Signed offset for op0 operand (biased by 2^15)
    public let offOp0: Int16
    /// Signed offset for op1 operand (biased by 2^15)
    public let offOp1: Int16

    /// Source register for dst: false = ap, true = fp
    public let dstRegIsFP: Bool
    /// Source register for op0: false = ap, true = fp
    public let op0RegIsFP: Bool

    /// Source for op1
    public let op1Src: CairoOp1Src
    /// Result computation logic
    public let resLogic: CairoResLogic
    /// PC update mode
    public let pcUpdate: CairoPCUpdate
    /// AP update mode
    public let apUpdate: CairoAPUpdate
    /// Opcode type
    public let opcodeType: CairoOpcodeType

    /// The instruction size (1 for normal, 2 if op1 source is immediate/pc)
    public var size: Int {
        return op1Src == .pc ? 2 : 1
    }

    public init(offDst: Int16, offOp0: Int16, offOp1: Int16,
                dstRegIsFP: Bool, op0RegIsFP: Bool,
                op1Src: CairoOp1Src, resLogic: CairoResLogic,
                pcUpdate: CairoPCUpdate, apUpdate: CairoAPUpdate,
                opcodeType: CairoOpcodeType) {
        self.offDst = offDst
        self.offOp0 = offOp0
        self.offOp1 = offOp1
        self.dstRegIsFP = dstRegIsFP
        self.op0RegIsFP = op0RegIsFP
        self.op1Src = op1Src
        self.resLogic = resLogic
        self.pcUpdate = pcUpdate
        self.apUpdate = apUpdate
        self.opcodeType = opcodeType
    }
}

// MARK: - Cairo State

/// The three Cairo VM registers at a given point in execution.
public struct CairoState {
    public var pc: UInt64
    public var ap: UInt64
    public var fp: UInt64

    public init(pc: UInt64, ap: UInt64, fp: UInt64) {
        self.pc = pc
        self.ap = ap
        self.fp = fp
    }
}

// MARK: - Cairo Trace Step

/// A single step in the Cairo execution trace, capturing all values
/// needed for AIR constraint checking and STARK proving.
public struct CairoTraceStep {
    /// Register state before this step
    public let state: CairoState
    /// The decoded instruction
    public let instruction: CairoInstruction

    /// Memory addresses and values for the three operands
    public let dstAddr: UInt64
    public let dstVal: Stark252
    public let op0Addr: UInt64
    public let op0Val: Stark252
    public let op1Addr: UInt64
    public let op1Val: Stark252

    /// Computed result value
    public let res: Stark252

    /// Step index (0-based)
    public let step: Int
}

// MARK: - Cairo Memory

/// Sparse, word-addressable memory over Stark252 field elements.
/// Tracks all reads and writes for the memory checking argument
/// (permutation-based memory consistency proof).
public class CairoMemory {
    /// The actual memory contents
    private var cells: [UInt64: Stark252] = [:]

    /// Ordered log of (address, value) for every access (read or write)
    public private(set) var accessLog: [(addr: UInt64, val: Stark252)] = []

    /// Set of addresses that have been written to
    public private(set) var writtenAddresses: Set<UInt64> = []

    /// Set of addresses that have been read from
    public private(set) var readAddresses: Set<UInt64> = []

    public init() {}

    /// Initialize memory with a dictionary of address -> value mappings.
    public init(initial: [UInt64: Stark252]) {
        self.cells = initial
        for (addr, _) in initial {
            writtenAddresses.insert(addr)
        }
    }

    /// Write a value to a memory address. Cairo memory is write-once:
    /// writing to an already-written address with a different value is an error.
    @discardableResult
    public func write(_ addr: UInt64, _ val: Stark252) -> Bool {
        if let existing = cells[addr] {
            // Write-once check: value must match
            let existingLimbs = stark252ToInt(existing)
            let newLimbs = stark252ToInt(val)
            if existingLimbs != newLimbs {
                return false
            }
        }
        cells[addr] = val
        writtenAddresses.insert(addr)
        accessLog.append((addr: addr, val: val))
        return true
    }

    /// Read a value from a memory address. Returns zero for uninitialized cells.
    public func read(_ addr: UInt64) -> Stark252 {
        let val = cells[addr] ?? Stark252.zero
        // Record access (mutable state for tracking)
        let mem = self
        mem.readAddresses.insert(addr)
        mem.accessLog.append((addr: addr, val: val))
        return val
    }

    /// Check if an address has been written to.
    public func hasValue(at addr: UInt64) -> Bool {
        return cells[addr] != nil
    }

    /// Get the raw value without logging access (for debugging).
    public func peek(_ addr: UInt64) -> Stark252? {
        return cells[addr]
    }

    /// Total number of distinct addresses with values.
    public var cellCount: Int {
        return cells.count
    }

    /// Total number of memory accesses logged.
    public var accessCount: Int {
        return accessLog.count
    }
}

// MARK: - Cairo VM Error

public enum CairoVMError: Error {
    case memoryWriteConflict(addr: UInt64)
    case invalidInstruction(pc: UInt64)
    case inconsistentOp0(expected: Stark252, got: Stark252)
    case stepLimitReached
    case invalidJnzTarget
}

// MARK: - CairoTraceGenerator

/// Executes Cairo programs step by step and records an execution trace
/// suitable for STARK proving.
///
/// Usage:
///   let gen = CairoTraceGenerator(stepLimit: 10_000)
///   let mem = CairoMemory(initial: [...])
///   let result = try gen.execute(instructions: [...], memory: mem)
///   // result.trace has per-step witness data for AIR
///   // result.memory has full access log for memory argument
public class CairoTraceGenerator {
    /// Maximum steps before forced halt.
    public let stepLimit: Int

    public init(stepLimit: Int = 100_000) {
        self.stepLimit = stepLimit
    }

    /// Execute a Cairo program.
    ///
    /// - Parameters:
    ///   - instructions: Array of decoded CairoInstructions indexed by PC.
    ///     Instructions are stored at addresses 0, 1, 2, ... (word-addressed).
    ///     Immediate values follow their instruction at pc+1.
    ///   - memory: Pre-initialized memory. Instructions are also loaded into memory.
    ///   - initialState: Starting register state. Default: pc=0, ap=programSize, fp=programSize.
    /// - Returns: CairoExecutionResult with the full trace.
    public func execute(
        instructions: [CairoInstruction],
        memory: CairoMemory,
        initialState: CairoState? = nil
    ) throws -> CairoExecutionResult {
        // Load instructions into memory as encoded field elements
        // (The actual encoding is opaque; we store raw for the trace)
        let programSize = UInt64(countProgramWords(instructions))

        var state: CairoState
        if let s = initialState {
            state = s
        } else {
            state = CairoState(pc: 0, ap: programSize, fp: programSize)
        }

        var trace: [CairoTraceStep] = []
        trace.reserveCapacity(min(stepLimit, instructions.count * 4))

        var halted = false

        for stepIdx in 0..<stepLimit {
            let pc = state.pc

            // Look up instruction at current PC
            guard let instrIndex = findInstructionIndex(pc: pc, instructions: instructions) else {
                halted = true
                break
            }
            let instr = instructions[instrIndex]

            // Compute dst address (safe: offsets can be negative)
            let dstBase: UInt64 = instr.dstRegIsFP ? state.fp : state.ap
            let dstAddr = UInt64(bitPattern: Int64(bitPattern: dstBase) &+ Int64(instr.offDst))

            // Compute op0 address
            let op0Base: UInt64 = instr.op0RegIsFP ? state.fp : state.ap
            let op0Addr = UInt64(bitPattern: Int64(bitPattern: op0Base) &+ Int64(instr.offOp0))

            // Compute op1 address
            let op1Addr: UInt64
            switch instr.op1Src {
            case .op0:
                let op0Val = memory.read(op0Addr)
                let op0Raw = stark252ToUInt64(op0Val)
                op1Addr = UInt64(bitPattern: Int64(bitPattern: op0Raw) &+ Int64(instr.offOp1))
            case .pc:
                op1Addr = UInt64(bitPattern: Int64(bitPattern: pc) &+ Int64(instr.offOp1))
            case .fp:
                op1Addr = UInt64(bitPattern: Int64(bitPattern: state.fp) &+ Int64(instr.offOp1))
            case .ap:
                op1Addr = UInt64(bitPattern: Int64(bitPattern: state.ap) &+ Int64(instr.offOp1))
            }

            // Read operand values
            let op0Val = memory.read(op0Addr)
            let op1Val = memory.read(op1Addr)
            let dstVal: Stark252

            // Compute result
            let res: Stark252
            switch instr.resLogic {
            case .op1:
                res = op1Val
            case .add:
                res = stark252Add(op0Val, op1Val)
            case .mul:
                res = stark252Mul(op0Val, op1Val)
            }

            // Execute opcode
            switch instr.opcodeType {
            case .assertEq:
                // dst = res (write or verify)
                if memory.hasValue(at: dstAddr) {
                    dstVal = memory.read(dstAddr)
                    // Verify consistency
                    let dstLimbs = stark252ToInt(dstVal)
                    let resLimbs = stark252ToInt(res)
                    if dstLimbs != resLimbs {
                        throw CairoVMError.inconsistentOp0(expected: res, got: dstVal)
                    }
                } else {
                    dstVal = res
                    let ok = memory.write(dstAddr, res)
                    if !ok {
                        throw CairoVMError.memoryWriteConflict(addr: dstAddr)
                    }
                }

            case .call:
                // Write [ap] = current fp (saved frame pointer)
                // Write [ap+1] = return address (pc + instruction_size)
                let fpVal = stark252FromInt(state.fp)
                let retAddr = stark252FromInt(pc + UInt64(instr.size))
                let ok1 = memory.write(state.ap, fpVal)
                let ok2 = memory.write(state.ap + 1, retAddr)
                if !ok1 || !ok2 {
                    throw CairoVMError.memoryWriteConflict(addr: state.ap)
                }
                dstVal = memory.read(dstAddr)

            case .ret:
                dstVal = memory.read(dstAddr)

            case .jmp:
                dstVal = memory.read(dstAddr)
            }

            // Record trace step
            let step = CairoTraceStep(
                state: state,
                instruction: instr,
                dstAddr: dstAddr,
                dstVal: dstVal,
                op0Addr: op0Addr,
                op0Val: op0Val,
                op1Addr: op1Addr,
                op1Val: op1Val,
                res: res,
                step: stepIdx
            )
            trace.append(step)

            // Update PC
            switch instr.pcUpdate {
            case .regular:
                state.pc = pc + UInt64(instr.size)
            case .jump:
                let resRaw = stark252ToUInt64(res)
                state.pc = resRaw
            case .jumpRel:
                let resRaw = stark252ToUInt64(res)
                state.pc = pc + resRaw
            case .jnz:
                let dstLimbs = stark252ToInt(dstVal)
                let isZero = dstLimbs[0] == 0 && dstLimbs[1] == 0 &&
                             dstLimbs[2] == 0 && dstLimbs[3] == 0
                if !isZero {
                    let op1Raw = stark252ToUInt64(op1Val)
                    state.pc = pc + op1Raw
                } else {
                    state.pc = pc + UInt64(instr.size)
                }
            }

            // Update FP (must happen before AP update since call uses old ap)
            switch instr.opcodeType {
            case .call:
                // fp = old_ap + 2 (pointing to the new frame)
                state.fp = state.ap + 2
            case .ret:
                let dstRaw = stark252ToUInt64(dstVal)
                state.fp = dstRaw
            case .assertEq, .jmp:
                break
            }

            // Update AP
            switch instr.apUpdate {
            case .regular:
                break
            case .add:
                let resRaw = stark252ToUInt64(res)
                state.ap = state.ap + resRaw
            case .addOne:
                state.ap = state.ap + 1
            case .addTwo:
                state.ap = state.ap + 2
            }
        }

        return CairoExecutionResult(
            trace: trace,
            finalState: state,
            memory: memory,
            halted: halted,
            hitStepLimit: !halted && trace.count >= stepLimit
        )
    }

    // MARK: - Helpers

    /// Count total memory words occupied by the instruction list
    /// (instructions with immediate values occupy 2 words).
    private func countProgramWords(_ instructions: [CairoInstruction]) -> Int {
        var count = 0
        for instr in instructions {
            count += instr.size
        }
        return count
    }

    /// Find the instruction index for a given PC, accounting for
    /// variable-size instructions (size 1 or 2).
    private func findInstructionIndex(pc: UInt64, instructions: [CairoInstruction]) -> Int? {
        var addr: UInt64 = 0
        for (i, instr) in instructions.enumerated() {
            if addr == pc { return i }
            addr += UInt64(instr.size)
        }
        return nil
    }
}

// MARK: - Execution Result

/// Result of executing a Cairo program.
public struct CairoExecutionResult {
    /// Ordered trace steps, one per executed instruction
    public let trace: [CairoTraceStep]

    /// Final register state
    public let finalState: CairoState

    /// Memory with full access log
    public let memory: CairoMemory

    /// Whether execution terminated normally (PC fell off program)
    public let halted: Bool

    /// Whether execution was cut short by the step limit
    public let hitStepLimit: Bool

    /// Number of steps executed
    public var stepCount: Int { trace.count }
}

// MARK: - Convenience Instruction Builders

/// Build an assert_eq instruction: mem[ap/fp + offDst] = op0 {+,*} op1
public func cairoAssertEq(
    offDst: Int16 = 0, offOp0: Int16 = 0, offOp1: Int16 = 1,
    dstRegIsFP: Bool = false, op0RegIsFP: Bool = false,
    op1Src: CairoOp1Src = .pc, resLogic: CairoResLogic = .op1,
    apUpdate: CairoAPUpdate = .addOne
) -> CairoInstruction {
    CairoInstruction(
        offDst: offDst, offOp0: offOp0, offOp1: offOp1,
        dstRegIsFP: dstRegIsFP, op0RegIsFP: op0RegIsFP,
        op1Src: op1Src, resLogic: resLogic,
        pcUpdate: .regular, apUpdate: apUpdate,
        opcodeType: .assertEq
    )
}

/// Build a call instruction: push (pc+size, fp) to [ap], [ap+1], jump to target.
public func cairoCall(
    offOp1: Int16 = 1,
    op1Src: CairoOp1Src = .pc,
    pcUpdate: CairoPCUpdate = .jumpRel
) -> CairoInstruction {
    CairoInstruction(
        offDst: 0, offOp0: 1, offOp1: offOp1,
        dstRegIsFP: false, op0RegIsFP: false,
        op1Src: op1Src, resLogic: .op1,
        pcUpdate: pcUpdate, apUpdate: .addTwo,
        opcodeType: .call
    )
}

/// Build a ret instruction: fp = mem[fp-2], pc = mem[fp-1].
public func cairoRet() -> CairoInstruction {
    CairoInstruction(
        offDst: -2, offOp0: -1, offOp1: -1,
        dstRegIsFP: true, op0RegIsFP: true,
        op1Src: .fp, resLogic: .op1,
        pcUpdate: .jump, apUpdate: .regular,
        opcodeType: .ret
    )
}

/// Build an unconditional jump instruction: pc = pc + op1 (relative).
public func cairoJmpRel(
    offOp1: Int16 = 1,
    op1Src: CairoOp1Src = .pc
) -> CairoInstruction {
    CairoInstruction(
        offDst: 0, offOp0: 0, offOp1: offOp1,
        dstRegIsFP: false, op0RegIsFP: false,
        op1Src: op1Src, resLogic: .op1,
        pcUpdate: .jumpRel, apUpdate: .regular,
        opcodeType: .jmp
    )
}

/// Build a conditional jump (jnz): if mem[ap+offDst] != 0, pc += op1.
public func cairoJnz(
    offDst: Int16 = -1,
    offOp1: Int16 = 1,
    op1Src: CairoOp1Src = .pc,
    dstRegIsFP: Bool = false
) -> CairoInstruction {
    CairoInstruction(
        offDst: offDst, offOp0: 0, offOp1: offOp1,
        dstRegIsFP: dstRegIsFP, op0RegIsFP: false,
        op1Src: op1Src, resLogic: .op1,
        pcUpdate: .jnz, apUpdate: .regular,
        opcodeType: .jmp
    )
}

/// Build an add_ap instruction: ap += res.
public func cairoAddAP(
    offOp1: Int16 = 1,
    op1Src: CairoOp1Src = .pc,
    resLogic: CairoResLogic = .op1
) -> CairoInstruction {
    CairoInstruction(
        offDst: 0, offOp0: 0, offOp1: offOp1,
        dstRegIsFP: false, op0RegIsFP: false,
        op1Src: op1Src, resLogic: resLogic,
        pcUpdate: .regular, apUpdate: .add,
        opcodeType: .jmp
    )
}
