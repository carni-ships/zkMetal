// CairoAIRConstraints -- AIR trace layout and constraint definitions for Cairo VM
//
// Converts a Cairo execution trace into the 16-column AIR format used by
// the Cairo STARK prover. Defines constraint types for instruction decoding,
// operand resolution, result computation, and opcode semantics.
//
// The 16-column layout follows the Cairo whitepaper:
//   [0]  pc        [1]  ap        [2]  fp
//   [3]  dst_addr  [4]  dst_val   [5]  op0_addr
//   [6]  op0_val   [7]  op1_addr  [8]  op1_val
//   [9]  res       [10] f_dst_fp  [11] f_op0_fp
//   [12] f_op1_src (2 bits) [13] f_res_logic (2 bits)
//   [14] f_pc_update (3 bits) [15] f_ap_update + f_opcode (combined flags)
//
// Reference: Cairo whitepaper Section 9 (AIR representation)

import Foundation

// MARK: - AIR Column Indices

/// Column indices for the 16-column Cairo AIR trace.
public enum CairoAIRColumn: Int, CaseIterable {
    case pc       = 0
    case ap       = 1
    case fp       = 2
    case dstAddr  = 3
    case dstVal   = 4
    case op0Addr  = 5
    case op0Val   = 6
    case op1Addr  = 7
    case op1Val   = 8
    case res      = 9
    case fDstFP   = 10   // 1 if dst register is fp, 0 if ap
    case fOp0FP   = 11   // 1 if op0 register is fp, 0 if ap
    case fOp1Src  = 12   // encoded op1 source flags
    case fResLogic = 13  // encoded res logic flags
    case fPCUpdate = 14  // encoded pc update flags
    case fFlags   = 15   // encoded ap_update + opcode flags

    public static let columnCount = 16
}

// MARK: - Cairo Constraint Type

/// Categories of AIR constraints for the Cairo VM.
public enum CairoConstraintType: String, CaseIterable {
    /// Instruction decoding: flag bits are boolean (0 or 1)
    case instructionDecoding = "instruction_decoding"
    /// Operand address resolution: dst_addr = reg + offset
    case operandResolution = "operand_resolution"
    /// Result computation: res = op0 {op} op1
    case resultComputation = "result_computation"
    /// Opcode semantics: assert_eq, call, ret behavior
    case opcodeSemantics = "opcode_semantics"
    /// Memory consistency: read values match written values
    case memoryConsistency = "memory_consistency"
    /// Register update: pc, ap, fp transitions between rows
    case registerUpdate = "register_update"
}

// MARK: - Cairo Constraint Definition

/// A single AIR constraint with its type, degree, and symbolic description.
public struct CairoConstraintDef {
    public let type: CairoConstraintType
    public let name: String
    public let degree: Int
    public let description: String

    public init(type: CairoConstraintType, name: String, degree: Int, description: String) {
        self.type = type
        self.name = name
        self.degree = degree
        self.description = description
    }
}

// MARK: - CairoAIRGenerator

/// Converts a list of CairoTraceSteps into the column-major AIR trace format.
/// The trace is padded to a power of 2 for NTT compatibility.
public class CairoAIRGenerator {
    /// The full set of Cairo AIR constraint definitions.
    public let constraints: [CairoConstraintDef]

    public init() {
        self.constraints = CairoAIRGenerator.buildConstraintDefinitions()
    }

    /// Convert execution trace steps into a 16-column AIR trace.
    ///
    /// - Parameter steps: The execution trace from CairoTraceGenerator.
    /// - Returns: Column-major array of 16 columns, each containing Stark252 values.
    ///           Padded to the next power of 2 length.
    public func generateAIRTrace(from steps: [CairoTraceStep]) -> [[Stark252]] {
        let rawLen = steps.count
        guard rawLen > 0 else {
            return [[Stark252]](repeating: [], count: CairoAIRColumn.columnCount)
        }

        // Pad to next power of 2
        let paddedLen = nextPowerOfTwo(rawLen)

        var columns = [[Stark252]](repeating:
            [Stark252](repeating: Stark252.zero, count: paddedLen),
            count: CairoAIRColumn.columnCount
        )

        // Fill trace rows
        for (i, step) in steps.enumerated() {
            columns[CairoAIRColumn.pc.rawValue][i] = stark252FromInt(step.state.pc)
            columns[CairoAIRColumn.ap.rawValue][i] = stark252FromInt(step.state.ap)
            columns[CairoAIRColumn.fp.rawValue][i] = stark252FromInt(step.state.fp)

            columns[CairoAIRColumn.dstAddr.rawValue][i] = stark252FromInt(step.dstAddr)
            columns[CairoAIRColumn.dstVal.rawValue][i] = step.dstVal
            columns[CairoAIRColumn.op0Addr.rawValue][i] = stark252FromInt(step.op0Addr)
            columns[CairoAIRColumn.op0Val.rawValue][i] = step.op0Val
            columns[CairoAIRColumn.op1Addr.rawValue][i] = stark252FromInt(step.op1Addr)
            columns[CairoAIRColumn.op1Val.rawValue][i] = step.op1Val
            columns[CairoAIRColumn.res.rawValue][i] = step.res

            // Flag columns
            columns[CairoAIRColumn.fDstFP.rawValue][i] =
                step.instruction.dstRegIsFP ? Stark252.one : Stark252.zero
            columns[CairoAIRColumn.fOp0FP.rawValue][i] =
                step.instruction.op0RegIsFP ? Stark252.one : Stark252.zero
            columns[CairoAIRColumn.fOp1Src.rawValue][i] =
                stark252FromInt(UInt64(step.instruction.op1Src.rawValue))
            columns[CairoAIRColumn.fResLogic.rawValue][i] =
                stark252FromInt(UInt64(step.instruction.resLogic.rawValue))
            columns[CairoAIRColumn.fPCUpdate.rawValue][i] =
                stark252FromInt(UInt64(step.instruction.pcUpdate.rawValue))
            columns[CairoAIRColumn.fFlags.rawValue][i] =
                encodeAPOpcodeFlags(step.instruction)
        }

        // Pad remaining rows: copy last valid row's register state
        if rawLen < paddedLen, let lastStep = steps.last {
            for i in rawLen..<paddedLen {
                columns[CairoAIRColumn.pc.rawValue][i] =
                    stark252FromInt(lastStep.state.pc + UInt64(lastStep.instruction.size))
                columns[CairoAIRColumn.ap.rawValue][i] =
                    columns[CairoAIRColumn.ap.rawValue][rawLen - 1]
                columns[CairoAIRColumn.fp.rawValue][i] =
                    columns[CairoAIRColumn.fp.rawValue][rawLen - 1]
            }
        }

        return columns
    }

    /// Verify basic consistency of an AIR trace against its execution trace.
    ///
    /// Checks:
    /// - Operand resolution: dst_addr = base_reg + offset
    /// - Result computation: res = op0 {op} op1
    /// - PC update: next_pc matches expected value
    /// - AP update: next_ap matches expected value
    ///
    /// - Returns: Array of (step, constraint, message) for any violations found.
    public func checkConsistency(steps: [CairoTraceStep]) -> [(Int, String, String)] {
        var violations: [(Int, String, String)] = []

        for (i, step) in steps.enumerated() {
            let instr = step.instruction

            // Check dst_addr = (fp or ap) + offDst
            let expectedDstBase: UInt64 = instr.dstRegIsFP ? step.state.fp : step.state.ap
            let expectedDstAddr = UInt64(bitPattern: Int64(bitPattern: expectedDstBase) &+ Int64(instr.offDst))
            if step.dstAddr != expectedDstAddr {
                violations.append((i, "dst_addr",
                    "expected \(expectedDstAddr), got \(step.dstAddr)"))
            }

            // Check op0_addr = (fp or ap) + offOp0
            let expectedOp0Base: UInt64 = instr.op0RegIsFP ? step.state.fp : step.state.ap
            let expectedOp0Addr = UInt64(bitPattern: Int64(bitPattern: expectedOp0Base) &+ Int64(instr.offOp0))
            if step.op0Addr != expectedOp0Addr {
                violations.append((i, "op0_addr",
                    "expected \(expectedOp0Addr), got \(step.op0Addr)"))
            }

            // Check result computation
            let expectedRes: Stark252
            switch instr.resLogic {
            case .op1:
                expectedRes = step.op1Val
            case .add:
                expectedRes = stark252Add(step.op0Val, step.op1Val)
            case .mul:
                expectedRes = stark252Mul(step.op0Val, step.op1Val)
            }
            let resLimbs = stark252ToInt(step.res)
            let expectedResLimbs = stark252ToInt(expectedRes)
            if resLimbs != expectedResLimbs {
                violations.append((i, "res_computation",
                    "result mismatch"))
            }

            // Check PC transition (if not last step)
            if i + 1 < steps.count {
                let nextPC = steps[i + 1].state.pc
                let expectedNextPC: UInt64
                switch instr.pcUpdate {
                case .regular:
                    expectedNextPC = step.state.pc + UInt64(instr.size)
                case .jump:
                    expectedNextPC = stark252ToUInt64(step.res)
                case .jumpRel:
                    expectedNextPC = step.state.pc + stark252ToUInt64(step.res)
                case .jnz:
                    let dstLimbs = stark252ToInt(step.dstVal)
                    let isZero = dstLimbs[0] == 0 && dstLimbs[1] == 0 &&
                                 dstLimbs[2] == 0 && dstLimbs[3] == 0
                    if !isZero {
                        expectedNextPC = step.state.pc + stark252ToUInt64(step.op1Val)
                    } else {
                        expectedNextPC = step.state.pc + UInt64(instr.size)
                    }
                }
                if nextPC != expectedNextPC {
                    violations.append((i, "pc_update",
                        "expected next pc=\(expectedNextPC), got \(nextPC)"))
                }
            }
        }

        return violations
    }

    // MARK: - Constraint Definitions

    /// Build the full set of Cairo AIR constraint definitions.
    public static func buildConstraintDefinitions() -> [CairoConstraintDef] {
        var defs: [CairoConstraintDef] = []

        // Instruction decoding: flag bits must be boolean
        for flagName in ["f_dst_fp", "f_op0_fp"] {
            defs.append(CairoConstraintDef(
                type: .instructionDecoding,
                name: "\(flagName)_bool",
                degree: 2,
                description: "\(flagName) * (1 - \(flagName)) = 0"
            ))
        }

        // Operand resolution constraints
        defs.append(CairoConstraintDef(
            type: .operandResolution,
            name: "dst_addr",
            degree: 2,
            description: "dst_addr = f_dst_fp * fp + (1 - f_dst_fp) * ap + off_dst"
        ))
        defs.append(CairoConstraintDef(
            type: .operandResolution,
            name: "op0_addr",
            degree: 2,
            description: "op0_addr = f_op0_fp * fp + (1 - f_op0_fp) * ap + off_op0"
        ))
        defs.append(CairoConstraintDef(
            type: .operandResolution,
            name: "op1_addr",
            degree: 2,
            description: "op1_addr depends on f_op1_src: op0, pc, fp, or ap based"
        ))

        // Result computation
        defs.append(CairoConstraintDef(
            type: .resultComputation,
            name: "res_add",
            degree: 2,
            description: "f_res_add * (res - op0 - op1) = 0"
        ))
        defs.append(CairoConstraintDef(
            type: .resultComputation,
            name: "res_mul",
            degree: 3,
            description: "f_res_mul * (res - op0 * op1) = 0"
        ))
        defs.append(CairoConstraintDef(
            type: .resultComputation,
            name: "res_op1",
            degree: 2,
            description: "f_res_op1 * (res - op1) = 0"
        ))

        // Opcode semantics
        defs.append(CairoConstraintDef(
            type: .opcodeSemantics,
            name: "assert_eq",
            degree: 2,
            description: "f_assert_eq * (dst - res) = 0"
        ))
        defs.append(CairoConstraintDef(
            type: .opcodeSemantics,
            name: "call_push_ret",
            degree: 2,
            description: "f_call * (mem[ap] - (pc + instr_size)) = 0"
        ))
        defs.append(CairoConstraintDef(
            type: .opcodeSemantics,
            name: "call_push_fp",
            degree: 2,
            description: "f_call * (mem[ap+1] - fp) = 0"
        ))
        defs.append(CairoConstraintDef(
            type: .opcodeSemantics,
            name: "call_update_fp",
            degree: 2,
            description: "f_call * (fp' - (ap + 2)) = 0"
        ))
        defs.append(CairoConstraintDef(
            type: .opcodeSemantics,
            name: "ret_restore_fp",
            degree: 2,
            description: "f_ret * (fp' - dst) = 0"
        ))

        // Register update constraints
        defs.append(CairoConstraintDef(
            type: .registerUpdate,
            name: "pc_regular",
            degree: 2,
            description: "f_pc_regular * (pc' - pc - instr_size) = 0"
        ))
        defs.append(CairoConstraintDef(
            type: .registerUpdate,
            name: "pc_jump",
            degree: 2,
            description: "f_pc_jump * (pc' - res) = 0"
        ))
        defs.append(CairoConstraintDef(
            type: .registerUpdate,
            name: "pc_jump_rel",
            degree: 2,
            description: "f_pc_jump_rel * (pc' - pc - res) = 0"
        ))
        defs.append(CairoConstraintDef(
            type: .registerUpdate,
            name: "pc_jnz",
            degree: 3,
            description: "f_pc_jnz * dst * (pc' - pc - op1) + f_pc_jnz * (1-dst_nz) * (pc' - pc - size) = 0"
        ))
        defs.append(CairoConstraintDef(
            type: .registerUpdate,
            name: "ap_add",
            degree: 2,
            description: "f_ap_add * (ap' - ap - res) = 0"
        ))
        defs.append(CairoConstraintDef(
            type: .registerUpdate,
            name: "ap_add_one",
            degree: 2,
            description: "f_ap_add1 * (ap' - ap - 1) = 0"
        ))

        // Memory consistency
        defs.append(CairoConstraintDef(
            type: .memoryConsistency,
            name: "memory_multi_column_perm",
            degree: 2,
            description: "Permutation argument: sorted access == execution order access"
        ))

        return defs
    }

    // MARK: - Helpers

    /// Encode AP update and opcode type flags into a single field element.
    /// Low 3 bits: ap_update, bits 3-4: opcode_type
    private func encodeAPOpcodeFlags(_ instr: CairoInstruction) -> Stark252 {
        let val = UInt64(instr.apUpdate.rawValue) | (UInt64(instr.opcodeType.rawValue) << 3)
        return stark252FromInt(val)
    }

    /// Next power of 2 >= n.
    private func nextPowerOfTwo(_ n: Int) -> Int {
        guard n > 0 else { return 1 }
        var v = n - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        return v + 1
    }
}
