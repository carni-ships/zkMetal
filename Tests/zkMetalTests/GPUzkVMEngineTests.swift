// Tests for GPUzkVMEngine (GPU-accelerated zkVM)

import Foundation
import zkMetal

public func runGPUzkVMEngineTests() {
    suite("GPUzkVMEngine - Basic Arithmetic")
    testzkVMAdd()
    testzkVMMul()
    testzkVMAddMulChain()

    suite("GPUzkVMEngine - Memory")
    testzkVMStoreLoad()
    testzkVMMemoryConsistency()
    testzkVMMemoryConsistencyFailure()

    suite("GPUzkVMEngine - Branches")
    testzkVMBranchEq()
    testzkVMBranchNe()
    testzkVMBranchNotTaken()

    suite("GPUzkVMEngine - Halt & Limits")
    testzkVMHalt()
    testzkVMStepLimit()

    suite("GPUzkVMEngine - Programs")
    testzkVMFibonacci()
    testzkVMAccumulator()

    suite("GPUzkVMEngine - Trace Validation")
    testzkVMTraceValidation()
    testzkVMTraceStepIndices()

    suite("GPUzkVMEngine - Configuration")
    testzkVMCustomRegisterCount()
    testzkVMInitialRegisters()
}

// MARK: - Basic Arithmetic

private func testzkVMAdd() {
    let engine = GPUzkVMEngine(registerCount: 8)
    // r0=3, r1=5, then r2 = r0 + r1
    let program: [zkVMInstruction] = [
        zkVMInstruction(opcode: .add, dst: 2, arg1: 0, arg2: 1),
        zkVMInstruction(opcode: .halt, dst: 0, arg1: 0, arg2: 0),
    ]
    let initRegs = [frFromInt(3), frFromInt(5)]
    let result = engine.execute(program: program, initialRegisters: initRegs)

    expect(result.halted, "should halt")
    expect(result.registers[2] == frFromInt(8), "r2 = 3 + 5 = 8")
    expectEqual(result.stepCount, 2, "2 steps (add + halt)")
}

private func testzkVMMul() {
    let engine = GPUzkVMEngine(registerCount: 8)
    // r0=7, r1=6, then r2 = r0 * r1
    let program: [zkVMInstruction] = [
        zkVMInstruction(opcode: .mul, dst: 2, arg1: 0, arg2: 1),
        zkVMInstruction(opcode: .halt, dst: 0, arg1: 0, arg2: 0),
    ]
    let initRegs = [frFromInt(7), frFromInt(6)]
    let result = engine.execute(program: program, initialRegisters: initRegs)

    expect(result.halted, "should halt")
    expect(result.registers[2] == frFromInt(42), "r2 = 7 * 6 = 42")
}

private func testzkVMAddMulChain() {
    let engine = GPUzkVMEngine(registerCount: 8)
    // r0=2, r1=3, r2=4
    // r3 = r0 + r1 = 5
    // r4 = r3 * r2 = 20
    let program: [zkVMInstruction] = [
        zkVMInstruction(opcode: .add, dst: 3, arg1: 0, arg2: 1),
        zkVMInstruction(opcode: .mul, dst: 4, arg1: 3, arg2: 2),
        zkVMInstruction(opcode: .halt, dst: 0, arg1: 0, arg2: 0),
    ]
    let initRegs = [frFromInt(2), frFromInt(3), frFromInt(4)]
    let result = engine.execute(program: program, initialRegisters: initRegs)

    expect(result.halted, "should halt")
    expect(result.registers[3] == frFromInt(5), "r3 = 2 + 3 = 5")
    expect(result.registers[4] == frFromInt(20), "r4 = 5 * 4 = 20")
}

// MARK: - Memory

private func testzkVMStoreLoad() {
    let engine = GPUzkVMEngine(registerCount: 8)
    // r0=100 (address base), r1=42 (value)
    // store r1 at memory[r0 + 0]
    // load r2 from memory[r0 + 0]
    let program: [zkVMInstruction] = [
        zkVMInstruction(opcode: .store, dst: 1, arg1: 0, arg2: 0),  // mem[100] = r1
        zkVMInstruction(opcode: .load, dst: 2, arg1: 0, arg2: 0),   // r2 = mem[100]
        zkVMInstruction(opcode: .halt, dst: 0, arg1: 0, arg2: 0),
    ]
    let initRegs = [frFromInt(100), frFromInt(42)]
    let result = engine.execute(program: program, initialRegisters: initRegs)

    expect(result.halted, "should halt")
    expect(result.registers[2] == frFromInt(42), "r2 = loaded 42 from memory")
    expectEqual(result.memoryAccesses.count, 2, "1 write + 1 read")
}

private func testzkVMMemoryConsistency() {
    let engine = GPUzkVMEngine(registerCount: 8)
    // Write then read -- should be consistent
    let program: [zkVMInstruction] = [
        zkVMInstruction(opcode: .store, dst: 1, arg1: 0, arg2: 0),
        zkVMInstruction(opcode: .load, dst: 2, arg1: 0, arg2: 0),
        zkVMInstruction(opcode: .halt, dst: 0, arg1: 0, arg2: 0),
    ]
    let initRegs = [frFromInt(10), frFromInt(99)]
    let result = engine.execute(program: program, initialRegisters: initRegs)

    let consistent = engine.verifyMemoryConsistency(result.memoryAccesses)
    expect(consistent, "memory should be consistent after write-then-read")
}

private func testzkVMMemoryConsistencyFailure() {
    let engine = GPUzkVMEngine(registerCount: 8)
    // Manually craft inconsistent memory accesses
    let badAccesses = [
        zkVMMemoryAccess(step: 0, address: 5, value: frFromInt(10), isWrite: true),
        zkVMMemoryAccess(step: 1, address: 5, value: frFromInt(99), isWrite: false),  // wrong!
    ]
    let consistent = engine.verifyMemoryConsistency(badAccesses)
    expect(!consistent, "should detect inconsistent memory")
}

// MARK: - Branches

private func testzkVMBranchEq() {
    let engine = GPUzkVMEngine(registerCount: 8)
    // r0=5, r1=5 (equal), branch to instruction 3 if equal
    // 0: branchEq to 2 if r0==r1 (taken)
    // 1: add r2 = r0 + r1 (skipped)
    // 2: halt
    let program: [zkVMInstruction] = [
        zkVMInstruction(opcode: .branchEq, dst: 2, arg1: 0, arg2: 1),
        zkVMInstruction(opcode: .add, dst: 2, arg1: 0, arg2: 1),
        zkVMInstruction(opcode: .halt, dst: 0, arg1: 0, arg2: 0),
    ]
    let initRegs = [frFromInt(5), frFromInt(5)]
    let result = engine.execute(program: program, initialRegisters: initRegs)

    expect(result.halted, "should halt")
    // r2 should still be zero (add was skipped)
    expect(result.registers[2] == Fr.zero, "add should be skipped by branch")
    expectEqual(result.stepCount, 2, "branch + halt = 2 steps")
}

private func testzkVMBranchNe() {
    let engine = GPUzkVMEngine(registerCount: 8)
    // r0=3, r1=7 (not equal), branch to 2 if not equal (taken)
    let program: [zkVMInstruction] = [
        zkVMInstruction(opcode: .branchNe, dst: 2, arg1: 0, arg2: 1),
        zkVMInstruction(opcode: .add, dst: 2, arg1: 0, arg2: 1),  // skipped
        zkVMInstruction(opcode: .halt, dst: 0, arg1: 0, arg2: 0),
    ]
    let initRegs = [frFromInt(3), frFromInt(7)]
    let result = engine.execute(program: program, initialRegisters: initRegs)

    expect(result.halted, "should halt")
    expect(result.registers[2] == Fr.zero, "add should be skipped")
}

private func testzkVMBranchNotTaken() {
    let engine = GPUzkVMEngine(registerCount: 8)
    // r0=3, r1=7 (not equal), branchEq not taken
    let program: [zkVMInstruction] = [
        zkVMInstruction(opcode: .branchEq, dst: 2, arg1: 0, arg2: 1),  // not taken
        zkVMInstruction(opcode: .add, dst: 2, arg1: 0, arg2: 1),       // executed
        zkVMInstruction(opcode: .halt, dst: 0, arg1: 0, arg2: 0),
    ]
    let initRegs = [frFromInt(3), frFromInt(7)]
    let result = engine.execute(program: program, initialRegisters: initRegs)

    expect(result.halted, "should halt")
    expect(result.registers[2] == frFromInt(10), "add should execute: 3+7=10")
    expectEqual(result.stepCount, 3, "branch(not taken) + add + halt = 3")
}

// MARK: - Halt & Limits

private func testzkVMHalt() {
    let engine = GPUzkVMEngine(registerCount: 4)
    let program: [zkVMInstruction] = [
        zkVMInstruction(opcode: .halt, dst: 0, arg1: 0, arg2: 0),
    ]
    let result = engine.execute(program: program)
    expect(result.halted, "should halt immediately")
    expectEqual(result.stepCount, 1, "one halt step")
}

private func testzkVMStepLimit() {
    let engine = GPUzkVMEngine(registerCount: 4)
    // Infinite loop: branch to self
    let program: [zkVMInstruction] = [
        zkVMInstruction(opcode: .add, dst: 0, arg1: 0, arg2: 0),       // r0 = r0+r0 (noop if 0)
        zkVMInstruction(opcode: .branchEq, dst: 0, arg1: 0, arg2: 0),  // always branch to 0
    ]
    let result = engine.execute(program: program, stepLimit: 100)
    expect(!result.halted, "should NOT halt (hit step limit)")
    expectEqual(result.stepCount, 100, "should execute exactly 100 steps")
}

// MARK: - Programs

private func testzkVMFibonacci() {
    // Compute fib(10) = 55
    // r0 = fib(n-2), r1 = fib(n-1), r2 = counter, r3 = limit, r4 = one, r5 = temp
    let engine = GPUzkVMEngine(registerCount: 8)
    let program: [zkVMInstruction] = [
        // 0: r5 = r0 + r1 (fib step)
        zkVMInstruction(opcode: .add, dst: 5, arg1: 0, arg2: 1),
        // 1: r0 = r1 (shift)
        zkVMInstruction(opcode: .add, dst: 0, arg1: 1, arg2: 6),  // r6=0, so r0 = r1+0
        // 2: r1 = r5 (shift)
        zkVMInstruction(opcode: .add, dst: 1, arg1: 5, arg2: 6),  // r1 = r5+0
        // 3: r2 = r2 + r4 (counter++)
        zkVMInstruction(opcode: .add, dst: 2, arg1: 2, arg2: 4),
        // 4: branchNe to 0 if r2 != r3
        zkVMInstruction(opcode: .branchNe, dst: 0, arg1: 2, arg2: 3),
        // 5: halt
        zkVMInstruction(opcode: .halt, dst: 0, arg1: 0, arg2: 0),
    ]
    // r0=0, r1=1, r2=0(counter), r3=10(limit), r4=1(increment), r5=0, r6=0
    let initRegs = [frFromInt(0), frFromInt(1), frFromInt(0), frFromInt(10),
                    frFromInt(1), frFromInt(0), frFromInt(0), frFromInt(0)]
    let result = engine.execute(program: program, stepLimit: 10000, initialRegisters: initRegs)

    expect(result.halted, "fibonacci should halt")
    expect(result.registers[1] == frFromInt(55), "fib(10) = 55")
}

private func testzkVMAccumulator() {
    // Sum r0 += r1 five times using a loop
    // r0 = accumulator, r1 = addend, r2 = counter, r3 = limit, r4 = one
    let engine = GPUzkVMEngine(registerCount: 8)
    let program: [zkVMInstruction] = [
        // 0: r0 = r0 + r1
        zkVMInstruction(opcode: .add, dst: 0, arg1: 0, arg2: 1),
        // 1: r2 = r2 + r4 (counter++)
        zkVMInstruction(opcode: .add, dst: 2, arg1: 2, arg2: 4),
        // 2: branchNe to 0 if r2 != r3
        zkVMInstruction(opcode: .branchNe, dst: 0, arg1: 2, arg2: 3),
        // 3: halt
        zkVMInstruction(opcode: .halt, dst: 0, arg1: 0, arg2: 0),
    ]
    // r0=0, r1=7, r2=0, r3=5, r4=1
    let initRegs = [frFromInt(0), frFromInt(7), frFromInt(0), frFromInt(5), frFromInt(1)]
    let result = engine.execute(program: program, stepLimit: 10000, initialRegisters: initRegs)

    expect(result.halted, "accumulator should halt")
    expect(result.registers[0] == frFromInt(35), "0 + 7*5 = 35")
}

// MARK: - Trace Validation

private func testzkVMTraceValidation() {
    let engine = GPUzkVMEngine(registerCount: 8)
    let program: [zkVMInstruction] = [
        zkVMInstruction(opcode: .add, dst: 2, arg1: 0, arg2: 1),
        zkVMInstruction(opcode: .mul, dst: 3, arg1: 0, arg2: 1),
        zkVMInstruction(opcode: .halt, dst: 0, arg1: 0, arg2: 0),
    ]
    let initRegs = [frFromInt(3), frFromInt(5)]
    let result = engine.execute(program: program, initialRegisters: initRegs)

    let valid = engine.validateTrace(result.trace)
    expect(valid, "trace should be valid")
}

private func testzkVMTraceStepIndices() {
    let engine = GPUzkVMEngine(registerCount: 4)
    let program: [zkVMInstruction] = [
        zkVMInstruction(opcode: .add, dst: 0, arg1: 0, arg2: 0),
        zkVMInstruction(opcode: .add, dst: 0, arg1: 0, arg2: 0),
        zkVMInstruction(opcode: .halt, dst: 0, arg1: 0, arg2: 0),
    ]
    let result = engine.execute(program: program)

    for (i, row) in result.trace.enumerated() {
        expectEqual(row.step, i, "step index \(i) should match")
    }
}

// MARK: - Configuration

private func testzkVMCustomRegisterCount() {
    let engine = GPUzkVMEngine(registerCount: 4)
    // Access register 3 (valid), register 5 should return zero (out of bounds)
    let program: [zkVMInstruction] = [
        zkVMInstruction(opcode: .add, dst: 2, arg1: 0, arg2: 1),
        zkVMInstruction(opcode: .halt, dst: 0, arg1: 0, arg2: 0),
    ]
    let initRegs = [frFromInt(10), frFromInt(20), frFromInt(0), frFromInt(0)]
    let result = engine.execute(program: program, initialRegisters: initRegs)

    expect(result.registers[2] == frFromInt(30), "r2 = 10+20 = 30")
    expectEqual(result.registers.count, 4, "should have 4 registers")
}

private func testzkVMInitialRegisters() {
    let engine = GPUzkVMEngine(registerCount: 8)
    // Verify initial register values are preserved if no instructions modify them
    let program: [zkVMInstruction] = [
        zkVMInstruction(opcode: .halt, dst: 0, arg1: 0, arg2: 0),
    ]
    let initRegs = [frFromInt(11), frFromInt(22), frFromInt(33)]
    let result = engine.execute(program: program, initialRegisters: initRegs)

    expect(result.registers[0] == frFromInt(11), "r0 preserved")
    expect(result.registers[1] == frFromInt(22), "r1 preserved")
    expect(result.registers[2] == frFromInt(33), "r2 preserved")
    expect(result.registers[3] == Fr.zero, "r3 default zero")
}
