// CairoTraceTests -- Tests for Cairo VM trace generator and AIR constraints
//
// Validates instruction execution, trace capture, memory tracking,
// AIR column layout, and constraint consistency for the Cairo VM.

import zkMetal

public func runCairoTraceTests() {
    suite("Cairo Trace — Assert Eq")
    testAssertEqSimple()
    testAssertEqAdd()
    testAssertEqMul()

    suite("Cairo Trace — Jump Instructions")
    testJmpRel()
    testJnzTaken()
    testJnzNotTaken()

    suite("Cairo Trace — Call and Return")
    testCallAndRet()
    testCallPushesReturnAddress()

    suite("Cairo Trace — Memory")
    testMemoryReadWriteTracking()
    testMemoryWriteOnce()

    suite("Cairo Trace — AIR Trace")
    testAIRColumnCount()
    testAIRRowCount()
    testAIRTracePadding()

    suite("Cairo Trace — Consistency")
    testPCAdvancesCorrectly()
    testAPUpdates()
    testAIRConsistencyCheck()
}

// MARK: - Helper: compare Stark252 values

private func stark252Eq(_ a: Stark252, _ b: Stark252) -> Bool {
    let al = stark252ToInt(a)
    let bl = stark252ToInt(b)
    return al == bl
}

private func stark252IsVal(_ a: Stark252, _ val: UInt64) -> Bool {
    let limbs = stark252ToInt(a)
    return limbs[0] == val && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0
}

// MARK: - Assert Eq Tests

func testAssertEqSimple() {
    // Program: assert mem[ap] = 5 (immediate)
    // Instruction: assert_eq, op1_src=pc (immediate), res=op1
    let instr = cairoAssertEq(
        offDst: 0, offOp0: 0, offOp1: 1,
        dstRegIsFP: false, op0RegIsFP: false,
        op1Src: .pc, resLogic: .op1,
        apUpdate: .addOne
    )

    let mem = CairoMemory()
    // Store immediate value 5 at address 1 (pc+1)
    mem.write(1, stark252FromInt(5))

    let gen = CairoTraceGenerator(stepLimit: 10)
    do {
        let result = try gen.execute(
            instructions: [instr],
            memory: mem,
            initialState: CairoState(pc: 0, ap: 2, fp: 2)
        )
        expectEqual(result.stepCount, 1, "assert_eq should execute 1 step")

        let step = result.trace[0]
        expect(stark252IsVal(step.res, 5), "res should be 5")
        expect(stark252IsVal(step.dstVal, 5), "dst should be written as 5")
    } catch {
        expect(false, "assert_eq(5) threw: \(error)")
    }
}

func testAssertEqAdd() {
    // Program: assert mem[ap] = mem[fp+0] + mem[fp+1]
    // Set up: mem[fp+0] = 3, mem[fp+1] = 7 => result should be 10
    let instr = cairoAssertEq(
        offDst: 0, offOp0: 0, offOp1: 1,
        dstRegIsFP: false, op0RegIsFP: true,
        op1Src: .fp, resLogic: .add,
        apUpdate: .addOne
    )

    let mem = CairoMemory()
    mem.write(4, stark252FromInt(3))   // fp+0 (fp=4)
    mem.write(5, stark252FromInt(7))   // fp+1

    let gen = CairoTraceGenerator(stepLimit: 10)
    do {
        let result = try gen.execute(
            instructions: [instr],
            memory: mem,
            initialState: CairoState(pc: 0, ap: 6, fp: 4)
        )
        let step = result.trace[0]
        expect(stark252IsVal(step.res, 10), "3 + 7 should be 10")
        expect(stark252IsVal(step.dstVal, 10), "dst should be 10")
        expect(stark252IsVal(step.op0Val, 3), "op0 should be 3")
        expect(stark252IsVal(step.op1Val, 7), "op1 should be 7")
    } catch {
        expect(false, "assert_eq(add) threw: \(error)")
    }
}

func testAssertEqMul() {
    // Program: assert mem[ap] = mem[fp+0] * mem[fp+1]
    // Set up: mem[fp+0] = 6, mem[fp+1] = 7 => result should be 42
    let instr = cairoAssertEq(
        offDst: 0, offOp0: 0, offOp1: 1,
        dstRegIsFP: false, op0RegIsFP: true,
        op1Src: .fp, resLogic: .mul,
        apUpdate: .addOne
    )

    let mem = CairoMemory()
    mem.write(4, stark252FromInt(6))
    mem.write(5, stark252FromInt(7))

    let gen = CairoTraceGenerator(stepLimit: 10)
    do {
        let result = try gen.execute(
            instructions: [instr],
            memory: mem,
            initialState: CairoState(pc: 0, ap: 6, fp: 4)
        )
        let step = result.trace[0]
        expect(stark252IsVal(step.res, 42), "6 * 7 should be 42")
    } catch {
        expect(false, "assert_eq(mul) threw: \(error)")
    }
}

// MARK: - Jump Tests

func testJmpRel() {
    // Jump forward by 3: pc = pc + 3
    let jmp = cairoJmpRel(offOp1: 1, op1Src: .pc)

    // After the jmp (size=2), we need a landing instruction at pc=3
    let landing = cairoAssertEq(
        offDst: 0, offOp0: 0, offOp1: 1,
        dstRegIsFP: false, op0RegIsFP: false,
        op1Src: .pc, resLogic: .op1,
        apUpdate: .addOne
    )

    let mem = CairoMemory()
    // Immediate for jmp: jump offset = 3
    mem.write(1, stark252FromInt(3))
    // Immediate for landing at pc=3: value = 99
    mem.write(4, stark252FromInt(99))

    let gen = CairoTraceGenerator(stepLimit: 10)
    do {
        let result = try gen.execute(
            instructions: [jmp, landing],
            memory: mem,
            initialState: CairoState(pc: 0, ap: 10, fp: 10)
        )
        // After jmp, pc should be 0+3 = 3
        expect(result.trace.count >= 1, "should execute at least jmp")
        if result.trace.count >= 2 {
            expectEqual(result.trace[1].state.pc, 3, "pc should jump to 3")
        }
    } catch {
        expect(false, "jmp_rel threw: \(error)")
    }
}

func testJnzTaken() {
    // jnz with non-zero dst: should jump
    // First write a non-zero value where dst will read
    let jnz = cairoJnz(offDst: 0, offOp1: 1, op1Src: .pc, dstRegIsFP: false)

    let mem = CairoMemory()
    // ap=10, so dst is at ap+0 = 10
    mem.write(10, stark252FromInt(1))  // non-zero => jump taken
    // Immediate: jump offset = 5
    mem.write(1, stark252FromInt(5))

    let gen = CairoTraceGenerator(stepLimit: 10)
    do {
        let result = try gen.execute(
            instructions: [jnz],
            memory: mem,
            initialState: CairoState(pc: 0, ap: 10, fp: 10)
        )
        expectEqual(result.stepCount, 1, "jnz executes 1 step")
        // PC should have jumped to 0 + 5 = 5
        expectEqual(result.finalState.pc, 5, "jnz taken: pc should be 5")
    } catch {
        expect(false, "jnz_taken threw: \(error)")
    }
}

func testJnzNotTaken() {
    // jnz with zero dst: should fall through
    let jnz = cairoJnz(offDst: 0, offOp1: 1, op1Src: .pc, dstRegIsFP: false)

    let mem = CairoMemory()
    // ap=10, dst at 10 is zero (default)
    mem.write(1, stark252FromInt(5))  // immediate (won't be used)

    let gen = CairoTraceGenerator(stepLimit: 10)
    do {
        let result = try gen.execute(
            instructions: [jnz],
            memory: mem,
            initialState: CairoState(pc: 0, ap: 10, fp: 10)
        )
        expectEqual(result.stepCount, 1, "jnz executes 1 step")
        // PC should fall through: 0 + 2 (size of jnz with immediate) = 2
        expectEqual(result.finalState.pc, 2, "jnz not taken: pc should be 2")
    } catch {
        expect(false, "jnz_not_taken threw: \(error)")
    }
}

// MARK: - Call and Return Tests

func testCallAndRet() {
    // Program layout (word addresses):
    //   pc=0: call (size=2, immediate at pc=1) -> jumps to pc=4
    //   pc=2: landing (assert_eq, size=2, immediate at pc=3) <- return lands here
    //   pc=4: function body (assert_eq, size=2, immediate at pc=5)
    //   pc=6: ret (size=1)
    //
    // Execution order: call(0) -> body(4) -> ret(6) -> landing(2) -> halt(4)

    let call = cairoCall(offOp1: 1, op1Src: .pc, pcUpdate: .jumpRel)

    let landing = cairoAssertEq(
        offDst: 0, offOp0: 0, offOp1: 1,
        op1Src: .pc, resLogic: .op1, apUpdate: .addOne
    )

    let body = cairoAssertEq(
        offDst: 0, offOp0: 0, offOp1: 1,
        op1Src: .pc, resLogic: .op1, apUpdate: .addOne
    )

    let ret = cairoRet()

    let mem = CairoMemory()
    // call immediate: jump offset = 4 (pc=0+4 = pc=4, the body)
    mem.write(1, stark252FromInt(4))
    // landing immediate at pc=3
    mem.write(3, stark252FromInt(99))
    // body immediate at pc=5
    mem.write(5, stark252FromInt(42))

    let gen = CairoTraceGenerator(stepLimit: 20)
    do {
        let result = try gen.execute(
            instructions: [call, landing, body, ret],
            memory: mem,
            initialState: CairoState(pc: 0, ap: 10, fp: 10)
        )

        // Execution: call(0) -> body(4) -> ret(6) -> landing(2) -> halt
        expect(result.stepCount >= 3, "call+body+ret should be >= 3 steps")

        if result.stepCount >= 3 {
            expectEqual(result.trace[0].state.pc, 0, "step 0: call at pc=0")
            expectEqual(result.trace[1].state.pc, 4, "step 1: body at pc=4")
            expectEqual(result.trace[2].state.pc, 6, "step 2: ret at pc=6")
            expect(result.trace[2].instruction.opcodeType == .ret, "step 2 should be ret")
        }

        // After ret, PC returns to 2 (the landing pad)
        if result.stepCount >= 4 {
            expectEqual(result.trace[3].state.pc, 2, "step 3: landing at pc=2")
        }
    } catch {
        expect(false, "call_and_ret threw: \(error)")
    }
}

func testCallPushesReturnAddress() {
    // Verify that call pushes return address and fp to [ap] and [ap+1]
    let call = cairoCall(offOp1: 1, op1Src: .pc, pcUpdate: .jumpRel)

    let mem = CairoMemory()
    mem.write(1, stark252FromInt(4))  // jump offset

    let gen = CairoTraceGenerator(stepLimit: 5)
    do {
        let result = try gen.execute(
            instructions: [call],
            memory: mem,
            initialState: CairoState(pc: 0, ap: 10, fp: 8)
        )

        // Check that [ap] = old fp = 8 (saved frame pointer)
        let fpSaved = mem.peek(10)
        expect(fpSaved != nil, "old fp should be written at ap=10")
        if let fv = fpSaved {
            expect(stark252IsVal(fv, 8), "pushed fp should be 8")
        }

        // Check that [ap+1] = return address = pc + instr_size = 0 + 2 = 2
        let retAddrVal = mem.peek(11)
        expect(retAddrVal != nil, "return address should be written at ap+1=11")
        if let rv = retAddrVal {
            expect(stark252IsVal(rv, 2), "return address should be 2")
        }

        // After call, fp should be ap+2 = 12
        expectEqual(result.finalState.fp, 12, "new fp should be ap+2=12")
        // ap should advance by 2
        expectEqual(result.finalState.ap, 12, "ap should advance by 2 to 12")
    } catch {
        expect(false, "call_pushes threw: \(error)")
    }
}

// MARK: - Memory Tests

func testMemoryReadWriteTracking() {
    let mem = CairoMemory()
    mem.write(0, stark252FromInt(100))
    mem.write(1, stark252FromInt(200))
    let _ = mem.read(0)
    let _ = mem.read(1)
    let _ = mem.read(2)  // uninitialized, returns zero

    expect(mem.writtenAddresses.count == 2, "should have 2 written addresses")
    expect(mem.readAddresses.count == 3, "should have 3 read addresses")
    expect(mem.accessCount == 5, "should have 5 total accesses (2 writes + 3 reads)")
    expect(mem.cellCount == 2, "should have 2 cells with values")
}

func testMemoryWriteOnce() {
    let mem = CairoMemory()
    let ok1 = mem.write(0, stark252FromInt(42))
    expect(ok1, "first write should succeed")

    // Writing same value should succeed (write-once consistency)
    let ok2 = mem.write(0, stark252FromInt(42))
    expect(ok2, "writing same value should succeed")

    // Writing different value should fail
    let ok3 = mem.write(0, stark252FromInt(99))
    expect(!ok3, "writing different value should fail")
}

// MARK: - AIR Trace Tests

func testAIRColumnCount() {
    // Execute a simple program and check AIR trace has 16 columns
    let instr = cairoAssertEq(
        offDst: 0, offOp0: 0, offOp1: 1,
        op1Src: .pc, resLogic: .op1, apUpdate: .addOne
    )

    let mem = CairoMemory()
    mem.write(1, stark252FromInt(5))

    let gen = CairoTraceGenerator(stepLimit: 10)
    do {
        let result = try gen.execute(
            instructions: [instr],
            memory: mem,
            initialState: CairoState(pc: 0, ap: 2, fp: 2)
        )

        let airGen = CairoAIRGenerator()
        let airTrace = airGen.generateAIRTrace(from: result.trace)

        expectEqual(airTrace.count, 16, "AIR trace should have 16 columns")
    } catch {
        expect(false, "AIR column count test threw: \(error)")
    }
}

func testAIRRowCount() {
    // Execute 3 steps, AIR trace should be padded to 4 (next power of 2)
    let instr = cairoAssertEq(
        offDst: 0, offOp0: 0, offOp1: 1,
        op1Src: .pc, resLogic: .op1, apUpdate: .addOne
    )

    let mem = CairoMemory()
    // 3 consecutive assert_eq instructions, each with immediate
    mem.write(1, stark252FromInt(10))
    mem.write(3, stark252FromInt(20))
    mem.write(5, stark252FromInt(30))

    let gen = CairoTraceGenerator(stepLimit: 10)
    do {
        let result = try gen.execute(
            instructions: [instr, instr, instr],
            memory: mem,
            initialState: CairoState(pc: 0, ap: 10, fp: 10)
        )

        expectEqual(result.stepCount, 3, "should execute 3 steps")

        let airGen = CairoAIRGenerator()
        let airTrace = airGen.generateAIRTrace(from: result.trace)

        // 3 rows padded to 4
        expectEqual(airTrace[0].count, 4, "AIR rows should be padded to 4")
    } catch {
        expect(false, "AIR row count test threw: \(error)")
    }
}

func testAIRTracePadding() {
    // 5 steps should pad to 8
    let instr = cairoAssertEq(
        offDst: 0, offOp0: 0, offOp1: 1,
        op1Src: .pc, resLogic: .op1, apUpdate: .addOne
    )

    let mem = CairoMemory()
    for i in 0..<5 {
        mem.write(UInt64(i * 2 + 1), stark252FromInt(UInt64(i + 1)))
    }

    let gen = CairoTraceGenerator(stepLimit: 20)
    do {
        let result = try gen.execute(
            instructions: [instr, instr, instr, instr, instr],
            memory: mem,
            initialState: CairoState(pc: 0, ap: 20, fp: 20)
        )

        let airGen = CairoAIRGenerator()
        let airTrace = airGen.generateAIRTrace(from: result.trace)

        expectEqual(airTrace[0].count, 8, "5 rows should pad to 8")
    } catch {
        expect(false, "AIR padding test threw: \(error)")
    }
}

// MARK: - Consistency Tests

func testPCAdvancesCorrectly() {
    // Two assert_eq instructions with immediates (size=2 each)
    // PC should go: 0 -> 2 -> 4
    let instr = cairoAssertEq(
        offDst: 0, offOp0: 0, offOp1: 1,
        op1Src: .pc, resLogic: .op1, apUpdate: .addOne
    )

    let mem = CairoMemory()
    mem.write(1, stark252FromInt(10))
    mem.write(3, stark252FromInt(20))

    let gen = CairoTraceGenerator(stepLimit: 10)
    do {
        let result = try gen.execute(
            instructions: [instr, instr],
            memory: mem,
            initialState: CairoState(pc: 0, ap: 10, fp: 10)
        )

        expectEqual(result.trace.count, 2, "should have 2 steps")
        expectEqual(result.trace[0].state.pc, 0, "step 0 pc should be 0")
        expectEqual(result.trace[1].state.pc, 2, "step 1 pc should be 2")
        expectEqual(result.finalState.pc, 4, "final pc should be 4")
    } catch {
        expect(false, "pc_advances threw: \(error)")
    }
}

func testAPUpdates() {
    // First instruction: ap += 1 (addOne)
    // Second instruction: ap += 1 (addOne)
    let instr = cairoAssertEq(
        offDst: 0, offOp0: 0, offOp1: 1,
        op1Src: .pc, resLogic: .op1, apUpdate: .addOne
    )

    let mem = CairoMemory()
    mem.write(1, stark252FromInt(10))
    mem.write(3, stark252FromInt(20))

    let gen = CairoTraceGenerator(stepLimit: 10)
    do {
        let result = try gen.execute(
            instructions: [instr, instr],
            memory: mem,
            initialState: CairoState(pc: 0, ap: 10, fp: 10)
        )

        expectEqual(result.trace[0].state.ap, 10, "step 0 ap should be 10")
        expectEqual(result.trace[1].state.ap, 11, "step 1 ap should be 11")
        expectEqual(result.finalState.ap, 12, "final ap should be 12")
    } catch {
        expect(false, "ap_updates threw: \(error)")
    }
}

func testAIRConsistencyCheck() {
    // Run a program and verify the AIR consistency checker finds no violations
    let instr = cairoAssertEq(
        offDst: 0, offOp0: 0, offOp1: 1,
        op1Src: .pc, resLogic: .op1, apUpdate: .addOne
    )

    let mem = CairoMemory()
    mem.write(1, stark252FromInt(5))
    mem.write(3, stark252FromInt(10))

    let gen = CairoTraceGenerator(stepLimit: 10)
    do {
        let result = try gen.execute(
            instructions: [instr, instr],
            memory: mem,
            initialState: CairoState(pc: 0, ap: 10, fp: 10)
        )

        let airGen = CairoAIRGenerator()
        let violations = airGen.checkConsistency(steps: result.trace)
        expectEqual(violations.count, 0, "should have no constraint violations, got \(violations)")

        // Verify constraint definitions exist
        expect(airGen.constraints.count > 0, "should have constraint definitions")
        expect(airGen.constraints.count >= 15, "should have at least 15 constraints")
    } catch {
        expect(false, "consistency_check threw: \(error)")
    }
}
