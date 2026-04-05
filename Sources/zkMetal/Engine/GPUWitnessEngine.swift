// GPUWitnessEngine — GPU-accelerated circuit witness generation
//
// Computes all intermediate wire values in a circuit from inputs.
// Uses layer-based parallelism: gates are topologically sorted by dependency
// level, and each level is dispatched as one GPU call.
//
// For circuits < 1024 gates, falls back to CPU (GPU dispatch overhead dominates).

import Foundation
import Metal

// MARK: - Witness Circuit Types

/// Operations that a witness gate can perform.
public enum WitnessOp: Equatable {
    case add
    case mul
    case linearCombination(qL: Fr, qR: Fr, qC: Fr)
    case poseidon2  // full Poseidon2 round on t=3 state (leftIdx = state base)
    case constant(Fr)
    case copy
}

/// A single gate in a witness circuit.
/// For most ops: output[outIdx] = op(wires[leftIdx], wires[rightIdx])
/// For constant: output[outIdx] = constant value (leftIdx/rightIdx ignored)
/// For copy: output[outIdx] = wires[leftIdx] (rightIdx ignored)
/// For poseidon2: reads 3 wires starting at leftIdx, writes 3 starting at outIdx,
///   rightIdx encodes round constant index, uses full round S-box.
public struct WitnessGate: Equatable {
    public let op: WitnessOp
    public let leftIdx: Int
    public let rightIdx: Int
    public let outIdx: Int

    public init(op: WitnessOp, leftIdx: Int, rightIdx: Int, outIdx: Int) {
        self.op = op
        self.leftIdx = leftIdx
        self.rightIdx = rightIdx
        self.outIdx = outIdx
    }
}

/// A circuit for witness generation: topologically sorted gates with dependency info.
public struct WitnessCircuit {
    /// Total number of wires (including inputs and outputs).
    public let numWires: Int
    /// Number of input wires (indices 0..<numInputs are inputs).
    public let numInputs: Int
    /// Gates in topological order. Each gate's inputs must come from
    /// either input wires or outputs of earlier gates.
    public let gates: [WitnessGate]
    /// Layer assignment for each gate. Gates in the same layer are independent.
    /// Layer 0 = gates that only depend on inputs, layer 1 = depends on layer 0, etc.
    public let layers: [Int]
    /// Number of distinct layers.
    public let numLayers: Int

    public init(numWires: Int, numInputs: Int, gates: [WitnessGate]) {
        self.numWires = numWires
        self.numInputs = numInputs
        self.gates = gates

        // Compute layer assignments by analyzing dependencies
        var wireReady = [Int](repeating: -1, count: numWires)  // layer when wire becomes available
        for i in 0..<numInputs {
            wireReady[i] = -1  // inputs available before any layer
        }

        var layerAssignment = [Int](repeating: 0, count: gates.count)
        var maxLayer = 0

        for (gi, gate) in gates.enumerated() {
            var depLayer = -1  // max layer of dependencies

            switch gate.op {
            case .constant:
                depLayer = -1  // no dependencies
            case .copy:
                depLayer = wireReady[gate.leftIdx]
            case .poseidon2:
                // Reads 3 wires starting at leftIdx
                for offset in 0..<3 {
                    let idx = gate.leftIdx + offset
                    if idx < numWires {
                        depLayer = max(depLayer, wireReady[idx])
                    }
                }
            default:
                depLayer = max(wireReady[gate.leftIdx], wireReady[gate.rightIdx])
            }

            let myLayer = depLayer + 1
            layerAssignment[gi] = myLayer
            maxLayer = max(maxLayer, myLayer)

            // Mark output wire(s) as available after this layer
            switch gate.op {
            case .poseidon2:
                for offset in 0..<3 {
                    let idx = gate.outIdx + offset
                    if idx < numWires {
                        wireReady[idx] = myLayer
                    }
                }
            default:
                wireReady[gate.outIdx] = myLayer
            }
        }

        self.layers = layerAssignment
        self.numLayers = maxLayer + 1
    }
}

// MARK: - GPU Witness Engine

/// GPU-accelerated witness generation engine.
/// Dispatches each dependency layer as a GPU compute pass.
/// Falls back to CPU for small circuits (< 1024 gates).
public class GPUWitnessEngine {
    public static let version = Versions.witness

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Pipeline states for each BN254 kernel
    private let addPipeline: MTLComputePipelineState
    private let mulPipeline: MTLComputePipelineState
    private let linearComboPipeline: MTLComputePipelineState
    private let poseidon2Pipeline: MTLComputePipelineState

    /// CPU fallback threshold: circuits with fewer gates than this use CPU.
    public var cpuFallbackThreshold: Int = 1024

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUWitnessEngine.compileShaders(device: device)

        guard let addFn = library.makeFunction(name: "witness_add_bn254"),
              let mulFn = library.makeFunction(name: "witness_mul_bn254"),
              let lcFn = library.makeFunction(name: "witness_linear_combination_bn254"),
              let p2Fn = library.makeFunction(name: "witness_poseidon2_round_bn254") else {
            throw MSMError.missingKernel
        }

        self.addPipeline = try device.makeComputePipelineState(function: addFn)
        self.mulPipeline = try device.makeComputePipelineState(function: mulFn)
        self.linearComboPipeline = try device.makeComputePipelineState(function: lcFn)
        self.poseidon2Pipeline = try device.makeComputePipelineState(function: p2Fn)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let bbSource = try String(contentsOfFile: shaderDir + "/fields/babybear.metal", encoding: .utf8)
        let witnessSource = try String(contentsOfFile: shaderDir + "/witness/witness_compute.metal", encoding: .utf8)

        // Strip include guards for inline compilation
        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let bbClean = bbSource
            .replacingOccurrences(of: "#ifndef BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#define BABYBEAR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BABYBEAR_METAL", with: "")

        let witnessClean = witnessSource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let combined = frClean + "\n" + bbClean + "\n" + witnessClean

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - Public API

    /// Generate all wire values for a circuit given inputs.
    ///
    /// - Parameters:
    ///   - circuit: The witness circuit (topologically sorted gates with layer info)
    ///   - inputs: Input wire values (must have circuit.numInputs elements)
    /// - Returns: Array of all wire values (length = circuit.numWires)
    public func generateWitness(circuit: WitnessCircuit, inputs: [Fr]) -> [Fr] {
        precondition(inputs.count == circuit.numInputs,
                     "Expected \(circuit.numInputs) inputs, got \(inputs.count)")

        if circuit.gates.count < cpuFallbackThreshold {
            return generateWitnessCPU(circuit: circuit, inputs: inputs)
        }
        return generateWitnessGPU(circuit: circuit, inputs: inputs)
    }

    // MARK: - CPU Fallback

    /// CPU witness generation for small circuits.
    public func generateWitnessCPU(circuit: WitnessCircuit, inputs: [Fr]) -> [Fr] {
        var wires = [Fr](repeating: Fr.zero, count: circuit.numWires)

        // Copy inputs
        for i in 0..<inputs.count {
            wires[i] = inputs[i]
        }

        // Evaluate gates in topological order
        for gate in circuit.gates {
            switch gate.op {
            case .add:
                wires[gate.outIdx] = frAdd(wires[gate.leftIdx], wires[gate.rightIdx])
            case .mul:
                wires[gate.outIdx] = frMul(wires[gate.leftIdx], wires[gate.rightIdx])
            case .linearCombination(let qL, let qR, let qC):
                let t1 = frMul(qL, wires[gate.leftIdx])
                let t2 = frMul(qR, wires[gate.rightIdx])
                wires[gate.outIdx] = frAdd(frAdd(t1, t2), qC)
            case .poseidon2:
                // Read 3-element state, apply one full Poseidon2 round, write back
                let base = gate.leftIdx
                var s0 = wires[base]
                var s1 = wires[base + 1]
                var s2 = wires[base + 2]

                // S-box x^5 on all (full round)
                let t0 = frMul(frSqr(frSqr(s0)), s0)
                let t1 = frMul(frSqr(frSqr(s1)), s1)
                let t2 = frMul(frSqr(frSqr(s2)), s2)
                s0 = t0; s1 = t1; s2 = t2

                // MDS: t = s0+s1+s2, s_i += t
                let sum = frAdd(frAdd(s0, s1), s2)
                s0 = frAdd(s0, sum)
                s1 = frAdd(s1, sum)
                s2 = frAdd(s2, sum)

                wires[gate.outIdx] = s0
                if gate.outIdx + 1 < circuit.numWires { wires[gate.outIdx + 1] = s1 }
                if gate.outIdx + 2 < circuit.numWires { wires[gate.outIdx + 2] = s2 }
            case .constant(let val):
                wires[gate.outIdx] = val
            case .copy:
                wires[gate.outIdx] = wires[gate.leftIdx]
            }
        }

        return wires
    }

    // MARK: - GPU Witness Generation

    /// GPU witness generation: dispatch each layer as GPU compute passes.
    private func generateWitnessGPU(circuit: WitnessCircuit, inputs: [Fr]) -> [Fr] {
        let frStride = MemoryLayout<Fr>.stride
        let u32Stride = MemoryLayout<UInt32>.stride

        // Allocate wire buffer on GPU
        let wireBytes = circuit.numWires * frStride
        guard let wireBuf = device.makeBuffer(length: wireBytes, options: .storageModeShared) else {
            return generateWitnessCPU(circuit: circuit, inputs: inputs)
        }

        // Copy inputs into wire buffer
        let wirePtr = wireBuf.contents().bindMemory(to: Fr.self, capacity: circuit.numWires)
        for i in 0..<inputs.count {
            wirePtr[i] = inputs[i]
        }

        // Group gates by layer and operation type
        var layerGates = [[Int]](repeating: [], count: circuit.numLayers)
        for (gi, _) in circuit.gates.enumerated() {
            layerGates[circuit.layers[gi]].append(gi)
        }

        // Process each layer
        for layer in 0..<circuit.numLayers {
            let gateIndices = layerGates[layer]
            if gateIndices.isEmpty { continue }

            // Sub-group by operation type for batched dispatch
            var addGates = [Int]()
            var mulGates = [Int]()
            var lcGates = [Int]()
            var poseidonGates = [Int]()
            var otherGates = [Int]()

            for gi in gateIndices {
                switch circuit.gates[gi].op {
                case .add:      addGates.append(gi)
                case .mul:      mulGates.append(gi)
                case .linearCombination: lcGates.append(gi)
                case .poseidon2: poseidonGates.append(gi)
                default:        otherGates.append(gi)
                }
            }

            // Handle constant/copy gates on CPU (typically few per layer)
            for gi in otherGates {
                let gate = circuit.gates[gi]
                switch gate.op {
                case .constant(let val):
                    wirePtr[gate.outIdx] = val
                case .copy:
                    wirePtr[gate.outIdx] = wirePtr[gate.leftIdx]
                default:
                    break
                }
            }

            // Dispatch GPU kernels for each operation type in this layer
            guard let cmdBuf = commandQueue.makeCommandBuffer() else {
                return generateWitnessCPU(circuit: circuit, inputs: inputs)
            }

            var didEncode = false

            // ADD gates
            if !addGates.isEmpty {
                if let result = encodeOpGates(cmdBuf: cmdBuf, wireBuf: wireBuf,
                                               gates: addGates, circuit: circuit,
                                               pipeline: addPipeline) {
                    didEncode = true
                    let _ = result
                }
            }

            // MUL gates
            if !mulGates.isEmpty {
                if let result = encodeOpGates(cmdBuf: cmdBuf, wireBuf: wireBuf,
                                               gates: mulGates, circuit: circuit,
                                               pipeline: mulPipeline) {
                    didEncode = true
                    let _ = result
                }
            }

            // LINEAR COMBINATION gates
            if !lcGates.isEmpty {
                if let _ = encodeLCGates(cmdBuf: cmdBuf, wireBuf: wireBuf,
                                          gates: lcGates, circuit: circuit) {
                    didEncode = true
                }
            }

            // POSEIDON2 gates
            if !poseidonGates.isEmpty {
                if let _ = encodePoseidon2Gates(cmdBuf: cmdBuf, wireBuf: wireBuf,
                                                 gates: poseidonGates, circuit: circuit) {
                    didEncode = true
                }
            }

            if didEncode {
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
            }
        }

        // Read results
        var wires = [Fr](repeating: Fr.zero, count: circuit.numWires)
        for i in 0..<circuit.numWires {
            wires[i] = wirePtr[i]
        }
        return wires
    }

    // MARK: - GPU Encoding Helpers

    /// Encode add/mul gates (same signature) into a command buffer.
    private func encodeOpGates(cmdBuf: MTLCommandBuffer, wireBuf: MTLBuffer,
                                gates: [Int], circuit: WitnessCircuit,
                                pipeline: MTLComputePipelineState) -> Bool? {
        let count = gates.count
        let u32Stride = MemoryLayout<UInt32>.stride

        var leftIndices = [UInt32](repeating: 0, count: count)
        var rightIndices = [UInt32](repeating: 0, count: count)
        var outIndices = [UInt32](repeating: 0, count: count)

        for (i, gi) in gates.enumerated() {
            let gate = circuit.gates[gi]
            leftIndices[i] = UInt32(gate.leftIdx)
            rightIndices[i] = UInt32(gate.rightIdx)
            outIndices[i] = UInt32(gate.outIdx)
        }

        guard let leftBuf = device.makeBuffer(bytes: leftIndices, length: count * u32Stride, options: .storageModeShared),
              let rightBuf = device.makeBuffer(bytes: rightIndices, length: count * u32Stride, options: .storageModeShared),
              let outBuf = device.makeBuffer(bytes: outIndices, length: count * u32Stride, options: .storageModeShared) else {
            return nil
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(wireBuf, offset: 0, index: 0)
        enc.setBuffer(leftBuf, offset: 0, index: 1)
        enc.setBuffer(rightBuf, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        var numGates = UInt32(count)
        enc.setBytes(&numGates, length: 4, index: 4)

        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
        )
        enc.endEncoding()
        return true
    }

    /// Encode linear combination gates.
    private func encodeLCGates(cmdBuf: MTLCommandBuffer, wireBuf: MTLBuffer,
                                gates: [Int], circuit: WitnessCircuit) -> Bool? {
        let count = gates.count
        let u32Stride = MemoryLayout<UInt32>.stride
        let frStride = MemoryLayout<Fr>.stride

        var leftIndices = [UInt32](repeating: 0, count: count)
        var rightIndices = [UInt32](repeating: 0, count: count)
        var outIndices = [UInt32](repeating: 0, count: count)
        var constants = [Fr](repeating: Fr.zero, count: count * 3)

        for (i, gi) in gates.enumerated() {
            let gate = circuit.gates[gi]
            leftIndices[i] = UInt32(gate.leftIdx)
            rightIndices[i] = UInt32(gate.rightIdx)
            outIndices[i] = UInt32(gate.outIdx)

            if case .linearCombination(let qL, let qR, let qC) = gate.op {
                constants[i * 3] = qL
                constants[i * 3 + 1] = qR
                constants[i * 3 + 2] = qC
            }
        }

        guard let leftBuf = device.makeBuffer(bytes: leftIndices, length: count * u32Stride, options: .storageModeShared),
              let rightBuf = device.makeBuffer(bytes: rightIndices, length: count * u32Stride, options: .storageModeShared),
              let outBuf = device.makeBuffer(bytes: outIndices, length: count * u32Stride, options: .storageModeShared),
              let constBuf = device.makeBuffer(bytes: constants, length: count * 3 * frStride, options: .storageModeShared) else {
            return nil
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(linearComboPipeline)
        enc.setBuffer(wireBuf, offset: 0, index: 0)
        enc.setBuffer(leftBuf, offset: 0, index: 1)
        enc.setBuffer(rightBuf, offset: 0, index: 2)
        enc.setBuffer(outBuf, offset: 0, index: 3)
        enc.setBuffer(constBuf, offset: 0, index: 4)
        var numGates = UInt32(count)
        enc.setBytes(&numGates, length: 4, index: 5)

        let tg = min(256, Int(linearComboPipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
        )
        enc.endEncoding()
        return true
    }

    /// Encode Poseidon2 round gates.
    private func encodePoseidon2Gates(cmdBuf: MTLCommandBuffer, wireBuf: MTLBuffer,
                                       gates: [Int], circuit: WitnessCircuit) -> Bool? {
        // For Poseidon2 gates, all instances in one dispatch share the same round constants
        // (zero round constants for basic witness gen -- round constants are baked into the circuit).
        // Each gate operates on a 3-element state block.
        let count = gates.count
        let frStride = MemoryLayout<Fr>.stride

        // Use zero round constants (the circuit builder should incorporate them into prior gates)
        var roundConsts = [Fr](repeating: Fr.zero, count: 3)
        guard let rcBuf = device.makeBuffer(bytes: roundConsts, length: 3 * frStride, options: .storageModeShared) else {
            return nil
        }

        // Dispatch each poseidon2 gate individually (they may have different state bases)
        for gi in gates {
            let gate = circuit.gates[gi]
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(poseidon2Pipeline)
            enc.setBuffer(wireBuf, offset: 0, index: 0)
            enc.setBuffer(rcBuf, offset: 0, index: 1)
            var stateBase = UInt32(gate.leftIdx)
            var numInstances: UInt32 = 1
            var fullRound: UInt32 = 1
            enc.setBytes(&stateBase, length: 4, index: 2)
            enc.setBytes(&numInstances, length: 4, index: 3)
            enc.setBytes(&fullRound, length: 4, index: 4)

            enc.dispatchThreads(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
            )
            enc.endEncoding()
        }

        return true
    }

    // MARK: - Circuit Builder Helpers

    /// Create a simple arithmetic circuit from a list of operations.
    /// Automatically computes layer assignments.
    public static func buildCircuit(numWires: Int, numInputs: Int, gates: [WitnessGate]) -> WitnessCircuit {
        return WitnessCircuit(numWires: numWires, numInputs: numInputs, gates: gates)
    }
}
