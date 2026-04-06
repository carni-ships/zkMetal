// GPUWitnessGenEngine — GPU-accelerated witness generation for arithmetic circuits
//
// Evaluates arithmetic circuits (add/mul/constant gates) with dependency-aware
// topological ordering and parallel evaluation of independent gate layers.
// Uses Metal compute shaders for large independent layers, CPU fallback for small ones.
//
// Architecture:
//   1. Circuit defined as a DAG of ArithGate nodes (add, mul, constant, publicInput)
//   2. Topological sort groups gates into layers of independent operations
//   3. Each layer dispatched to GPU (batch field ops) or CPU
//   4. Witness validation checks all constraints are satisfied
//
// Works with BN254 Fr field type.

import Foundation
import Metal

// MARK: - Gate Types

/// Gate types supported by the witness generation engine.
public enum ArithGateType {
    /// output = left + right (field addition)
    case add
    /// output = left * right (field multiplication)
    case mul
    /// output = constantValue (no inputs)
    case constant
    /// output = publicInputs[inputIndex]
    case publicInput
}

/// A single arithmetic gate in the circuit DAG.
/// Each gate has an output wire and references input wires by index.
public struct ArithGate {
    public let type: ArithGateType
    /// Wire index for the output of this gate
    public let output: Int
    /// Wire index for left input (unused for constant/publicInput gates)
    public let left: Int
    /// Wire index for right input (unused for constant/publicInput gates)
    public let right: Int
    /// Constant value (only used for .constant gates)
    public let constantValue: Fr
    /// Index into the public inputs array (only for .publicInput gates)
    public let inputIndex: Int

    public init(type: ArithGateType, output: Int, left: Int = 0, right: Int = 0,
                constantValue: Fr = Fr.zero, inputIndex: Int = 0) {
        self.type = type
        self.output = output
        self.left = left
        self.right = right
        self.constantValue = constantValue
        self.inputIndex = inputIndex
    }
}

/// A constraint that must hold: left * right == output (all wire indices).
public struct ArithConstraint {
    public let leftWire: Int
    public let rightWire: Int
    public let outputWire: Int
    /// Type of constraint: .mul means L*R=O, .add means L+R=O
    public let type: ArithGateType

    public init(type: ArithGateType, leftWire: Int, rightWire: Int, outputWire: Int) {
        self.type = type
        self.leftWire = leftWire
        self.rightWire = rightWire
        self.outputWire = outputWire
    }
}

/// Result of witness generation: wire assignments + metadata.
public struct WitnessResult {
    /// Wire values indexed by wire ID
    public let wires: [Fr]
    /// Number of layers in the topological sort
    public let numLayers: Int
    /// Whether GPU was used for evaluation
    public let usedGPU: Bool
}

// MARK: - GPUWitnessGenEngine

/// GPU-accelerated witness generation engine for arithmetic circuits.
///
/// Performs topological sort of circuit gates, groups independent gates
/// into parallel layers, and evaluates each layer using Metal GPU compute
/// (falling back to CPU for small layers).
public class GPUWitnessGenEngine {
    public static let version = Versions.gpuWitnessGen

    public let device: MTLDevice?
    public let commandQueue: MTLCommandQueue?

    /// Minimum layer size to dispatch to GPU (below this, use CPU)
    public var gpuLayerThreshold: Int = 32

    /// CPU-only mode (no Metal device required)
    private let cpuOnly: Bool

    /// Initialize with GPU support. Falls back to CPU-only if Metal unavailable.
    public init(forceGPU: Bool = false) throws {
        if let device = MTLCreateSystemDefaultDevice(),
           let queue = device.makeCommandQueue() {
            self.device = device
            self.commandQueue = queue
            self.cpuOnly = false
        } else if forceGPU {
            throw MSMError.noGPU
        } else {
            self.device = nil
            self.commandQueue = nil
            self.cpuOnly = true
        }
    }

    /// CPU-only initializer (never attempts GPU).
    public init(cpuOnly: Bool) {
        self.device = nil
        self.commandQueue = nil
        self.cpuOnly = true
    }

    // MARK: - Topological Sort

    /// Perform topological sort of gates, returning layers of independent gates.
    /// Each layer's gates can be evaluated in parallel since they only depend
    /// on wires computed in earlier layers.
    ///
    /// Returns: array of layers, each layer is an array of gate indices.
    public func topologicalSort(gates: [ArithGate], numWires: Int) -> [[Int]] {
        // Build dependency info: for each gate, which wires does it read?
        // For each wire, which gate produces it?
        var wireProducer = [Int: Int]()  // wire -> gate index that produces it
        for (i, gate) in gates.enumerated() {
            wireProducer[gate.output] = i
        }

        // For each gate, find which gates it depends on
        var deps = [[Int]](repeating: [], count: gates.count)
        var inDegree = [Int](repeating: 0, count: gates.count)

        for (i, gate) in gates.enumerated() {
            var depSet = Set<Int>()
            switch gate.type {
            case .add, .mul:
                if let prodL = wireProducer[gate.left], prodL != i {
                    depSet.insert(prodL)
                }
                if let prodR = wireProducer[gate.right], prodR != i {
                    depSet.insert(prodR)
                }
            case .constant, .publicInput:
                break  // no dependencies
            }
            deps[i] = Array(depSet)
            inDegree[i] = depSet.count
        }

        // BFS layer-by-layer (Kahn's algorithm)
        var layers = [[Int]]()
        var ready = [Int]()
        for i in 0..<gates.count {
            if inDegree[i] == 0 {
                ready.append(i)
            }
        }

        // Reverse map: gate -> which gates depend on it
        var dependents = [[Int]](repeating: [], count: gates.count)
        for (i, depList) in deps.enumerated() {
            for d in depList {
                dependents[d].append(i)
            }
        }

        while !ready.isEmpty {
            layers.append(ready)
            var nextReady = [Int]()
            for gateIdx in ready {
                for dependent in dependents[gateIdx] {
                    inDegree[dependent] -= 1
                    if inDegree[dependent] == 0 {
                        nextReady.append(dependent)
                    }
                }
            }
            ready = nextReady
        }

        return layers
    }

    // MARK: - Witness Generation

    /// Generate witness for an arithmetic circuit.
    ///
    /// - Parameters:
    ///   - gates: The circuit gates (DAG of arithmetic operations)
    ///   - numWires: Total number of wires in the circuit
    ///   - publicInputs: Public input values, injected via .publicInput gates
    /// - Returns: WitnessResult with all wire values
    public func generateWitness(gates: [ArithGate], numWires: Int,
                                publicInputs: [Fr] = []) throws -> WitnessResult {
        let layers = topologicalSort(gates: gates, numWires: numWires)

        var wires = [Fr](repeating: Fr.zero, count: numWires)
        var usedGPU = false

        for layer in layers {
            if !cpuOnly && layer.count >= gpuLayerThreshold {
                // GPU path for large independent layers
                try evaluateLayerGPU(gates: gates, layer: layer,
                                     wires: &wires, publicInputs: publicInputs)
                usedGPU = true
            } else {
                // CPU path
                evaluateLayerCPU(gates: gates, layer: layer,
                                 wires: &wires, publicInputs: publicInputs)
            }
        }

        return WitnessResult(wires: wires, numLayers: layers.count, usedGPU: usedGPU)
    }

    // MARK: - CPU Evaluation

    /// Evaluate a layer of independent gates on CPU.
    private func evaluateLayerCPU(gates: [ArithGate], layer: [Int],
                                  wires: inout [Fr], publicInputs: [Fr]) {
        for gateIdx in layer {
            let gate = gates[gateIdx]
            switch gate.type {
            case .add:
                wires[gate.output] = frAdd(wires[gate.left], wires[gate.right])
            case .mul:
                wires[gate.output] = frMul(wires[gate.left], wires[gate.right])
            case .constant:
                wires[gate.output] = gate.constantValue
            case .publicInput:
                if gate.inputIndex < publicInputs.count {
                    wires[gate.output] = publicInputs[gate.inputIndex]
                }
            }
        }
    }

    // MARK: - GPU Evaluation

    /// Evaluate a layer of independent gates on GPU using Metal.
    /// Packs gate info into buffers, dispatches parallel field ops.
    private func evaluateLayerGPU(gates: [ArithGate], layer: [Int],
                                  wires: inout [Fr], publicInputs: [Fr]) throws {
        guard let device = self.device, let queue = self.commandQueue else {
            evaluateLayerCPU(gates: gates, layer: layer, wires: &wires, publicInputs: publicInputs)
            return
        }

        // For GPU dispatch, we pack operations into arrays and do batch field ops.
        // Since Metal shader compilation for BN254 Fr is complex, we use a hybrid:
        // pack left/right operands, dispatch GPU add/mul, read back results.

        let count = layer.count
        let frSize = MemoryLayout<Fr>.stride

        // Separate into add and mul batches
        var addGates = [Int]()
        var mulGates = [Int]()
        for gateIdx in layer {
            switch gates[gateIdx].type {
            case .add: addGates.append(gateIdx)
            case .mul: mulGates.append(gateIdx)
            case .constant:
                wires[gates[gateIdx].output] = gates[gateIdx].constantValue
            case .publicInput:
                let g = gates[gateIdx]
                if g.inputIndex < publicInputs.count {
                    wires[g.output] = publicInputs[g.inputIndex]
                }
            }
        }

        // Process add batch on CPU (field add is cheap, GPU overhead not worth it)
        for gateIdx in addGates {
            let g = gates[gateIdx]
            wires[g.output] = frAdd(wires[g.left], wires[g.right])
        }

        // Process mul batch: if large enough, use GPU
        if mulGates.count >= gpuLayerThreshold, let device = self.device {
            // Pack left and right operands
            var lefts = [Fr]()
            var rights = [Fr]()
            lefts.reserveCapacity(mulGates.count)
            rights.reserveCapacity(mulGates.count)
            for gateIdx in mulGates {
                let g = gates[gateIdx]
                lefts.append(wires[g.left])
                rights.append(wires[g.right])
            }

            // Use GPU batch multiply
            let results = try gpuBatchMul(lefts: lefts, rights: rights, device: device, queue: queue)

            // Write results back
            for (i, gateIdx) in mulGates.enumerated() {
                wires[gates[gateIdx].output] = results[i]
            }
        } else {
            // CPU fallback for small mul batches
            for gateIdx in mulGates {
                let g = gates[gateIdx]
                wires[g.output] = frMul(wires[g.left], wires[g.right])
            }
        }
    }

    /// GPU batch field multiplication using Metal.
    private func gpuBatchMul(lefts: [Fr], rights: [Fr],
                             device: MTLDevice, queue: MTLCommandQueue) throws -> [Fr] {
        let n = lefts.count
        guard n > 0 else { return [] }

        let frSize = MemoryLayout<Fr>.stride
        let bufSize = n * frSize

        guard let leftBuf = device.makeBuffer(length: bufSize, options: .storageModeShared),
              let rightBuf = device.makeBuffer(length: bufSize, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: bufSize, options: .storageModeShared) else {
            throw MSMError.gpuError("Failed to allocate GPU buffers for batch mul")
        }

        // Copy data
        lefts.withUnsafeBytes { src in
            memcpy(leftBuf.contents(), src.baseAddress!, bufSize)
        }
        rights.withUnsafeBytes { src in
            memcpy(rightBuf.contents(), src.baseAddress!, bufSize)
        }

        // Compile shader
        let shaderSource = GPUWitnessGenEngine.batchMulShader()
        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        do {
            let library = try device.makeLibrary(source: shaderSource, options: options)
            guard let fn = library.makeFunction(name: "batch_fr_mul_witness") else {
                throw MSMError.missingKernel
            }
            let pipeline = try device.makeComputePipelineState(function: fn)

            guard let cmdBuf = queue.makeCommandBuffer() else {
                throw MSMError.noCommandBuffer
            }

            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(leftBuf, offset: 0, index: 0)
            enc.setBuffer(rightBuf, offset: 0, index: 1)
            enc.setBuffer(outBuf, offset: 0, index: 2)
            var count = UInt32(n)
            enc.setBytes(&count, length: 4, index: 3)

            let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(
                MTLSize(width: n, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            if let error = cmdBuf.error {
                throw MSMError.gpuError(error.localizedDescription)
            }

            // Read results
            var results = [Fr](repeating: Fr.zero, count: n)
            let ptr = outBuf.contents()
            results.withUnsafeMutableBytes { dst in
                memcpy(dst.baseAddress!, ptr, bufSize)
            }
            return results
        } catch {
            // Fallback to CPU on shader compilation failure
            var results = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                results[i] = frMul(lefts[i], rights[i])
            }
            return results
        }
    }

    // MARK: - Witness Validation

    /// Validate that the witness satisfies all constraints.
    ///
    /// - Parameters:
    ///   - wires: The computed witness (wire assignments)
    ///   - constraints: Constraints to check (L op R == O)
    /// - Returns: true if all constraints are satisfied
    public func validateWitness(wires: [Fr], constraints: [ArithConstraint]) -> Bool {
        for c in constraints {
            let l = wires[c.leftWire]
            let r = wires[c.rightWire]
            let o = wires[c.outputWire]

            let expected: Fr
            switch c.type {
            case .add:
                expected = frAdd(l, r)
            case .mul:
                expected = frMul(l, r)
            default:
                continue  // constant/publicInput constraints don't need validation
            }

            if expected != o {
                return false
            }
        }
        return true
    }

    // MARK: - Metal Shader

    /// BN254 Fr batch multiplication shader.
    /// Uses Montgomery multiplication matching the CPU implementation.
    private static func batchMulShader() -> String {
        return """
        #include <metal_stdlib>
        using namespace metal;

        // BN254 Fr modulus in 8x32-bit limbs (little-endian)
        constant uint FR_P[8] = {
            0xf0000001u, 0x43e1f593u, 0x79b97091u, 0x2833e848u,
            0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u
        };

        // Montgomery parameter: inv = -p^{-1} mod 2^32
        constant uint FR_INV = 0xefffffff;

        // 256-bit addition with carry
        uint add256(thread uint* r, const thread uint* a, const thread uint* b) {
            ulong carry = 0;
            for (int i = 0; i < 8; i++) {
                carry += ulong(a[i]) + ulong(b[i]);
                r[i] = uint(carry & 0xFFFFFFFF);
                carry >>= 32;
            }
            return uint(carry);
        }

        // 256-bit subtraction with borrow
        uint sub256(thread uint* r, const thread uint* a, const thread uint* b) {
            long borrow = 0;
            for (int i = 0; i < 8; i++) {
                borrow += long(a[i]) - long(b[i]);
                r[i] = uint(borrow & 0xFFFFFFFF);
                borrow >>= 32;
            }
            return uint(borrow != 0 ? 1 : 0);
        }

        // Compare a >= b
        bool gte256(const thread uint* a, constant uint* b) {
            for (int i = 7; i >= 0; i--) {
                if (a[i] > b[i]) return true;
                if (a[i] < b[i]) return false;
            }
            return true;
        }

        // Montgomery multiplication: a * b mod p
        void montMul(thread uint* result, const device uint* a, const device uint* b) {
            uint t[17] = {0};

            for (int i = 0; i < 8; i++) {
                ulong carry = 0;
                for (int j = 0; j < 8; j++) {
                    carry += ulong(t[i+j]) + ulong(a[i]) * ulong(b[j]);
                    t[i+j] = uint(carry & 0xFFFFFFFF);
                    carry >>= 32;
                }
                t[i+8] += uint(carry);

                uint m = t[i] * FR_INV;
                carry = 0;
                for (int j = 0; j < 8; j++) {
                    carry += ulong(t[i+j]) + ulong(m) * ulong(FR_P[j]);
                    t[i+j] = uint(carry & 0xFFFFFFFF);
                    carry >>= 32;
                }
                for (int j = i + 8; carry > 0 && j < 17; j++) {
                    carry += ulong(t[j]);
                    t[j] = uint(carry & 0xFFFFFFFF);
                    carry >>= 32;
                }
            }

            // Copy upper half
            for (int i = 0; i < 8; i++) {
                result[i] = t[i + 8];
            }

            // Final reduction
            if (gte256(result, FR_P)) {
                sub256(result, result, (const thread uint*)FR_P);
            }
        }

        kernel void batch_fr_mul_witness(
            device const uint* lefts  [[buffer(0)]],
            device const uint* rights [[buffer(1)]],
            device uint* outputs      [[buffer(2)]],
            constant uint& count      [[buffer(3)]],
            uint tid                  [[thread_position_in_grid]]
        ) {
            if (tid >= count) return;

            uint r[8];
            montMul(r, lefts + tid * 8, rights + tid * 8);

            for (int i = 0; i < 8; i++) {
                outputs[tid * 8 + i] = r[i];
            }
        }
        """;
    }
}
