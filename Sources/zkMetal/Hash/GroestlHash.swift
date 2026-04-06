// Groestl-256 Hash — CPU reference + GPU engine for batch hashing
// Groestl is a ZK-friendly hash used by Binius for binary-field Merkle commitments.
// Spec: Groestl-256 with 512-bit state, 10 rounds, AES-based permutations P and Q.
import Foundation
import Metal

// MARK: - AES S-box

private let SBOX: [UInt8] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
]

// MARK: - GF(2^8) multiplication for MixBytes

/// Multiply by 2 in GF(2^8) with irreducible polynomial x^8 + x^4 + x^3 + x + 1 (0x1b)
@inline(__always)
private func gf2Mul2(_ x: UInt8) -> UInt8 {
    let shifted = x << 1
    return x & 0x80 != 0 ? shifted ^ 0x1b : shifted
}

/// Multiply by arbitrary value in GF(2^8) via repeated doubling
@inline(__always)
private func gfMul(_ a: UInt8, _ b: UInt8) -> UInt8 {
    var result: UInt8 = 0
    var aa = a
    var bb = b
    while bb != 0 {
        if bb & 1 != 0 { result ^= aa }
        aa = gf2Mul2(aa)
        bb >>= 1
    }
    return result
}

// MARK: - Groestl-256 state: 8x8 byte matrix stored column-major

/// State is 8 columns x 8 rows = 64 bytes, stored as state[col][row]
private typealias GroestlState = [[UInt8]]

private func makeState() -> GroestlState {
    [[UInt8]](repeating: [UInt8](repeating: 0, count: 8), count: 8)
}

private func bytesToState(_ bytes: [UInt8]) -> GroestlState {
    var s = makeState()
    for i in 0..<64 {
        // Column-major: byte index i -> column i/8, row i%8
        s[i / 8][i % 8] = bytes[i]
    }
    return s
}

private func stateToBytes(_ s: GroestlState) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: 64)
    for i in 0..<64 {
        out[i] = s[i / 8][i % 8]
    }
    return out
}

// MARK: - AddRoundConstant

/// P permutation round constants: XOR byte (round << 4) | col to row 0
private func addRoundConstantP(_ s: inout GroestlState, round: Int) {
    for col in 0..<8 {
        s[col][0] ^= UInt8((col << 4) ^ round)
    }
}

/// Q permutation round constants
private func addRoundConstantQ(_ s: inout GroestlState, round: Int) {
    for col in 0..<8 {
        // Q constants: XOR 0xff to all rows, then XOR specific pattern to last row
        for row in 0..<7 {
            s[col][row] ^= 0xff
        }
        s[col][7] ^= UInt8(0xff ^ ((col << 4) ^ round))
    }
}

// MARK: - SubBytes

private func subBytes(_ s: inout GroestlState) {
    for col in 0..<8 {
        for row in 0..<8 {
            s[col][row] = SBOX[Int(s[col][row])]
        }
    }
}

// MARK: - ShiftBytes

/// P permutation shift amounts for each row
private let SHIFT_P: [Int] = [0, 1, 2, 3, 4, 5, 6, 7]
/// Q permutation shift amounts for each row
private let SHIFT_Q: [Int] = [1, 3, 5, 7, 0, 2, 4, 6]

private func shiftBytes(_ s: inout GroestlState, shifts: [Int]) {
    for row in 0..<8 {
        let shift = shifts[row]
        if shift == 0 { continue }
        var tmp = [UInt8](repeating: 0, count: 8)
        for col in 0..<8 {
            tmp[(col + 8 - shift) % 8] = s[col][row]
        }
        for col in 0..<8 {
            s[col][row] = tmp[col]
        }
    }
}

// MARK: - MixBytes

/// Groestl MixBytes: multiply each column by a circulant matrix in GF(2^8)
/// The matrix row is [2, 2, 3, 4, 5, 3, 5, 7] circularly shifted
private let MIX_MATRIX: [[UInt8]] = {
    let row: [UInt8] = [2, 2, 3, 4, 5, 3, 5, 7]
    var m = [[UInt8]]()
    for i in 0..<8 {
        var r = [UInt8](repeating: 0, count: 8)
        for j in 0..<8 {
            r[j] = row[(j + 8 - i) % 8]
        }
        m.append(r)
    }
    return m
}()

private func mixBytes(_ s: inout GroestlState) {
    for col in 0..<8 {
        var newCol = [UInt8](repeating: 0, count: 8)
        for row in 0..<8 {
            var acc: UInt8 = 0
            for k in 0..<8 {
                acc ^= gfMul(MIX_MATRIX[row][k], s[col][k])
            }
            newCol[row] = acc
        }
        s[col] = newCol
    }
}

// MARK: - P and Q permutations

private func permutationP(_ input: [UInt8]) -> [UInt8] {
    var s = bytesToState(input)
    for round in 0..<10 {
        addRoundConstantP(&s, round: round)
        subBytes(&s)
        shiftBytes(&s, shifts: SHIFT_P)
        mixBytes(&s)
    }
    return stateToBytes(s)
}

private func permutationQ(_ input: [UInt8]) -> [UInt8] {
    var s = bytesToState(input)
    for round in 0..<10 {
        addRoundConstantQ(&s, round: round)
        subBytes(&s)
        shiftBytes(&s, shifts: SHIFT_Q)
        mixBytes(&s)
    }
    return stateToBytes(s)
}

// MARK: - Compression function

/// Groestl compression: h' = P(h XOR m) XOR Q(m) XOR h
private func compress(_ h: [UInt8], _ m: [UInt8]) -> [UInt8] {
    // h XOR m
    var hxm = [UInt8](repeating: 0, count: 64)
    for i in 0..<64 { hxm[i] = h[i] ^ m[i] }

    let ph = permutationP(hxm)
    let qm = permutationQ(m)

    var result = [UInt8](repeating: 0, count: 64)
    for i in 0..<64 {
        result[i] = ph[i] ^ qm[i] ^ h[i]
    }
    return result
}

// MARK: - Output transformation

/// Omega(h) = truncate_256( P(h) XOR h )
private func outputTransformation(_ h: [UInt8]) -> [UInt8] {
    let ph = permutationP(h)
    var full = [UInt8](repeating: 0, count: 64)
    for i in 0..<64 {
        full[i] = ph[i] ^ h[i]
    }
    // Truncate: take last 32 bytes (bytes 32..63)
    return Array(full[32..<64])
}

// MARK: - Groestl-256 padding

/// Pad message: append bit 1 (0x80), then zeros, then 64-bit block count (big-endian)
/// Block size is 64 bytes for Groestl-256
private func padMessage(_ data: [UInt8]) -> [UInt8] {
    var msg = data
    // Number of blocks after padding (including the final block with length)
    // We need: msg.count + 1 (0x80 byte) + 8 (length) padded to multiple of 64
    let msgLen = msg.count
    msg.append(0x80)

    // Pad with zeros until we have room for 8-byte length and are at block boundary
    while (msg.count + 8) % 64 != 0 {
        msg.append(0x00)
    }

    // Append 64-bit block count (big-endian) - number of message blocks including padding block(s)
    let numBlocks = UInt64(msg.count / 64 + 1)  // +1 for IV block
    for i in (0..<8).reversed() {
        msg.append(UInt8((numBlocks >> (i * 8)) & 0xff))
    }

    return msg
}

// MARK: - Public API

/// Groestl-256 hash of arbitrary-length input, returns 32 bytes.
public func groestl256(_ data: [UInt8]) -> [UInt8] {
    // IV: first byte = 0x01 at position 62 (output length 256 = 0x0100, big-endian in last 2 bytes of state)
    var h = [UInt8](repeating: 0, count: 64)
    h[62] = 0x01  // encode output length 256 in bits as big-endian 16-bit at end
    h[63] = 0x00

    // Pad message
    let padded = padMessage(data)

    // Process each 64-byte block
    let numBlocks = padded.count / 64
    for i in 0..<numBlocks {
        let block = Array(padded[i * 64 ..< (i + 1) * 64])
        h = compress(h, block)
    }

    // Output transformation
    return outputTransformation(h)
}

// MARK: - Groestl-256 GPU Engine

public class Groestl256Engine {
    public static let version = Versions.groestl256
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let hashBatchFunction: MTLComputePipelineState

    private var cachedInputBuf: MTLBuffer?
    private var cachedOutputBuf: MTLBuffer?
    private var cachedCount: Int = 0
    private let tuning: TuningConfig

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try Groestl256Engine.compileShaders(device: device)
        guard let hashBatchFn = library.makeFunction(name: "groestl256_hash_batch") else {
            throw MSMError.missingKernel
        }
        self.hashBatchFunction = try device.makeComputePipelineState(function: hashBatchFn)
        self.tuning = TuningManager.shared.config(device: device)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let source = try String(contentsOfFile: shaderDir + "/hash/groestl256.metal", encoding: .utf8)
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: source, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("hash/groestl256.metal").path
                if FileManager.default.fileExists(atPath: path) { return url.path }
            }
        }
        let candidates = [
            "\(execDir)/../Sources/Shaders",
            "./Sources/Shaders",
        ]
        for path in candidates {
            if FileManager.default.fileExists(atPath: "\(path)/hash/groestl256.metal") { return path }
        }
        return "./Sources/Shaders"
    }

    /// Batch Groestl-256 hash of fixed 64-byte inputs.
    /// Input: n * 64 bytes, Output: n * 32 bytes
    public func hashBatch(_ input: [UInt8]) throws -> [UInt8] {
        precondition(input.count % 64 == 0)
        let n = input.count / 64

        if n > cachedCount {
            guard let inBuf = device.makeBuffer(length: n * 64, options: .storageModeShared),
                  let outBuf = device.makeBuffer(length: n * 32, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to allocate buffers")
            }
            cachedInputBuf = inBuf
            cachedOutputBuf = outBuf
            cachedCount = n
        }

        let inputBuf = cachedInputBuf!
        let outputBuf = cachedOutputBuf!
        input.withUnsafeBytes { src in
            memcpy(inputBuf.contents(), src.baseAddress!, input.count)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(hashBatchFunction)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        var countVal = UInt32(n)
        enc.setBytes(&countVal, length: 4, index: 2)
        let tg = min(tuning.hashThreadgroupSize, Int(hashBatchFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        let ptr = outputBuf.contents().bindMemory(to: UInt8.self, capacity: n * 32)
        return Array(UnsafeBufferPointer(start: ptr, count: n * 32))
    }
}
