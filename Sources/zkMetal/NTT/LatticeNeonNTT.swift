// NEON-accelerated NTT wrapper for lattice-based cryptography
// Wraps the C NEON intrinsic implementations in lattice_ntt_neon.c
//
// Two modes:
//   - Single polynomial: lattice_ntt_{kyber,dilithium}_neon (4 adjacent butterflies)
//   - Batch-4: 4 independent NTTs in 4 NEON lanes (interleaved layout)
//
// These operate on uint32_t arrays directly (no GPU, pure ARM NEON).

import Foundation
import NeonFieldOps

// MARK: - Signed-coefficient NEON NTT (standard PQC representations)

/// Kyber forward NTT via ARM NEON with signed int16 coefficients (in-place).
/// Uses vmull_s16 for 8-way SIMD Barrett reduction.
/// Coefficients must be in [0, 3329).
public func kyberNTTNeonS16(_ poly: inout [Int16], logN: Int = 8) {
    precondition(poly.count == (1 << logN))
    poly.withUnsafeMutableBufferPointer { buf in
        kyber_ntt_neon(buf.baseAddress!, Int32(logN))
    }
}

/// Kyber inverse NTT via ARM NEON with signed int16 coefficients (in-place).
public func kyberINTTNeonS16(_ poly: inout [Int16], logN: Int = 8) {
    precondition(poly.count == (1 << logN))
    poly.withUnsafeMutableBufferPointer { buf in
        kyber_intt_neon(buf.baseAddress!, Int32(logN))
    }
}

/// Dilithium forward NTT via ARM NEON with signed int32 coefficients (in-place).
/// Uses vmull_s32 for 4-way SIMD Barrett reduction.
/// Coefficients must be in [0, 8380417).
public func dilithiumNTTNeonS32(_ poly: inout [Int32], logN: Int = 8) {
    precondition(poly.count == (1 << logN))
    poly.withUnsafeMutableBufferPointer { buf in
        dilithium_ntt_neon(buf.baseAddress!, Int32(logN))
    }
}

/// Dilithium inverse NTT via ARM NEON with signed int32 coefficients (in-place).
public func dilithiumINTTNeonS32(_ poly: inout [Int32], logN: Int = 8) {
    precondition(poly.count == (1 << logN))
    poly.withUnsafeMutableBufferPointer { buf in
        dilithium_intt_neon(buf.baseAddress!, Int32(logN))
    }
}

// MARK: - Single-polynomial NEON NTT (unsigned, existing API)

/// Kyber forward NTT via ARM NEON (in-place)
public func kyberNTTNeon(_ poly: inout [UInt32]) {
    precondition(poly.count == 256)
    poly.withUnsafeMutableBufferPointer { buf in
        lattice_ntt_kyber_neon(buf.baseAddress!)
    }
}

/// Kyber inverse NTT via ARM NEON (in-place)
public func kyberINTTNeon(_ poly: inout [UInt32]) {
    precondition(poly.count == 256)
    poly.withUnsafeMutableBufferPointer { buf in
        lattice_intt_kyber_neon(buf.baseAddress!)
    }
}

/// Dilithium forward NTT via ARM NEON (in-place)
public func dilithiumNTTNeon(_ poly: inout [UInt32]) {
    precondition(poly.count == 256)
    poly.withUnsafeMutableBufferPointer { buf in
        lattice_ntt_dilithium_neon(buf.baseAddress!)
    }
}

/// Dilithium inverse NTT via ARM NEON (in-place)
public func dilithiumINTTNeon(_ poly: inout [UInt32]) {
    precondition(poly.count == 256)
    poly.withUnsafeMutableBufferPointer { buf in
        lattice_intt_dilithium_neon(buf.baseAddress!)
    }
}

// MARK: - Batch-4 NEON NTT (4 independent NTTs in 4 NEON lanes)

/// Process 4 Kyber polynomials simultaneously using NEON lane parallelism.
/// Each NEON instruction operates on all 4 polynomials at once.
/// Input: 4 arrays of 256 elements each.
/// Returns: 4 NTT-domain arrays.
public func kyberNTTNeonBatch4(_ p0: [UInt32], _ p1: [UInt32],
                                _ p2: [UInt32], _ p3: [UInt32]) -> ([UInt32], [UInt32], [UInt32], [UInt32]) {
    precondition(p0.count == 256 && p1.count == 256 && p2.count == 256 && p3.count == 256)

    var interleaved = [UInt32](repeating: 0, count: 1024)
    p0.withUnsafeBufferPointer { b0 in
        p1.withUnsafeBufferPointer { b1 in
            p2.withUnsafeBufferPointer { b2 in
                p3.withUnsafeBufferPointer { b3 in
                    interleaved.withUnsafeMutableBufferPointer { out in
                        lattice_interleave4(b0.baseAddress!, b1.baseAddress!,
                                           b2.baseAddress!, b3.baseAddress!, out.baseAddress!)
                    }
                }
            }
        }
    }

    interleaved.withUnsafeMutableBufferPointer { buf in
        lattice_ntt_kyber_batch4(buf.baseAddress!)
    }

    var r0 = [UInt32](repeating: 0, count: 256)
    var r1 = [UInt32](repeating: 0, count: 256)
    var r2 = [UInt32](repeating: 0, count: 256)
    var r3 = [UInt32](repeating: 0, count: 256)
    interleaved.withUnsafeBufferPointer { src in
        r0.withUnsafeMutableBufferPointer { b0 in
            r1.withUnsafeMutableBufferPointer { b1 in
                r2.withUnsafeMutableBufferPointer { b2 in
                    r3.withUnsafeMutableBufferPointer { b3 in
                        lattice_deinterleave4(src.baseAddress!,
                                              b0.baseAddress!, b1.baseAddress!,
                                              b2.baseAddress!, b3.baseAddress!)
                    }
                }
            }
        }
    }
    return (r0, r1, r2, r3)
}

/// Process 4 Kyber inverse NTTs simultaneously.
public func kyberINTTNeonBatch4(_ p0: [UInt32], _ p1: [UInt32],
                                 _ p2: [UInt32], _ p3: [UInt32]) -> ([UInt32], [UInt32], [UInt32], [UInt32]) {
    precondition(p0.count == 256 && p1.count == 256 && p2.count == 256 && p3.count == 256)

    var interleaved = [UInt32](repeating: 0, count: 1024)
    p0.withUnsafeBufferPointer { b0 in
        p1.withUnsafeBufferPointer { b1 in
            p2.withUnsafeBufferPointer { b2 in
                p3.withUnsafeBufferPointer { b3 in
                    interleaved.withUnsafeMutableBufferPointer { out in
                        lattice_interleave4(b0.baseAddress!, b1.baseAddress!,
                                           b2.baseAddress!, b3.baseAddress!, out.baseAddress!)
                    }
                }
            }
        }
    }

    interleaved.withUnsafeMutableBufferPointer { buf in
        lattice_intt_kyber_batch4(buf.baseAddress!)
    }

    var r0 = [UInt32](repeating: 0, count: 256)
    var r1 = [UInt32](repeating: 0, count: 256)
    var r2 = [UInt32](repeating: 0, count: 256)
    var r3 = [UInt32](repeating: 0, count: 256)
    interleaved.withUnsafeBufferPointer { src in
        r0.withUnsafeMutableBufferPointer { b0 in
            r1.withUnsafeMutableBufferPointer { b1 in
                r2.withUnsafeMutableBufferPointer { b2 in
                    r3.withUnsafeMutableBufferPointer { b3 in
                        lattice_deinterleave4(src.baseAddress!,
                                              b0.baseAddress!, b1.baseAddress!,
                                              b2.baseAddress!, b3.baseAddress!)
                    }
                }
            }
        }
    }
    return (r0, r1, r2, r3)
}

/// Process 4 Dilithium NTTs simultaneously.
public func dilithiumNTTNeonBatch4(_ p0: [UInt32], _ p1: [UInt32],
                                    _ p2: [UInt32], _ p3: [UInt32]) -> ([UInt32], [UInt32], [UInt32], [UInt32]) {
    precondition(p0.count == 256 && p1.count == 256 && p2.count == 256 && p3.count == 256)

    var interleaved = [UInt32](repeating: 0, count: 1024)
    p0.withUnsafeBufferPointer { b0 in
        p1.withUnsafeBufferPointer { b1 in
            p2.withUnsafeBufferPointer { b2 in
                p3.withUnsafeBufferPointer { b3 in
                    interleaved.withUnsafeMutableBufferPointer { out in
                        lattice_interleave4(b0.baseAddress!, b1.baseAddress!,
                                           b2.baseAddress!, b3.baseAddress!, out.baseAddress!)
                    }
                }
            }
        }
    }

    interleaved.withUnsafeMutableBufferPointer { buf in
        lattice_ntt_dilithium_batch4(buf.baseAddress!)
    }

    var r0 = [UInt32](repeating: 0, count: 256)
    var r1 = [UInt32](repeating: 0, count: 256)
    var r2 = [UInt32](repeating: 0, count: 256)
    var r3 = [UInt32](repeating: 0, count: 256)
    interleaved.withUnsafeBufferPointer { src in
        r0.withUnsafeMutableBufferPointer { b0 in
            r1.withUnsafeMutableBufferPointer { b1 in
                r2.withUnsafeMutableBufferPointer { b2 in
                    r3.withUnsafeMutableBufferPointer { b3 in
                        lattice_deinterleave4(src.baseAddress!,
                                              b0.baseAddress!, b1.baseAddress!,
                                              b2.baseAddress!, b3.baseAddress!)
                    }
                }
            }
        }
    }
    return (r0, r1, r2, r3)
}

/// Process 4 Dilithium inverse NTTs simultaneously.
public func dilithiumINTTNeonBatch4(_ p0: [UInt32], _ p1: [UInt32],
                                     _ p2: [UInt32], _ p3: [UInt32]) -> ([UInt32], [UInt32], [UInt32], [UInt32]) {
    precondition(p0.count == 256 && p1.count == 256 && p2.count == 256 && p3.count == 256)

    var interleaved = [UInt32](repeating: 0, count: 1024)
    p0.withUnsafeBufferPointer { b0 in
        p1.withUnsafeBufferPointer { b1 in
            p2.withUnsafeBufferPointer { b2 in
                p3.withUnsafeBufferPointer { b3 in
                    interleaved.withUnsafeMutableBufferPointer { out in
                        lattice_interleave4(b0.baseAddress!, b1.baseAddress!,
                                           b2.baseAddress!, b3.baseAddress!, out.baseAddress!)
                    }
                }
            }
        }
    }

    interleaved.withUnsafeMutableBufferPointer { buf in
        lattice_intt_dilithium_batch4(buf.baseAddress!)
    }

    var r0 = [UInt32](repeating: 0, count: 256)
    var r1 = [UInt32](repeating: 0, count: 256)
    var r2 = [UInt32](repeating: 0, count: 256)
    var r3 = [UInt32](repeating: 0, count: 256)
    interleaved.withUnsafeBufferPointer { src in
        r0.withUnsafeMutableBufferPointer { b0 in
            r1.withUnsafeMutableBufferPointer { b1 in
                r2.withUnsafeMutableBufferPointer { b2 in
                    r3.withUnsafeMutableBufferPointer { b3 in
                        lattice_deinterleave4(src.baseAddress!,
                                              b0.baseAddress!, b1.baseAddress!,
                                              b2.baseAddress!, b3.baseAddress!)
                    }
                }
            }
        }
    }
    return (r0, r1, r2, r3)
}
