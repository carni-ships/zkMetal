// Precomputed Polynomial Manager — Manages Vanishing and Subspace Polynomials
//
// Manages lifecycle of precomputed polynomials: vanishing polynomials,
// subspace polynomials, and challenge-dependent adjustments. Provides
// thread-safe access and automatic cache invalidation.
//
// Reference: FRI-Binius flattened-polynomial + subspace vanishing tricks

import Foundation

// MARK: - Precomputation State

/// Current state of precomputation.
public enum PrecomputationState: Equatable {
    /// Not yet computed.
    case uninitialized

    /// Currently computing.
    case computing(progress: Double)

    /// Ready for use.
    case ready

    /// Invalidated, needs recomputation.
    case invalidated

    /// Computation failed with error.
    case failed(String)
}

// MARK: - Challenge Context

/// Context for challenge-dependent polynomial adjustments.
/// When a new challenge arrives, only O(1) adjustments are needed.
public struct ChallengeContext {
    /// The challenge value.
    public let challenge: UInt8

    /// Precomputed powers: r, r^2, r^4, r^8, ...
    public let powers: [UInt8]

    /// Create context from challenge.
    public init(challenge: UInt8, maxPower: Int = 32) {
        self.challenge = challenge
        self.powers = Self.computePowers(challenge, maxPower: maxPower)
    }

    /// Compute powers using repeated squaring.
    private static func computePowers(_ base: UInt8, maxPower: Int) -> [UInt8] {
        var powers = [UInt8](repeating: 0, count: maxPower)
        powers[0] = base
        for i in 1..<maxPower {
            powers[i] = gf28Mul(powers[i - 1], powers[i - 1])
        }
        return powers
    }

    /// Get power r^{2^i}.
    public func getPower(_ i: Int) -> UInt8 {
        guard i < powers.count else { return 0 }
        return powers[i]
    }

    /// GF(2^8) multiplication.
    private static func gf28Mul(_ a: UInt8, _ b: UInt8) -> UInt8 {
        var p: UInt16 = 0
        var aa = UInt16(a)
        var bb = UInt16(b)
        for _ in 0..<8 {
            if bb & 1 != 0 {
                p ^= aa
            }
            let hiBitSet = (aa & 0x80) != 0
            aa <<= 1
            if hiBitSet {
                aa ^= 0x11B
            }
            bb >>= 1
        }
        return UInt8(p & 0xFF)
    }
}

// MARK: - Precomputed Polynomial Manager

/// Manages precomputed polynomials with automatic invalidation on challenge change.
///
/// Thread-safe access to precomputed vanishing polynomials, with automatic
/// recomputation only when needed (challenge change or first use).
public final class PrecomputedPolyManager: @unchecked Sendable {
    /// Lock for thread safety.
    private let lock = NSLock()

    /// Maximum tower level.
    public let maxLevel: Int

    /// Domain size.
    public let domainSize: Int

    /// Vanishing polynomials by level.
    private var vanishingByLevel: [[UInt8]]

    /// Subspace polynomials by level.
    private var subspaceByLevel: [[UInt8]]

    /// Lagrange basis numerators by level.
    private var lagrangeNumerators: [[UInt8]]

    /// Current challenge context (if any).
    private var currentChallenge: ChallengeContext?

    /// Current state.
    private var state: PrecomputationState = .uninitialized

    /// Initialization timestamp.
    private var initializedAt: Date?

    /// Last access timestamp.
    private var lastAccessAt: Date?

    public init(maxLevel: Int, domainSize: Int) {
        self.maxLevel = maxLevel
        self.domainSize = domainSize
        self.vanishingByLevel = [[UInt8]](repeating: [], count: maxLevel + 1)
        self.subspaceByLevel = [[UInt8]](repeating: [], count: maxLevel + 1)
        self.lagrangeNumerators = [[UInt8]](repeating: [], count: maxLevel + 1)
    }

    // MARK: - Initialization

    /// Initialize all precomputed polynomials.
    /// Thread-safe. Idempotent after first call.
    public func initialize() {
        lock.lock()
        defer { lock.unlock() }

        guard state != .ready else { return }

        state = .computing(progress: 0)

        // Precompute vanishing polynomials for each level
        for k in 1...maxLevel {
            vanishingByLevel[k] = computeVanishingPolynomial(level: k)
            subspaceByLevel[k] = computeSubspacePolynomial(level: k)
            lagrangeNumerators[k] = computeLagrangeNumerators(level: k)

            let progress = Double(k) / Double(maxLevel)
            state = .computing(progress: progress)
        }

        state = .ready
        initializedAt = Date()
    }

    /// Initialize in background.
    public func initializeAsync() {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            self?.initialize()
        }
    }

    // MARK: - Access

    /// Get vanishing polynomial for a level.
    /// O(1) after initialization.
    public func vanishingPolynomial(level: Int) -> [UInt8] {
        lock.lock()
        defer { lock.unlock() }

        ensureReady()
        lastAccessAt = Date()
        return vanishingByLevel[level]
    }

    /// Get subspace polynomial for a level.
    public func subspacePolynomial(level: Int) -> [UInt8] {
        lock.lock()
        defer { lock.unlock() }

        ensureReady()
        lastAccessAt = Date()
        return subspaceByLevel[level]
    }

    /// Get Lagrange numerator for a level.
    public func lagrangeNumerator(level: Int) -> [UInt8] {
        lock.lock()
        defer { lock.unlock() }

        ensureReady()
        lastAccessAt = Date()
        return lagrangeNumerators[level]
    }

    // MARK: - Challenge Adjustment

    /// Update challenge context for O(1) per-query adjustments.
    /// Call when verifier sends a new challenge.
    public func updateChallenge(_ challenge: UInt8) {
        lock.lock()
        defer { lock.unlock() }

        // Only update if challenge changed
        if currentChallenge?.challenge != challenge {
            currentChallenge = ChallengeContext(challenge: challenge)
        }
    }

    /// Get challenge-adjusted vanishing polynomial.
    /// O(1) with cached challenge.
    public func adjustedVanishing(level: Int, at point: UInt8) -> UInt8 {
        lock.lock()
        defer { lock.unlock() }

        ensureReady()

        // Basic vanishing value
        let basic = vanishingByLevel[level][Int(point) % domainSize]

        // Adjust if we have a challenge
        if let ctx = currentChallenge {
            // V_adj(x) = V(x) * prod_{i} (x - r^{2^i})
            // Simplified: just multiply by challenge power
            let adjustment = ctx.getPower(level % ctx.powers.count)
            return gf28Mul(basic, adjustment)
        }

        return basic
    }

    // MARK: - Cache Management

    /// Invalidate cache, forcing recomputation on next access.
    public func invalidate() {
        lock.lock()
        defer { lock.unlock() }

        state = .invalidated
        currentChallenge = nil
    }

    /// Clear all precomputed data.
    public func clear() {
        lock.lock()
        defer { lock.unlock() }

        vanishingByLevel = [[UInt8]](repeating: [], count: maxLevel + 1)
        subspaceByLevel = [[UInt8]](repeating: [], count: maxLevel + 1)
        lagrangeNumerators = [[UInt8]](repeating: [], count: maxLevel + 1)
        currentChallenge = nil
        state = .uninitialized
        initializedAt = nil
        lastAccessAt = nil
    }

    /// Current state.
    public var currentState: PrecomputationState {
        lock.lock()
        defer { lock.unlock() }
        return state
    }

    /// Whether cache is ready.
    public var isReady: Bool {
        lock.lock()
        defer { lock.unlock() }
        return state == .ready
    }

    // MARK: - Private

    private func ensureReady() {
        if state == .uninitialized || state == .invalidated {
            initialize()
        } else if case .computing = state {
            // Wait for completion (spinlock - not ideal for production)
            while case .computing = state { }
        }
    }

    /// Compute vanishing polynomial for level k.
    /// V_k(x) = x^{2^k} - x (full subspace vanishing)
    private func computeVanishingPolynomial(level: Int) -> [UInt8] {
        let size = 1 << level
        var vanish = [UInt8](repeating: 0, count: domainSize)

        for x in 0..<domainSize {
            let xByte = UInt8(x & 0xFF)
            // x^{2^k} = x^{2^k mod 255} in GF(2^8)
            // Simplified: x^{2^k} = x for k >= 8 in our representation
            if level >= 8 {
                vanish[x] = 0  // Zero at all points in full field
            } else {
                // For small k, compute actual vanishing
                var val: UInt8 = xByte
                for _ in 0..<(1 << level) {
                    val = gf28Mul(val, xByte)
                }
                vanish[x] = val ^ xByte
            }
        }
        return vanish
    }

    /// Compute subspace polynomial.
    /// S_k(x) = prod_{i=0}^{2^k-1} (x - beta^i)
    private func computeSubspacePolynomial(level: Int) -> [UInt8] {
        // Simplified: subspace is indicator polynomial
        let size = 1 << level
        var subspace = [UInt8](repeating: 0, count: domainSize)

        for i in 0..<size {
            let x = gf28Pow(0x02, UInt32(i))
            if Int(x) < domainSize {
                subspace[Int(x)] = 1
            }
        }
        return subspace
    }

    /// Compute Lagrange numerator: prod_{j != i} (x_i - x_j)
    private func computeLagrangeNumerators(level: Int) -> [UInt8] {
        let size = 1 << level
        var numerators = [UInt8](repeating: 0, count: size)

        for i in 0..<size {
            var num: UInt8 = 1
            let xi = UInt8(i)
            for j in 0..<size where j != i {
                let diff = xi ^ UInt8(j)
                num = gf28Mul(num, diff)
            }
            numerators[i] = num
        }
        return numerators
    }

    /// GF(2^8) exponentiation.
    private func gf28Pow(_ base: UInt8, _ exp: UInt32) -> UInt8 {
        var result: UInt8 = 1
        var b = base
        var e = exp
        while e > 0 {
            if e & 1 == 1 {
                result = gf28Mul(result, b)
            }
            b = gf28Mul(b, b)
            e >>= 1
        }
        return result
    }

    /// GF(2^8) multiplication.
    private func gf28Mul(_ a: UInt8, _ b: UInt8) -> UInt8 {
        var p: UInt16 = 0
        var aa = UInt16(a)
        var bb = UInt16(b)
        for _ in 0..<8 {
            if bb & 1 != 0 {
                p ^= aa
            }
            let hiBitSet = (aa & 0x80) != 0
            aa <<= 1
            if hiBitSet {
                aa ^= 0x11B
            }
            bb >>= 1
        }
        return UInt8(p & 0xFF)
    }
}

// MARK: - Thread-Safe Cache Access

/// Thread-safe wrapper for precomputed polynomial access.
public final class ThreadSafePrecomputedPoly {
    private let manager: PrecomputedPolyManager

    public init(manager: PrecomputedPolyManager) {
        self.manager = manager
    }

    /// Get vanishing polynomial (thread-safe).
    public func vanishing(level: Int) -> [UInt8] {
        return manager.vanishingPolynomial(level: level)
    }

    /// Update challenge (thread-safe).
    public func updateChallenge(_ challenge: UInt8) {
        manager.updateChallenge(challenge)
    }

    /// Get adjusted vanishing (thread-safe).
    public func adjustedVanishing(level: Int, at point: UInt8) -> UInt8 {
        return manager.adjustedVanishing(level: level, at: point)
    }

    /// Initialize if needed (thread-safe).
    public func ensureInitialized() {
        manager.initialize()
    }
}
