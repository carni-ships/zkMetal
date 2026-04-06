import Foundation
import zkMetal

public func runBatchEd25519Tests() {
    suite("Batch Ed25519 Verification")

    let engine = EdDSAEngine()
    let verifier = BatchEd25519Verifier()

    func makeKey(_ seedByte: UInt8) -> EdDSASecretKey {
        var seed = [UInt8](repeating: 0, count: 32)
        seed[0] = seedByte
        return EdDSASecretKey(seed: seed)
    }

    func makeSig(_ seedByte: UInt8, _ message: String) -> (EdDSASecretKey, [UInt8], Ed25519Signature) {
        let sk = makeKey(seedByte)
        let msg = Array(message.utf8)
        let sig = engine.sign(message: msg, secretKey: sk)
        return (sk, msg, sig)
    }

    func makeBatch(_ n: Int, offset: Int = 0) -> ([[UInt8]], [Ed25519Signature], [Ed25519PublicKey]) {
        var sigs = [Ed25519Signature]()
        var msgs = [[UInt8]]()
        var pks = [Ed25519PublicKey]()
        for i in 0..<n {
            var si = [UInt8](repeating: 0, count: 32)
            si[0] = UInt8((i + offset + 1) & 0xFF)
            let ski = EdDSASecretKey(seed: si)
            let msgi = Array("Batch message \(i)".utf8)
            sigs.append(engine.sign(message: msgi, secretKey: ski))
            msgs.append(msgi)
            pks.append(ski.publicKey)
        }
        return (msgs, sigs, pks)
    }

    // Single verification
    do {
        let (sk, msg, sig) = makeSig(42, "Hello, Ed25519!")
        expect(verifier.verifySingle(message: msg, signature: sig, publicKey: sk.publicKey), "valid single sig")
    }

    // Reject wrong message
    do {
        let (sk, _, sig) = makeSig(42, "Hello, Ed25519!")
        let wrongMsg = Array("Wrong message".utf8)
        expect(!verifier.verifySingle(message: wrongMsg, signature: sig, publicKey: sk.publicKey), "reject wrong msg")
    }

    // Reject wrong key
    do {
        let (_, msg, sig) = makeSig(42, "Hello, Ed25519!")
        let sk2 = makeKey(99)
        expect(!verifier.verifySingle(message: msg, signature: sig, publicKey: sk2.publicKey), "reject wrong key")
    }

    // Batch all valid
    do {
        let (msgs, sigs, pks) = makeBatch(16)
        expect(verifier.verifyBatch(messages: msgs, signatures: sigs, publicKeys: pks), "batch all valid")
    }

    // Batch detect invalid
    do {
        var (msgs, sigs, pks) = makeBatch(16)
        var badS = sigs[8].s
        badS[0] ^= 0xFF
        sigs[8] = Ed25519Signature(r: sigs[8].r, s: badS)
        expect(!verifier.verifyBatch(messages: msgs, signatures: sigs, publicKeys: pks), "batch detect invalid")
    }

    // Empty batch
    do {
        expect(verifier.verifyBatch(messages: [], signatures: [], publicKeys: []), "empty batch")
    }
}
