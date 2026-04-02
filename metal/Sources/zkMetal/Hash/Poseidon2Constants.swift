// Poseidon2 round constants for BN254 Fr, t=3
// Source: HorizenLabs reference implementation
// https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_bn256.rs
//
// Parameters: t=3, d=5 (x^5 S-box), rounds_f=8, rounds_p=56, total=64
// External matrix M_E: circulant [2,1,1]
// Internal matrix M_I: [[2,1,1],[1,2,1],[1,1,3]]

import Foundation

public enum Poseidon2Config {
    public static let t = 3
    public static let alpha = 5
    public static let roundsF = 8     // 4 beginning + 4 end
    public static let roundsP = 56
    public static let totalRounds = 64
}

/// Parse a hex string (with or without 0x prefix) into 4 UInt64 limbs (little-endian).
private func hexToLimbs(_ hex: String) -> [UInt64] {
    let h = hex.hasPrefix("0x") ? String(hex.dropFirst(2)) : hex
    let padded = String(repeating: "0", count: max(0, 64 - h.count)) + h
    // padded is 64 hex chars = 256 bits, big-endian
    // Split into 4 x 16 hex chars (64 bits each), reverse for little-endian
    var limbs = [UInt64](repeating: 0, count: 4)
    for i in 0..<4 {
        let start = padded.index(padded.startIndex, offsetBy: (3 - i) * 16)
        let end = padded.index(start, offsetBy: 16)
        limbs[i] = UInt64(padded[start..<end], radix: 16)!
    }
    return limbs
}

/// Convert hex constant to Fr in Montgomery form.
private func hexToFr(_ hex: String) -> Fr {
    let limbs = hexToLimbs(hex)
    let raw = Fr.from64(limbs)
    return frMul(raw, Fr.from64(Fr.R2_MOD_R))  // convert to Montgomery form
}

// Round constants as hex strings (from HorizenLabs reference)
// Format: [round][element] where element is 0..2
private let RC_HEX: [[String]] = [
    // Full rounds (beginning): 0-3
    ["1d066a255517b7fd8bddd3a93f7804ef7f8fcde48bb4c37a59a09a1a97052816",
     "29daefb55f6f2dc6ac3f089cebcc6120b7c6fef31367b68eb7238547d32c1610",
     "1f2cb1624a78ee001ecbd88ad959d7012572d76f08ec5c4f9e8b7ad7b0b4e1d1"],
    ["0aad2e79f15735f2bd77c0ed3d14aa27b11f092a53bbc6e1db0672ded84f31e5",
     "2252624f8617738cd6f661dd4094375f37028a98f1dece66091ccf1595b43f28",
     "1a24913a928b38485a65a84a291da1ff91c20626524b2b87d49f4f2c9018d735"],
    ["22fc468f1759b74d7bfc427b5f11ebb10a41515ddff497b14fd6dae1508fc47a",
     "1059ca787f1f89ed9cd026e9c9ca107ae61956ff0b4121d5efd65515617f6e4d",
     "02be9473358461d8f61f3536d877de982123011f0bf6f155a45cbbfae8b981ce"],
    ["0ec96c8e32962d462778a749c82ed623aba9b669ac5b8736a1ff3a441a5084a4",
     "292f906e073677405442d9553c45fa3f5a47a7cdb8c99f9648fb2e4d814df57e",
     "274982444157b86726c11b9a0f5e39a5cc611160a394ea460c63f0b2ffe5657e"],
    // Partial rounds: 4-59 (only first element, others are zero)
    ["1a1d063e54b1e764b63e1855bff015b8cedd192f47308731499573f23597d4b5", "0", "0"],
    ["26abc66f3fdf8e68839d10956259063708235dccc1aa3793b91b002c5b257c37", "0", "0"],
    ["0c7c64a9d887385381a578cfed5aed370754427aabca92a70b3c2b12ff4d7be8", "0", "0"],
    ["1cf5998769e9fab79e17f0b6d08b2d1eba2ebac30dc386b0edd383831354b495", "0", "0"],
    ["0f5e3a8566be31b7564ca60461e9e08b19828764a9669bc17aba0b97e66b0109", "0", "0"],
    ["18df6a9d19ea90d895e60e4db0794a01f359a53a180b7d4b42bf3d7a531c976e", "0", "0"],
    ["04f7bf2c5c0538ac6e4b782c3c6e601ad0ea1d3a3b9d25ef4e324055fa3123dc", "0", "0"],
    ["29c76ce22255206e3c40058523748531e770c0584aa2328ce55d54628b89ebe6", "0", "0"],
    ["198d425a45b78e85c053659ab4347f5d65b1b8e9c6108dbe00e0e945dbc5ff15", "0", "0"],
    ["25ee27ab6296cd5e6af3cc79c598a1daa7ff7f6878b3c49d49d3a9a90c3fdf74", "0", "0"],
    ["138ea8e0af41a1e024561001c0b6eb1505845d7d0c55b1b2c0f88687a96d1381", "0", "0"],
    ["306197fb3fab671ef6e7c2cba2eefd0e42851b5b9811f2ca4013370a01d95687", "0", "0"],
    ["1a0c7d52dc32a4432b66f0b4894d4f1a21db7565e5b4250486419eaf00e8f620", "0", "0"],
    ["2b46b418de80915f3ff86a8e5c8bdfccebfbe5f55163cd6caa52997da2c54a9f", "0", "0"],
    ["12d3e0dc0085873701f8b777b9673af9613a1af5db48e05bfb46e312b5829f64", "0", "0"],
    ["263390cf74dc3a8870f5002ed21d089ffb2bf768230f648dba338a5cb19b3a1f", "0", "0"],
    ["0a14f33a5fe668a60ac884b4ca607ad0f8abb5af40f96f1d7d543db52b003dcd", "0", "0"],
    ["28ead9c586513eab1a5e86509d68b2da27be3a4f01171a1dd847df829bc683b9", "0", "0"],
    ["1c6ab1c328c3c6430972031f1bdb2ac9888f0ea1abe71cffea16cda6e1a7416c", "0", "0"],
    ["1fc7e71bc0b819792b2500239f7f8de04f6decd608cb98a932346015c5b42c94", "0", "0"],
    ["03e107eb3a42b2ece380e0d860298f17c0c1e197c952650ee6dd85b93a0ddaa8", "0", "0"],
    ["2d354a251f381a4669c0d52bf88b772c46452ca57c08697f454505f6941d78cd", "0", "0"],
    ["094af88ab05d94baf687ef14bc566d1c522551d61606eda3d14b4606826f794b", "0", "0"],
    ["19705b783bf3d2dc19bcaeabf02f8ca5e1ab5b6f2e3195a9d52b2d249d1396f7", "0", "0"],
    ["09bf4acc3a8bce3f1fcc33fee54fc5b28723b16b7d740a3e60cef6852271200e", "0", "0"],
    ["1803f8200db6013c50f83c0c8fab62843413732f301f7058543a073f3f3b5e4e", "0", "0"],
    ["0f80afb5046244de30595b160b8d1f38bf6fb02d4454c0add41f7fef2faf3e5c", "0", "0"],
    ["126ee1f8504f15c3d77f0088c1cfc964abcfcf643f4a6fea7dc3f98219529d78", "0", "0"],
    ["23c203d10cfcc60f69bfb3d919552ca10ffb4ee63175ddf8ef86f991d7d0a591", "0", "0"],
    ["2a2ae15d8b143709ec0d09705fa3a6303dec1ee4eec2cf747c5a339f7744fb94", "0", "0"],
    ["07b60dee586ed6ef47e5c381ab6343ecc3d3b3006cb461bbb6b5d89081970b2b", "0", "0"],
    ["27316b559be3edfd885d95c494c1ae3d8a98a320baa7d152132cfe583c9311bd", "0", "0"],
    ["1d5c49ba157c32b8d8937cb2d3f84311ef834cc2a743ed662f5f9af0c0342e76", "0", "0"],
    ["2f8b124e78163b2f332774e0b850b5ec09c01bf6979938f67c24bd5940968488", "0", "0"],
    ["1e6843a5457416b6dc5b7aa09a9ce21b1d4cba6554e51d84665f75260113b3d5", "0", "0"],
    ["11cdf00a35f650c55fca25c9929c8ad9a68daf9ac6a189ab1f5bc79f21641d4b", "0", "0"],
    ["21632de3d3bbc5e42ef36e588158d6d4608b2815c77355b7e82b5b9b7eb560bc", "0", "0"],
    ["0de625758452efbd97b27025fbd245e0255ae48ef2a329e449d7b5c51c18498a", "0", "0"],
    ["2ad253c053e75213e2febfd4d976cc01dd9e1e1c6f0fb6b09b09546ba0838098", "0", "0"],
    ["1d6b169ed63872dc6ec7681ec39b3be93dd49cdd13c813b7d35702e38d60b077", "0", "0"],
    ["1660b740a143664bb9127c4941b67fed0be3ea70a24d5568c3a54e706cfef7fe", "0", "0"],
    ["0065a92d1de81f34114f4ca2deef76e0ceacdddb12cf879096a29f10376ccbfe", "0", "0"],
    ["1f11f065202535987367f823da7d672c353ebe2ccbc4869bcf30d50a5871040d", "0", "0"],
    ["26596f5c5dd5a5d1b437ce7b14a2c3dd3bd1d1a39b6759ba110852d17df0693e", "0", "0"],
    ["16f49bc727e45a2f7bf3056efcf8b6d38539c4163a5f1e706743db15af91860f", "0", "0"],
    ["1abe1deb45b3e3119954175efb331bf4568feaf7ea8b3dc5e1a4e7438dd39e5f", "0", "0"],
    ["0e426ccab66984d1d8993a74ca548b779f5db92aaec5f102020d34aea15fba59", "0", "0"],
    ["0e7c30c2e2e8957f4933bd1942053f1f0071684b902d534fa841924303f6a6c6", "0", "0"],
    ["0812a017ca92cf0a1622708fc7edff1d6166ded6e3528ead4c76e1f31d3fc69d", "0", "0"],
    ["21a5ade3df2bc1b5bba949d1db96040068afe5026edd7a9c2e276b47cf010d54", "0", "0"],
    ["01f3035463816c84ad711bf1a058c6c6bd101945f50e5afe72b1a5233f8749ce", "0", "0"],
    ["0b115572f038c0e2028c2aafc2d06a5e8bf2f9398dbd0fdf4dcaa82b0f0c1c8b", "0", "0"],
    ["1c38ec0b99b62fd4f0ef255543f50d2e27fc24db42bc910a3460613b6ef59e2f", "0", "0"],
    ["1c89c6d9666272e8425c3ff1f4ac737b2f5d314606a297d4b1d0b254d880c53e", "0", "0"],
    ["03326e643580356bf6d44008ae4c042a21ad4880097a5eb38b71e2311bb88f8f", "0", "0"],
    ["268076b0054fb73f67cee9ea0e51e3ad50f27a6434b5dceb5bdde2299910a4c9", "0", "0"],
    // Full rounds (end): 60-63
    ["1acd63c67fbc9ab1626ed93491bda32e5da18ea9d8e4f10178d04aa6f8747ad0",
     "19f8a5d670e8ab66c4e3144be58ef6901bf93375e2323ec3ca8c86cd2a28b5a5",
     "1c0dc443519ad7a86efa40d2df10a011068193ea51f6c92ae1cfbb5f7b9b6893"],
    ["14b39e7aa4068dbe50fe7190e421dc19fbeab33cb4f6a2c4180e4c3224987d3d",
     "1d449b71bd826ec58f28c63ea6c561b7b820fc519f01f021afb1e35e28b0795e",
     "1ea2c9a89baaddbb60fa97fe60fe9d8e89de141689d1252276524dc0a9e987fc"],
    ["0478d66d43535a8cb57e9c1c3d6a2bd7591f9a46a0e9c058134d5cefdb3c7ff1",
     "19272db71eece6a6f608f3b2717f9cd2662e26ad86c400b21cde5e4a7b00bebe",
     "14226537335cab33c749c746f09208abb2dd1bd66a87ef75039be846af134166"],
    ["01fd6af15956294f9dfe38c0d976a088b21c21e4a1c2e823f912f44961f9a9ce",
     "18e5abedd626ec307bca190b8b2cab1aaee2e62ed229ba5a5ad8518d4e5f2a57",
     "0fc1bbceba0590f5abbdffa6d3b35e3297c021a3a409926d0e2d54dc1c84fda6"],
]

/// Lazily computed round constants in Montgomery form.
public let POSEIDON2_ROUND_CONSTANTS: [[Fr]] = {
    var rc = [[Fr]]()
    rc.reserveCapacity(Poseidon2Config.totalRounds)
    for round in RC_HEX {
        var row = [Fr]()
        row.reserveCapacity(3)
        for hex in round {
            row.append(hexToFr(hex))
        }
        rc.append(row)
    }
    return rc
}()
