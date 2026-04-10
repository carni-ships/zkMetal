import Foundation

func extractFieldDefs(_ src: String) -> String {
    src.split(separator: "\n")
        .filter { line in
            let l = line.trimmingCharacters(in: .whitespaces)
            return !l.hasPrefix("#ifndef") && !l.hasPrefix("#define") &&
                   !l.hasPrefix("#endif") && !l.hasPrefix("#include") &&
                   !l.hasPrefix("#pragma")
        }
        .joined(separator: "\n")
}

func cleanPoseidon(_ src: String) -> String {
    src.split(separator: "\n")
        .filter { line in
            !line.contains("#include \"../fields/pallas_fp.metal\"") &&
            !line.contains("#include \"../fields/vesta_fp.metal\"")
        }
        .joined(separator: "\n")
}

let shaderDir = "/Users/carnation/Documents/Claude/zkMetal/Sources/Shaders"
let pallasFpSrc = try! String(contentsOfFile: shaderDir + "/fields/pallas_fp.metal", encoding: .utf8)
let vestaFpSrc = try! String(contentsOfFile: shaderDir + "/fields/vesta_fp.metal", encoding: .utf8)
let poseidonSrc = try! String(contentsOfFile: shaderDir + "/hash/pasta_poseidon.metal", encoding: .utf8)

let pallasClean = extractFieldDefs(pallasFpSrc)
let vestaClean = extractFieldDefs(vestaFpSrc)
let poseidonClean = cleanPoseidon(poseidonSrc)

let combined = pallasClean + "\n" + vestaClean + "\n" + poseidonClean

try! combined.write(toFile: "/tmp/combined_pasta_poseidon.metal", atomically: true, encoding: .utf8)

print("Combined shader written to /tmp/combined_pasta_poseidon.metal")
print("Total lines: \(combined.split(separator: "\n").count)")

// Find the line with PALLAS_POS_MDS
let lines = combined.split(separator: "\n")
for (i, line) in lines.enumerated() {
    if line.contains("PALLAS_POS_MDS") || line.contains("VESTA_POS_MDS") {
        print("Line \(i+1): \(line.prefix(100))...")
    }
}
