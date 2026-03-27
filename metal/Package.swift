// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "zkmsm",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "zkmsm",
            path: "Sources/zkmsm",
            exclude: ["shaders"],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
    ]
)
