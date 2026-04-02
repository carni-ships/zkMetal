// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "zkMetal",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        .library(name: "zkMetal", targets: ["zkMetal"]),
        .executable(name: "zkmsm-cli", targets: ["zkmsm-cli"]),
        .executable(name: "zkbench", targets: ["zkbench"]),
    ],
    targets: [
        .target(
            name: "zkMetal",
            path: "Sources/zkMetal",
            resources: [
                .copy("../../Sources/Shaders"),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .executableTarget(
            name: "zkmsm-cli",
            dependencies: ["zkMetal"],
            path: "Sources/zkmsm-cli"
        ),
        .executableTarget(
            name: "zkbench",
            dependencies: ["zkMetal"],
            path: "Sources/zkbench"
        ),
        .executableTarget(
            name: "zkMetalTests",
            dependencies: ["zkMetal"],
            path: "Tests/zkMetalTests"
        ),
        // Keep the old target working during migration
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
