// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "zkMetal",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        .library(name: "zkMetal", targets: ["zkMetal"]),
        .library(name: "zkMetal-ffi", type: .dynamic, targets: ["zkMetal-ffi"]),
        .executable(name: "zkmsm", targets: ["zkmsm-cli"]),
        .executable(name: "zkbench", targets: ["zkbench"]),
    ],
    targets: [
        .target(
            name: "NeonFieldOps",
            path: "Sources/NeonFieldOps",
            publicHeadersPath: "include",
            cSettings: [
                .unsafeFlags(["-O3"]),
            ]
        ),
        .target(
            name: "zkMetal",
            dependencies: ["NeonFieldOps"],
            path: "Sources/zkMetal",
            resources: [
                .copy("../../Sources/Shaders"),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        ),
        .target(
            name: "zkMetal-ffi",
            dependencies: ["zkMetal"],
            path: "Sources/zkMetal-ffi",
            publicHeadersPath: "include",
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
    ]
)
