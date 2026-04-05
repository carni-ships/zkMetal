// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ZKMetalDemo",
    platforms: [.macOS(.v13), .iOS(.v16)],
    dependencies: [
        .package(name: "zkMetal", path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "ZKMetalDemo",
            dependencies: [
                .product(name: "zkMetal", package: "zkMetal"),
            ],
            path: "Sources/ZKMetalDemo"
        ),
    ]
)
