// swift-tools-version:5.9
import PackageDescription

// Two-target layout:
//   StreamMoECore — pure-Swift services (ServerStatus client, OllamaController,
//                   AppDetector, OpenWebUIProbe, NetworkAccessSettings,
//                   ModelManager, WelcomeWizard state). All testable.
//   StreamMoE     — SwiftUI/AppKit shell that imports StreamMoECore and
//                   wires it into NSStatusItem and SwiftUI panes.
//
// Tests live in StreamMoECoreTests and hit the core types directly; the UI
// target is intentionally kept thin so it doesn't need XCTest coverage.
let package = Package(
    name: "StreamMoE",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "StreamMoE", targets: ["StreamMoE"]),
        .library(name: "StreamMoECore", targets: ["StreamMoECore"]),
    ],
    targets: [
        .target(
            name: "StreamMoECore",
            path: "Sources/StreamMoECore"
        ),
        .executableTarget(
            name: "StreamMoE",
            dependencies: ["StreamMoECore"],
            path: "Sources/StreamMoE",
            resources: [
                .copy("../../Resources/Guides"),
            ]
        ),
        .testTarget(
            name: "StreamMoECoreTests",
            dependencies: ["StreamMoECore"],
            path: "Tests/StreamMoECoreTests"
        ),
    ]
)
