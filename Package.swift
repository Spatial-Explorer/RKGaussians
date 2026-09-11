// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "RKGaussians",
    platforms: [
        // Gaussian Splat rendering requires visionOS 27; earlier versions are not supported.
        .visionOS("27.0")
    ],
    products: [
        .library(
            name: "RKGaussians",
            targets: ["RKGaussians"]
        )
    ],
    targets: [
        .target(name: "RKGaussians")
    ]
)
