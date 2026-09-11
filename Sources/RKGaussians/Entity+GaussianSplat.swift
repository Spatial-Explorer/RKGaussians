/*
Developer: Spatial Explorer Labs
 
Derived from Apple's "Gaussian Splats on visionOS" sample code.
See the LICENSE.txt file in that sample for licensing information.

Abstract:
The package's public API: async Entity initializers that load Gaussian Splat
  PLY files from a bundle, a local file URL, or an http(s) URL.
*/

import Foundation
import RealityKit

/// The splat file formats the loader understands.
///
/// The `type:` parameter on the `Entity` initializers exists both to
/// disambiguate them from RealityKit's own `Entity(named:in:)` /
/// `Entity(contentsOf:withName:)` and to leave room for future formats.
public enum SplatFormat: Sendable {
    case gaussianSplat
}

/// Errors thrown while loading a Gaussian Splat file.
public enum GaussianSplatLoadingError: Error, LocalizedError, Sendable {
    /// No `<name>.ply` resource exists in the bundle.
    case resourceNotFound(name: String)
    /// The file is PLY, but a variant the loader does not read
    /// (ASCII, big-endian, list properties, or a non-float anchor property).
    case unsupportedFormat(String)
    /// The header is malformed (missing magic or `end_header`, no vertex
    /// element, or an invalid spherical harmonics coefficient count).
    case invalidHeader(String)
    /// The vertex element lacks a property the splat layout requires.
    case missingRequiredProperty(String)
    /// The binary body is shorter than the header promises.
    case truncatedBody(expectedBytes: Int, availableBytes: Int)
    case bufferAllocationFailed
    case downloadFailed(statusCode: Int?)
    /// Gaussian Splat rendering requires a physical Vision Pro.
    case splatRenderingUnavailableInSimulator

    public var errorDescription: String? {
        switch self {
        case .resourceNotFound(let name):
            return "\(name).ply was not found in the bundle."
        case .unsupportedFormat(let detail):
            return "Unsupported PLY format: \(detail)"
        case .invalidHeader(let detail):
            return "Invalid PLY header: \(detail)"
        case .missingRequiredProperty(let name):
            return "The PLY vertex element is missing the required property '\(name)'."
        case .truncatedBody(let expectedBytes, let availableBytes):
            return "Truncated PLY body: expected \(expectedBytes) bytes of vertex data, found \(availableBytes)."
        case .bufferAllocationFailed:
            return "Failed to allocate GPU buffers for the splat data."
        case .downloadFailed(let statusCode):
            if let statusCode {
                return "Download failed with HTTP status \(statusCode)."
            }
            return "Download failed."
        case .splatRenderingUnavailableInSimulator:
            return "Gaussian Splat rendering requires a Vision Pro running visionOS 27; it is unavailable in the simulator."
        }
    }
}

extension Entity {
    /// Loads a Gaussian Splat PLY resource from `bundle` and returns an entity
    /// with a configured `GaussianSplatComponent`.
    ///
    /// `name` may be given with or without the `.ply` suffix.
    @MainActor
    public convenience init(named name: String, type: SplatFormat, in bundle: Bundle = .main) async throws {
        let baseName = name.lowercased().hasSuffix(".ply") ? String(name.dropLast(4)) : name
        guard let url = bundle.url(forResource: baseName, withExtension: "ply") else {
            throw GaussianSplatLoadingError.resourceNotFound(name: baseName)
        }
        try await self.init(loadingSplatAt: url)
    }

    /// Loads a Gaussian Splat PLY file from a local URL (including
    /// security-scoped URLs from a Files picker) or an http(s) URL.
    @MainActor
    public convenience init(contentsOf url: URL, type: SplatFormat) async throws {
        if let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https" {
            let (tempURL, response) = try await URLSession.shared.download(from: url)
            defer { try? FileManager.default.removeItem(at: tempURL) }
            if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
                throw GaussianSplatLoadingError.downloadFailed(statusCode: http.statusCode)
            }
            try await self.init(loadingSplatAt: tempURL)
        } else {
            try await self.init(loadingSplatAt: url)
        }
    }

    @MainActor
    private convenience init(loadingSplatAt url: URL) async throws {
#if targetEnvironment(simulator)
        throw GaussianSplatLoadingError.splatRenderingUnavailableInSimulator
#else
        let buffers = try await Task.detached(priority: .userInitiated) {
            // Security-scoped access is required for Files-picker URLs; the
            // call harmlessly returns false for bundle and temporary URLs.
            let didAccess = url.startAccessingSecurityScopedResource()
            defer { if didAccess { url.stopAccessingSecurityScopedResource() } }
            let splatData = try readGaussianSplatFile(url)
            let buffers = try deinterleaveGaussianSplatData(splatData)
            rotateStandardSplatsToRealityKit(buffers)
            return buffers
        }.value
        let component = try assembleSplatComponent(from: buffers)
        self.init()
        components.set(component)
#endif
    }
}
