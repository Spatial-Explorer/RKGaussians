/*
Developer: Spatial Explorer Labs

Derived from Apple's "Gaussian Splats on visionOS" sample code.
See the LICENSE.txt file in that sample for licensing information.

Abstract:
Assembles a RealityKit GaussianSplatComponent from the GPU buffers the PLY
  parser produces.
*/

import Foundation
import RealityKit

// MARK: - Shared types

/// GPU buffers for one splat asset. The parser fills these in, and the splat
/// component assembly reads them out.
///
/// This type uses `@unchecked Sendable` because `LowLevelBuffer` does not
/// conform to `Sendable`. The loader writes to the buffers on a background
/// task and then passes them to the main actor. Nothing writes to them after
/// that, so passing the struct between tasks is safe.
struct GaussianSplatBuffers: @unchecked Sendable {
    let splatCount: Int
    let degreeSH: UInt8
    let tupleSH: UInt8

    let positionBuffer: LowLevelBuffer
    let scaleBuffer: LowLevelBuffer
    let rotationBuffer: LowLevelBuffer
    let opacityBuffer: LowLevelBuffer
    let shBuffer: LowLevelBuffer
}

// MARK: - Component construction

// Gaussian Splats render only on device, and the splat API below is not part
// of the simulator SDK. The public Entity initializers throw
// `.splatRenderingUnavailableInSimulator` before ever reaching this code on
// a simulator build.
#if !targetEnvironment(simulator)

import Metal

/// Wraps a `LowLevelBuffer` in the descriptor the splat resource expects.
private func makeDescriptor(
    buffer: LowLevelBuffer,
    format: MTLAttributeFormat,
    stride: Int
) -> GaussianSplatResource.BufferDescriptor {
    GaussianSplatResource.BufferDescriptor(buffer: buffer, format: format, stride: stride, offset: 0)
}

/// Builds the descriptor for the combined diffuse-plus-spherical-harmonics
/// buffer. The stride covers `tupleSH + 1` RGB triplets per splat (the +1 is
/// the diffuse color the buffer prepends).
private func makeSHDescriptor(
    buffer: LowLevelBuffer,
    tupleSH: UInt8
) -> GaussianSplatResource.BufferDescriptor {
    makeDescriptor(buffer: buffer, format: .float3, stride: 3 * 4 * Int(tupleSH + 1))
}

/// Builds a `GaussianSplatComponent` from the loader's buffers. Pass
/// `isLinear: true` for assets whose scale and opacity values are
/// already in linear space and do not need the default exp/sigmoid activations.
///
/// The package's default isolation is nonisolated (unlike an app target built
/// with main-actor default isolation), so the explicit `@MainActor` here is
/// load-bearing: `GaussianSplatResource` and component assembly must run on
/// the main actor.
@MainActor
func assembleSplatComponent(from buffers: GaussianSplatBuffers, isLinear: Bool = false) throws -> GaussianSplatComponent {
    let degree = GaussianSplatResource.SphericalHarmonicDegree(rawValue: buffers.degreeSH) ?? .zero
    let bufferResource = try GaussianSplatResource.BufferResource(
        count: buffers.splatCount,
        position: makeDescriptor(buffer: buffers.positionBuffer, format: .float3, stride: 3 * 4),
        scale: makeDescriptor(buffer: buffers.scaleBuffer, format: .float3, stride: 3 * 4),
        rotation: makeDescriptor(buffer: buffers.rotationBuffer, format: .float4, stride: 4 * 4),
        opacity: makeDescriptor(buffer: buffers.opacityBuffer, format: .float, stride: 1 * 4),
        sphericalHarmonics: (makeSHDescriptor(buffer: buffers.shBuffer, tupleSH: buffers.tupleSH), degree)
    )

    let splatResource = GaussianSplatResource(bufferResource)
    if isLinear {
        splatResource.scaleActivation   = .identity
        splatResource.opacityActivation = .identity
    } else {
        splatResource.scaleActivation   = .exponential
        splatResource.opacityActivation = .sigmoid
    }
    return GaussianSplatComponent(splatResource)
}

#endif
