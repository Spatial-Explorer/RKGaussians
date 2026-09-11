# RKGaussians

A RealityKit framework for loading and rendering Gaussian splats on visionOS.

Developed by **Spatial Explorer Labs**.

Load 3D Gaussian Splat PLY files as native RealityKit entities on visionOS.

```swift
import RKGaussians

let bear = try await Entity(named: "Bear", type: .gaussianSplat)
```

RKGaussians wraps RealityKit's Gaussian splatting API (`GaussianSplatComponent` /
`GaussianSplatResource`) behind two `Entity` initializers, handling PLY parsing,
GPU buffer assembly, and coordinate-convention correction internally. Consuming
apps need no Gaussian-specific loader code.

## Requirements

| | |
|---|---|
| Platform | visionOS 27.0+ |
| Rendering | Physical Apple Vision Pro only — on the simulator the initializers throw `.splatRenderingUnavailableInSimulator` |
| Swift | 6.2 toolchain (Xcode 27) |

## Installation

In Xcode, choose **File → Add Package Dependencies…** and enter:

```
https://github.com/Spatial-Explorer/RKGaussians
```

Or add it to a `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/Spatial-Explorer/RKGaussians", branch: "main")
]
```

Then:

```swift
import RKGaussians
```

For local development against a checkout, use **Add Local…** instead — a local
copy dragged into the workspace overrides the remote package automatically.

## Usage

### From the app bundle

```swift
// "seahorse" or "seahorse.ply" both work.
let splat = try await Entity(named: "seahorse", type: .gaussianSplat)
```

### From a file URL

Security-scoped URLs (for example from `.fileImporter`) are handled internally —
pass the picked URL straight through:

```swift
let splat = try await Entity(contentsOf: pickedURL, type: .gaussianSplat)
```

### From a remote URL

`http(s)` URLs are downloaded to a temporary file, validated, parsed, and
cleaned up. The initializer returns when the splat is ready, so a large file
means a visible wait — plan UI accordingly.

```swift
let url = URL(string: "https://demos.spatial-explorer.com/framework/seahorse.ply")!
let splat = try await Entity(contentsOf: url, type: .gaussianSplat)
```

### Handling errors

All failures throw `GaussianSplatLoadingError`, which is `LocalizedError`:

```swift
do {
    splatEntity = try await Entity(named: name, type: .gaussianSplat)
} catch {
    logger.error("Failed to load splat: \(error, privacy: .public)")
    loadErrorMessage = error.localizedDescription
}
```

| Case | Meaning |
|---|---|
| `.resourceNotFound(name:)` | No `<name>.ply` in the bundle |
| `.unsupportedFormat(String)` | ASCII or big-endian PLY, list properties, or a non-float anchor property |
| `.invalidHeader(String)` | Missing `ply` magic or `end_header`, no vertex element, or an invalid SH coefficient count |
| `.missingRequiredProperty(String)` | A required splat property (see below) is absent |
| `.truncatedBody(expectedBytes:availableBytes:)` | The binary body is shorter than the header promises |
| `.bufferAllocationFailed` | GPU buffer allocation failed |
| `.downloadFailed(statusCode:)` | Remote fetch returned a non-2xx status |
| `.splatRenderingUnavailableInSimulator` | Running on the simulator |

## Supported PLY files

RKGaussians reads the standard 3D Gaussian Splatting vertex layout produced by
INRIA-style training pipelines and tools like SuperSplat:

- **Binary little-endian** only (ASCII and big-endian files are rejected with a
  descriptive error rather than misparsed).
- Required float properties: `x y z`, `f_dc_0..2`, `opacity`, `scale_0..2`
  (log-space), `rot_0..3` (w, x, y, z quaternion).
- Spherical harmonics degrees 0–3 (`f_rest_*` optional; 9, 24, or 45
  coefficients when present).
- Extra vertex properties of any scalar type (normals, custom channels) are
  tolerated — offsets are tracked byte-accurately, so they can't corrupt the
  splat fields.
- CRLF line endings in the header are accepted.
- `opacity` and `scale` use the standard pre-activation encoding; sigmoid and
  exponential activations are applied at render time.
- NaN/Inf values are sanitized to 0 (RealityKit rejects non-finite buffers).

## Orientation baseline

Standard 3DGS exports use a convention that loads upside down and facing away
in RealityKit. RKGaussians applies a **180° rotation about the Z axis** at load
time — positions, per-splat quaternions, and spherical-harmonics signs — so
standard exports arrive upright and facing the viewer (device-verified).

If your pipeline pre-rotates its exports, they will come out flipped; re-export
in the standard convention, or counter-rotate the returned entity.

## Interaction tip

Splats have no mesh, so gesture systems need an explicit collision shape — and
collision shapes live in the entity's **local** space, scaling with the entity.
Derive the shape from the visual bounds rather than hardcoding a size:

```swift
let bounds = splat.visualBounds(relativeTo: splat)
ManipulationComponent.configureEntity(
    splat,
    collisionShapes: [.generateBox(size: bounds.extents).offsetBy(translation: bounds.center)]
)
```

## Implementation notes

- Parsing and deinterleaving run on a detached background task; only the final
  component assembly touches the main actor.
- The packed PLY rows (array-of-structures) are deinterleaved into per-field
  `LowLevelBuffer`s (structure-of-arrays) as the splatting API requires.
- The `type:` label exists to disambiguate from RealityKit's own
  `Entity(named:in:)` / `Entity(contentsOf:withName:)` initializers, and leaves
  room for future formats in `SplatFormat`.


## About

RKGaussians is developed and maintained by **Spatial Explorer Labs**, focused on native spatial computing, Gaussian splatting, and immersive media technologies for Apple platforms.

https://spatial-explorer.com
## License

Derived from Apple's "Gaussian Splats on visionOS" sample code. See
`LICENSE.txt` for the license and retained attribution.

Copyright © 2026 Spatial Explorer Labs. All rights reserved.