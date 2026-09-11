/*
 Developer: Spatial Explorer Labs

Derived from Apple's "Gaussian Splats on visionOS" sample code.
See the LICENSE.txt file in that sample for licensing information.

Abstract:
Parses binary PLY files exported by 3D Gaussian Splatting tools and deinterleaves
  the splat data into per-field GPU buffers ready for the RealityKit splatting API.
*/

import Foundation
import RealityKit

// MARK: - PLY format overview
//
// A PLY file starts with a plain-text header that declares each `element`
// (such as `vertex`) and its typed `property` fields. The header also
// specifies the body format: ASCII or binary, little- or big-endian.
// After `end_header`, the body packs each element's records in declaration
// order, with no padding.
//
// Splat files store one splat per `vertex` record, using little-endian
// binary and these float properties:
//
//   x, y, z         position
//   f_dc_0..2       diffuse color (degree-0 spherical harmonics)
//   f_rest_0..N     higher-order SH coefficients, channel-major:
//                   all R, then all G, then all B
//   opacity         pre-sigmoid logit; apply sigmoid to get alpha
//   scale_0..2      per-axis log-space scale; apply exp to get scale
//   rot_0..3        orientation quaternion (w, x, y, z)
//
// Exporters may interleave extra properties (such as `nx, ny, nz` normals) or
// use non-float types for them; the parser tracks a byte-accurate offset for
// every property so those extras cannot corrupt the splat field offsets.
// The anchor properties above must themselves be `float`, and each anchor's
// continuation fields (`y`/`z`, `f_dc_1..2`, …) are assumed to follow it
// contiguously, as every known splat exporter lays them out.
//
// To reach the vertex data, walk every preceding element and sum its
// per-record size times its count.

nonisolated struct PLYProperty {
    /// The byte offset of this field within one vertex row, or `-1` when unset.
    var byteOffset = -1
    /// The number of consecutive floats this field spans.
    var floatCount = 0

    var isSet: Bool { byteOffset >= 0 }
}

nonisolated struct GaussianSplatData {
    /// The entire PLY file contents.
    let data: Data
    /// The byte offset of the first vertex row within `data`.
    let vertexStart: Int
    /// The byte size of one vertex row, covering properties of every type.
    let rowByteStride: Int
    let splatCount: Int
    let degreeSH: UInt8
    /// The number of higher-order RGB spherical harmonics triplets, excluding the diffuse color.
    let tupleSH: UInt8

    let position, scale, rotation, opacity: PLYProperty
    /// The `f_dc_0..2` properties.
    let diffuseColor: PLYProperty
    /// The `f_rest_0..N` properties; unset for degree-0 files.
    let sphericalHarmonics: PLYProperty
}

// MARK: - Header parsing

private nonisolated struct PLYHeader {
    var vertexCount = 0
    var binaryStart = 0
    var formatValidated = false
    var numSHCoeffs = 0
    var position = PLYProperty()
    var scale = PLYProperty()
    var rotation = PLYProperty()
    var diffuseColor = PLYProperty()
    var sphericalHarmonics = PLYProperty()
    var opacity = PLYProperty()
    var inVertexElement = false
    var vertexSeen = false

    struct ElementInfo {
        let name: String
        let count: Int
        /// The byte size of one record, summed over every property.
        var byteSize = 0
    }
    var elements: [ElementInfo] = []
}

nonisolated extension PLYHeader {
    /// Processes one ASCII header line. Returns `true` when `end_header` is reached.
    mutating func processLine(_ rawLine: String, at lineIndex: Int, endingAt idx: Int) throws -> Bool {
        // Tolerate CRLF exporters.
        let line = rawLine.hasSuffix("\r") ? String(rawLine.dropLast()) : rawLine

        if lineIndex == 0 {
            guard line == "ply" else {
                throw GaussianSplatLoadingError.invalidHeader("missing 'ply' magic line")
            }
            return false
        }
        if line == "end_header" {
            guard formatValidated else {
                throw GaussianSplatLoadingError.invalidHeader("missing 'format' line")
            }
            binaryStart = idx + 1
            return true
        }
        if line.hasPrefix("format") {
            let parts = line.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
            guard parts.count >= 2 else {
                throw GaussianSplatLoadingError.invalidHeader("malformed format line: '\(line)'")
            }
            guard parts[1] == "binary_little_endian" else {
                throw GaussianSplatLoadingError.unsupportedFormat(
                    "only binary_little_endian PLY files are supported, found '\(parts[1])'"
                )
            }
            formatValidated = true
        } else if line.hasPrefix("element ") {
            let parts = line.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
            guard parts.count >= 3, let count = Int(parts[2]), count >= 0 else {
                throw GaussianSplatLoadingError.invalidHeader("malformed element line: '\(line)'")
            }
            let name = parts[1]
            elements.append(ElementInfo(name: name, count: count))
            inVertexElement = (name == "vertex")
            if inVertexElement {
                vertexSeen = true
                vertexCount = count
            }
        } else if line.hasPrefix("property") {
            try processProperty(line)
        }
        // Ignore comment and obj_info lines.
        return false
    }

    private mutating func processProperty(_ line: String) throws {
        let parts = line.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        guard parts.count >= 3 else {
            throw GaussianSplatLoadingError.invalidHeader("malformed property line: '\(line)'")
        }
        guard !elements.isEmpty else {
            throw GaussianSplatLoadingError.invalidHeader("property declared before any element")
        }

        let typeName = parts[1]
        if typeName == "list" {
            // A list property has no fixed record size, which breaks the byte
            // offsets of the vertex rows and of any element preceding them.
            // Lists in elements that come after vertex are harmless.
            guard vertexSeen && !inVertexElement else {
                throw GaussianSplatLoadingError.unsupportedFormat("list properties are not supported: '\(line)'")
            }
            return
        }
        guard let typeSize = propertyTypeSize(typeName) else {
            throw GaussianSplatLoadingError.unsupportedFormat("unsupported property type '\(typeName)'")
        }

        // Advance a byte-accurate cursor for every property of every element so
        // the vertex data offset and vertex row stride stay correct for any mix
        // of property types.
        let byteOffsetInRow = elements[elements.count - 1].byteSize
        elements[elements.count - 1].byteSize += typeSize

        guard inVertexElement else { return }

        let name = parts[parts.count - 1]
        let isFloat = (typeName == "float" || typeName == "float32")
        switch name {
        case "x", "f_dc_0", "opacity", "rot_0", "scale_0":
            guard isFloat else {
                throw GaussianSplatLoadingError.unsupportedFormat("property '\(name)' must be float, found '\(typeName)'")
            }
        default:
            break
        }
        if name.hasPrefix("f_rest_") {
            guard isFloat else {
                throw GaussianSplatLoadingError.unsupportedFormat("property '\(name)' must be float, found '\(typeName)'")
            }
            numSHCoeffs += 1
        }

        switch name {
        case "x":        position           = PLYProperty(byteOffset: byteOffsetInRow, floatCount: 3)
        case "f_dc_0":   diffuseColor       = PLYProperty(byteOffset: byteOffsetInRow, floatCount: 3)
        case "opacity":  opacity            = PLYProperty(byteOffset: byteOffsetInRow, floatCount: 1)
        case "rot_0":    rotation           = PLYProperty(byteOffset: byteOffsetInRow, floatCount: 4)
        case "scale_0":  scale              = PLYProperty(byteOffset: byteOffsetInRow, floatCount: 3)
        case "f_rest_0": sphericalHarmonics = PLYProperty(byteOffset: byteOffsetInRow, floatCount: 3)
        default: break // continuation of the current field, like position .y or .z
        }
    }

    /// Returns the byte size of a PLY scalar property type, or `nil` if the type is unknown.
    private func propertyTypeSize(_ typeName: String) -> Int? {
        switch typeName {
        case "char", "uchar", "int8", "uint8":         return 1
        case "short", "ushort", "int16", "uint16":     return 2
        case "int", "uint", "float",
             "int32", "uint32", "float32":             return 4
        case "double", "float64", "int64", "uint64":   return 8
        default:                                       return nil
        }
    }
}

// MARK: - File loading

/// Reads the PLY file at `url`, parses its header, locates the binary `vertex`
/// section, and returns a `GaussianSplatData` that points into the loaded data
/// with the byte offsets needed to walk each splat field.
nonisolated func readGaussianSplatFile(_ url: URL) throws -> GaussianSplatData {
    let fileData = try Data(contentsOf: url)
    let header = try parseHeader(from: fileData)

    guard let vertexElementIndex = header.elements.firstIndex(where: { $0.name == "vertex" }) else {
        throw GaussianSplatLoadingError.invalidHeader("no vertex element found")
    }
    guard header.vertexCount > 0 else {
        throw GaussianSplatLoadingError.invalidHeader("vertex element declares zero vertices")
    }

    let requiredAnchors: [(name: String, property: PLYProperty)] = [
        ("x", header.position),
        ("f_dc_0", header.diffuseColor),
        ("opacity", header.opacity),
        ("scale_0", header.scale),
        ("rot_0", header.rotation),
    ]
    for anchor in requiredAnchors where !anchor.property.isSet {
        throw GaussianSplatLoadingError.missingRequiredProperty(anchor.name)
    }

    guard header.numSHCoeffs.isMultiple(of: 3) else {
        throw GaussianSplatLoadingError.invalidHeader(
            "f_rest property count \(header.numSHCoeffs) is not divisible by 3"
        )
    }
    let tupleSH = header.numSHCoeffs / 3
    let degreeSH: UInt8
    switch tupleSH + 1 {
    case 1:  degreeSH = 0
    case 4:  degreeSH = 1
    case 9:  degreeSH = 2
    case 16: degreeSH = 3
    default:
        throw GaussianSplatLoadingError.invalidHeader(
            "unsupported spherical harmonics layout (\(header.numSHCoeffs) f_rest properties)"
        )
    }

    // Sum byte sizes of all elements that appear before vertex in the binary section.
    var vertexDataOffset = 0
    for index in 0..<vertexElementIndex {
        vertexDataOffset += header.elements[index].byteSize * header.elements[index].count
    }

    let rowByteStride = header.elements[vertexElementIndex].byteSize
    guard rowByteStride > 0 else {
        throw GaussianSplatLoadingError.invalidHeader("vertex element declares no properties")
    }
    let vertexStart = header.binaryStart + vertexDataOffset
    let expectedBytes = rowByteStride * header.vertexCount
    let availableBytes = fileData.count - vertexStart
    guard availableBytes >= expectedBytes else {
        throw GaussianSplatLoadingError.truncatedBody(
            expectedBytes: expectedBytes,
            availableBytes: max(0, availableBytes)
        )
    }

    return GaussianSplatData(
        data: fileData,
        vertexStart: vertexStart,
        rowByteStride: rowByteStride,
        splatCount: header.vertexCount,
        degreeSH: degreeSH,
        tupleSH: UInt8(tupleSH),
        position: header.position,
        scale: header.scale,
        rotation: header.rotation,
        opacity: header.opacity,
        diffuseColor: header.diffuseColor,
        sphericalHarmonics: header.sphericalHarmonics
    )
}

/// Walks the file byte by byte until `end_header`, feeding each line to
/// `PLYHeader.processLine`. Stops at the first byte after `end_header`, which
/// is also the first byte of the binary body.
private nonisolated func parseHeader(from fileData: Data) throws -> PLYHeader {
    // Real splat headers are a few KB; bail early on binary garbage instead of
    // walking hundreds of MB looking for `end_header`.
    let maxHeaderBytes = 1 << 20
    var header = PLYHeader()
    var line = ""
    var lineIndex = 0
    for (idx, byte) in fileData.enumerated() {
        guard idx < maxHeaderBytes else {
            throw GaussianSplatLoadingError.invalidHeader("missing end_header in the first \(maxHeaderBytes) bytes")
        }
        if byte == 0xA {
            if try header.processLine(line, at: lineIndex, endingAt: idx) { return header }
            line = ""
            lineIndex += 1
        } else {
            line.append(Character(Unicode.Scalar(byte)))
        }
    }
    throw GaussianSplatLoadingError.invalidHeader("missing end_header")
}

// MARK: - Deinterleaving

/// Replaces NaN/Inf with 0. RealityKit's splat asset validation rejects any
/// buffer containing non-finite floats, so loaders must sanitize before upload.
@inline(__always)
private nonisolated func sanitize(_ value: Float) -> Float {
    value.isFinite ? value : 0
}

/// Copies a single PLY field from the interleaved vertex rows into a destination buffer.
private nonisolated func copyPLYField(
    _ prop: PLYProperty,
    from src: UnsafeRawBufferPointer,
    into buffer: LowLevelBuffer,
    splatData: GaussianSplatData
) {
    buffer.withUnsafeMutableBytes { dst in
        let out = dst.bindMemory(to: Float.self)
        var rowStart = splatData.vertexStart
        var dstOff = 0
        for _ in 0..<splatData.splatCount {
            for floatIdx in 0..<prop.floatCount {
                let value = src.loadUnaligned(
                    fromByteOffset: rowStart + prop.byteOffset + floatIdx * 4,
                    as: Float.self
                )
                out[dstOff + floatIdx] = sanitize(value)
            }
            rowStart += splatData.rowByteStride
            dstOff += prop.floatCount
        }
    }
}

/// Writes the combined SH+DC buffer: prepends DC (diffuse color), then transposes
/// higher-order coefficients from channel-major (PLY) to RGB-interleaved (GPU) order.
///
/// PLY layout:  [R0...RN, G0...GN, B0...BN] per splat
/// GPU layout:  [RGB_dc, RGB_0, RGB_1, ..., RGB_N] per splat
private nonisolated func fillSHBuffer(
    _ buffer: LowLevelBuffer,
    from src: UnsafeRawBufferPointer,
    splatData: GaussianSplatData
) {
    buffer.withUnsafeMutableBytes { dst in
        let out = dst.bindMemory(to: Float.self)
        let tupleCount = Int(splatData.tupleSH)
        var rowStart = splatData.vertexStart
        var dstOff = 0
        for _ in 0..<splatData.splatCount {
            let dcBase = rowStart + splatData.diffuseColor.byteOffset
            out[dstOff]     = sanitize(src.loadUnaligned(fromByteOffset: dcBase, as: Float.self))
            out[dstOff + 1] = sanitize(src.loadUnaligned(fromByteOffset: dcBase + 4, as: Float.self))
            out[dstOff + 2] = sanitize(src.loadUnaligned(fromByteOffset: dcBase + 8, as: Float.self))
            dstOff += 3
            // Degree-0 files have no f_rest properties; never touch the unset SH offset.
            if tupleCount > 0 {
                let shBase = rowStart + splatData.sphericalHarmonics.byteOffset
                for tuple in 0..<tupleCount {
                    for channel in 0..<3 {
                        let value = src.loadUnaligned(
                            fromByteOffset: shBase + (channel * tupleCount + tuple) * 4,
                            as: Float.self
                        )
                        out[dstOff] = sanitize(value)
                        dstOff += 1
                    }
                }
            }
            rowStart += splatData.rowByteStride
        }
    }
}

/// Deinterleaves the packed PLY binary data into separate per-field GPU buffers.
///
/// The PLY binary section stores all properties for each splat consecutively
/// (AoS layout). This function extracts each field into its own `LowLevelBuffer`
/// (SoA layout) as required by the RealityKit splatting API.
nonisolated func deinterleaveGaussianSplatData(_ splatData: GaussianSplatData) throws -> GaussianSplatBuffers {
    let count = splatData.splatCount
    func alignedSize(floatsPerSplat: Int) -> Int { (count * floatsPerSplat * 4 + 15) & ~0xF }
    // The output SH buffer holds the diffuse color triplet plus `tupleSH`
    // higher-order triplets per splat, sized independently of whether the
    // file declared any f_rest properties.
    let shFloatsPerSplat = 3 * (Int(splatData.tupleSH) + 1)

    guard
        let positionBuffer = try? LowLevelBuffer(
            descriptor: .init(capacity: alignedSize(floatsPerSplat: splatData.position.floatCount), sizeMultiple: 16)),
        let scaleBuffer = try? LowLevelBuffer(
            descriptor: .init(capacity: alignedSize(floatsPerSplat: splatData.scale.floatCount), sizeMultiple: 16)),
        let rotationBuffer = try? LowLevelBuffer(
            descriptor: .init(capacity: alignedSize(floatsPerSplat: splatData.rotation.floatCount), sizeMultiple: 16)),
        let opacityBuffer = try? LowLevelBuffer(
            descriptor: .init(capacity: alignedSize(floatsPerSplat: splatData.opacity.floatCount), sizeMultiple: 16)),
        let shBuffer = try? LowLevelBuffer(
            descriptor: .init(capacity: alignedSize(floatsPerSplat: shFloatsPerSplat), sizeMultiple: 16))
    else {
        throw GaussianSplatLoadingError.bufferAllocationFailed
    }

    splatData.data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
        copyPLYField(splatData.position, from: src, into: positionBuffer, splatData: splatData)
        copyPLYField(splatData.scale, from: src, into: scaleBuffer, splatData: splatData)
        copyPLYField(splatData.rotation, from: src, into: rotationBuffer, splatData: splatData)
        copyPLYField(splatData.opacity, from: src, into: opacityBuffer, splatData: splatData)
        fillSHBuffer(shBuffer, from: src, splatData: splatData)
    }

    return GaussianSplatBuffers(
        splatCount: splatData.splatCount,
        degreeSH: splatData.degreeSH,
        tupleSH: splatData.tupleSH,
        positionBuffer: positionBuffer,
        scaleBuffer: scaleBuffer,
        rotationBuffer: rotationBuffer,
        opacityBuffer: opacityBuffer,
        shBuffer: shBuffer
    )
}

// MARK: - Orientation

/// Standard 3D Gaussian Splatting exports (INRIA, SuperSplat) load upside down
/// and facing away in RealityKit's coordinate space (verified on device with
/// Bear.ply). The baseline correction is a 180° rotation about the Z axis
/// (x → -x, y → -y), which stands the asset upright and turns it toward the
/// viewer. Pre-rotated exports (such as Apple's plant.ply sample asset) would
/// come out upside down; standard exporter output is the convention this
/// package targets.
nonisolated func rotateStandardSplatsToRealityKit(_ buffers: GaussianSplatBuffers) {
    let count = buffers.splatCount

    buffers.positionBuffer.withUnsafeMutableBytes { raw in
        let floats = raw.bindMemory(to: Float.self)
        for splat in 0..<count {
            floats[splat * 3]     = -floats[splat * 3]
            floats[splat * 3 + 1] = -floats[splat * 3 + 1]
        }
    }

    // Rotating by r = (w: 0, x: 0, y: 0, z: 1) — 180° about Z — maps each
    // splat's orientation quaternion q = (w, x, y, z) to r ⊗ q = (-z, -y, x, w).
    buffers.rotationBuffer.withUnsafeMutableBytes { raw in
        let floats = raw.bindMemory(to: Float.self)
        for splat in 0..<count {
            let base = splat * 4
            let (w, x, y, z) = (floats[base], floats[base + 1], floats[base + 2], floats[base + 3])
            floats[base]     = -z
            floats[base + 1] = -y
            floats[base + 2] = x
            floats[base + 3] = w
        }
    }

    // Under (x, y, z) → (-x, -y, z), each real spherical harmonics basis
    // polynomial keeps or flips its sign by the parity of its x and y powers.
    // The diffuse color (l = 0) is invariant. Tuples follow the 3DGS f_rest
    // order: l = 1 (m = -1, 0, 1), l = 2 (m = -2…2), l = 3 (m = -3…3).
    let tupleSigns: [Float] = [
        -1, +1, -1,                     // l = 1: y, z, x
        +1, -1, +1, -1, +1,             // l = 2: xy, yz, z², xz, x²-y²
        -1, +1, -1, +1, -1, +1, -1,     // l = 3
    ]
    let tupleCount = min(Int(buffers.tupleSH), tupleSigns.count)
    guard tupleCount > 0 else { return }
    buffers.shBuffer.withUnsafeMutableBytes { raw in
        let floats = raw.bindMemory(to: Float.self)
        let strideFloats = 3 * (Int(buffers.tupleSH) + 1)
        for splat in 0..<count {
            let base = splat * strideFloats + 3 // Skip the diffuse color triplet.
            for tuple in 0..<tupleCount where tupleSigns[tuple] < 0 {
                floats[base + tuple * 3]     = -floats[base + tuple * 3]
                floats[base + tuple * 3 + 1] = -floats[base + tuple * 3 + 1]
                floats[base + tuple * 3 + 2] = -floats[base + tuple * 3 + 2]
            }
        }
    }
}
