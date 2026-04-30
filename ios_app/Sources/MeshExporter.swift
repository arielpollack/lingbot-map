import ARKit
import Foundation
import simd

/// Serialize a snapshot of `ARMeshAnchor`s to a single Wavefront OBJ file
/// in the ARKit world coordinate frame.
///
/// Each anchor exposes its geometry in its own local frame and carries a
/// 4×4 `transform` mapping local → world. We concatenate per-anchor verts
/// after transforming to world, and offset face indices so the resulting
/// OBJ refers to a single combined vertex list.
///
/// This is what lets us upload a single ~2-5 MB mesh instead of ~25 MB of
/// per-frame depth + confidence — ARKit has already done the LiDAR depth
/// fusion for us using Metal compute shaders on the device.
enum MeshExporter {
    struct Stats {
        let vertexCount: Int
        let faceCount: Int
        let anchorCount: Int
        let bytes: Int64
        let secondsElapsed: Double
    }

    static func writeOBJ(anchors: [ARMeshAnchor], to url: URL) throws -> Stats {
        let start = Date()
        var output = String()
        output.reserveCapacity(anchors.reduce(0) { $0 + $1.geometry.vertices.count } * 32)
        var vertexBase = 0
        var totalVertices = 0
        var totalFaces = 0

        for anchor in anchors {
            let geom = anchor.geometry
            let transform = anchor.transform

            // ── vertices ──────────────────────────────────────────────
            let vSource = geom.vertices
            let vBytesPerVec = vSource.stride
            let vBufferBase = vSource.buffer.contents().advanced(by: vSource.offset)

            for i in 0..<vSource.count {
                let p = vBufferBase
                    .advanced(by: i * vBytesPerVec)
                    .bindMemory(to: SIMD3<Float>.self, capacity: 1)
                    .pointee
                let world4 = transform * SIMD4<Float>(p.x, p.y, p.z, 1)
                output.append("v \(world4.x) \(world4.y) \(world4.z)\n")
            }
            totalVertices += vSource.count

            // ── faces ─────────────────────────────────────────────────
            let fElement = geom.faces
            let perFace = fElement.indexCountPerPrimitive  // 3 for triangles
            let bytesPerIndex = fElement.bytesPerIndex
            let fPtr = fElement.buffer.contents()

            for f in 0..<fElement.count {
                let base = f * perFace * bytesPerIndex
                var indices: [Int] = []
                indices.reserveCapacity(perFace)
                for k in 0..<perFace {
                    let off = base + k * bytesPerIndex
                    let raw: Int
                    switch bytesPerIndex {
                    case 4:
                        raw = Int(fPtr.advanced(by: off)
                            .bindMemory(to: UInt32.self, capacity: 1).pointee)
                    case 2:
                        raw = Int(fPtr.advanced(by: off)
                            .bindMemory(to: UInt16.self, capacity: 1).pointee)
                    default:
                        raw = 0
                    }
                    indices.append(raw + vertexBase + 1)  // OBJ is 1-indexed
                }
                if indices.count == 3 {
                    output.append("f \(indices[0]) \(indices[1]) \(indices[2])\n")
                }
            }
            totalFaces += fElement.count
            vertexBase += vSource.count
        }

        try output.write(to: url, atomically: true, encoding: .utf8)
        let bytes =
            ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size]
                as? NSNumber)?.int64Value ?? 0
        return Stats(
            vertexCount: totalVertices,
            faceCount: totalFaces,
            anchorCount: anchors.count,
            bytes: bytes,
            secondsElapsed: Date().timeIntervalSince(start)
        )
    }
}
