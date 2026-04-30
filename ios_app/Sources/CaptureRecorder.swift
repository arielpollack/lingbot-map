import ARKit
import CoreImage
import Foundation
import simd
import UIKit

@MainActor
final class CaptureRecorder: NSObject, ObservableObject {
    enum Phase: Equatable {
        case initializing
        case ready
        case recording
        case stopped(directory: URL, frameCount: Int)
    }

    @Published private(set) var phase: Phase = .initializing
    @Published private(set) var status: String = "Initializing AR…"
    @Published private(set) var frameCount: Int = 0
    @Published private(set) var sessionBytes: Int64 = 0
    @Published private(set) var lidarSupported: Bool = false
    @Published private(set) var cameraOn: Bool = false
    /// Human-readable AR tracking quality string surfaced from
    /// `session(_:cameraDidChangeTrackingState:)`. Lets the user know when
    /// SLAM has actually initialized — until then ARKit refuses to integrate
    /// LiDAR depth into the on-device mesh, so REC won't produce anything
    /// useful.
    @Published private(set) var trackingState: String = "—"
    @Published private(set) var trackingOK: Bool = false

    private weak var session: ARSession?
    let log: LogStream

    // ── Capture quality knobs ──
    // We rely on ARKit's on-device LiDAR mesh fusion (sceneReconstruction =
    // .mesh) for the geometry, but we ALSO ship per-frame LiDAR depth +
    // ARConfidence so the server's gsplat trainer can use them as
    // depth-supervised loss (lidar_mesh tier). Bundle stays small: 30
    // frames × (192×256×4 + 192×256×1) ≈ 7 MB uncompressed, ~3 MB after
    // the zip deflate.
    //
    // 2 fps is plenty for that: top-K=4 frame selection in the texture
    // baker means anything beyond ~30 well-spaced frames per scan adds
    // server time without improving the atlas.
    private let captureFps: Double = 2.0
    /// Long-side cap for the saved JPEG.
    private let jpegLongSideCap: CGFloat = 1280
    /// JPEG quality. 0.8 keeps texture sharpness usable for close-up
    /// viewing of the textured mesh.
    private let jpegQuality: Double = 0.8
    /// Serial queue. Concurrency here previously let many in-flight writes
    /// each retain a CVPixelBuffer (which retains its parent ARFrame),
    /// triggering ARKit's "delegate is retaining N ARFrames" warning. We now
    /// pre-encode JPEG on the AR queue and only ship `Data` here, so the
    /// ARFrame is always released before this queue ever runs.
    private let writeQueue = DispatchQueue(
        label: "com.lingbot.capture.write",
        qos: .userInitiated
    )
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let delegateProxy = SessionDelegateProxy()

    // Capture state — accessed from the AR delegate queue and from main.
    private let stateLock = NSLock()
    private var recording = false
    private var captureDirectory: URL? = nil
    private var lastCaptureTimestamp: TimeInterval = 0
    private var captureCount: Int = 0
    private var imageResolution: CGSize = .zero
    private var depthResolution: CGSize = .zero

    init(log: LogStream) {
        self.log = log
        super.init()
        delegateProxy.owner = self
        lidarSupported = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        log.log("recorder init; lidar supported: \(lidarSupported)")
    }

    /// Attach to the ARSession owned by ARSCNView. Called by ARSessionView
    /// once the view is in the hierarchy. Wires the delegate and runs the
    /// world-tracking config so the camera feed starts.
    func attach(to session: ARSession) {
        self.session = session
        session.delegate = delegateProxy
        log.log("attached to ARSession")
        runConfig(on: session)
    }

    private func runConfig(on session: ARSession) {
        // Idempotent: silently no-op if the session is already running. The
        // ARKit runtime would log "Attempting to enable an already-enabled
        // session. Ignoring..." otherwise — it's harmless but noisy and
        // points at a real bug (we used to call this both from attach() and
        // from a re-render of ARSessionView).
        if cameraOn { return }
        guard ARWorldTrackingConfiguration.isSupported else {
            phase = .initializing
            status = "World tracking not supported on this device."
            log.log("ERROR: world tracking unsupported")
            return
        }
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.planeDetection = []
        // Enable on-device LiDAR mesh fusion. ARKit accumulates depth into
        // ARMeshAnchors using Metal compute, so we don't have to upload the
        // raw per-frame depth maps. We still leave .sceneDepth on so the
        // depth buffer is ALSO available if we want diagnostics, but we
        // don't write it to disk.
        if type(of: config).supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        // Keep the default videoFormat — picking from supportedVideoFormats
        // can land on a mode incompatible with sceneDepth, which makes the
        // camera fail to start (FigCaptureSourceRemote err=-12784).
        session.run(config, options: [.removeExistingAnchors, .resetTracking])
        cameraOn = true
        phase = .ready
        status = lidarSupported ? "LiDAR ready. Tap REC." : "AR ready (no LiDAR). Tap REC."
        log.log(
            "ARSession running; sceneDepth=\(config.frameSemantics.contains(.sceneDepth)) "
            + "meshRecon=\(config.sceneReconstruction == .mesh)"
        )
    }

    func toggleCamera() {
        guard let session = session else { return }
        if cameraOn {
            session.pause()
            cameraOn = false
            status = "Camera paused."
            log.log("camera paused")
        } else {
            runConfig(on: session)
        }
    }

    func pause() {
        session?.pause()
        cameraOn = false
        log.log("recorder paused")
    }

    // MARK: Recording control

    func startRecording() {
        switch phase {
        case .ready, .stopped:
            break
        default:
            return
        }
        let id = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("session-\(id)", isDirectory: true)
        do {
            try? FileManager.default.removeItem(at: dir)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(
                at: dir.appendingPathComponent("frames"), withIntermediateDirectories: true)
            try FileManager.default.createDirectory(
                at: dir.appendingPathComponent("poses"), withIntermediateDirectories: true)
            try FileManager.default.createDirectory(
                at: dir.appendingPathComponent("depth"), withIntermediateDirectories: true)
        } catch {
            status = "Cannot create session dir: \(error.localizedDescription)"
            log.log("ERROR creating session dir: \(error)")
            return
        }
        stateLock.lock()
        recording = true
        captureDirectory = dir
        lastCaptureTimestamp = 0
        captureCount = 0
        imageResolution = .zero
        depthResolution = .zero
        stateLock.unlock()
        frameCount = 0
        sessionBytes = 0
        phase = .recording
        status = "Recording…"
        log.log("REC started → \(dir.lastPathComponent)")
    }

    func stopRecording() {
        guard case .recording = phase else { return }
        log.log("STOP requested; draining writes")
        stateLock.lock()
        recording = false
        let dir = captureDirectory
        let count = captureCount
        let imageRes = imageResolution
        let depthRes = depthResolution
        captureDirectory = nil
        stateLock.unlock()

        guard let dir = dir else { return }

        // Snapshot the current mesh anchors from the AR session BEFORE we
        // hand off — these MTLBuffers are owned by ARKit and are alive only
        // as long as the anchor is in the current frame.
        let meshAnchors: [ARMeshAnchor] = (session?.currentFrame?.anchors ?? [])
            .compactMap { $0 as? ARMeshAnchor }
        log.log("STOP: \(meshAnchors.count) live mesh anchors")

        // Drain pending writes (serial queue), then write OBJ + manifest.
        writeQueue.async { [weak self] in
            guard let self = self else { return }
            let lidar = self.lidarSupportedSnapshot()

            // Write the on-device-fused mesh (LiDAR-only path).
            var meshStats: MeshExporter.Stats? = nil
            if lidar, !meshAnchors.isEmpty {
                let meshURL = dir.appendingPathComponent("mesh.obj")
                do {
                    meshStats = try MeshExporter.writeOBJ(anchors: meshAnchors, to: meshURL)
                    self.log.log(
                        String(
                            format: "mesh.obj written in %.1fs · %d verts · %d faces · %@",
                            meshStats!.secondsElapsed,
                            meshStats!.vertexCount,
                            meshStats!.faceCount,
                            formatBytes(meshStats!.bytes)
                        ))
                } catch {
                    self.log.log("mesh.obj FAILED: \(error)")
                }
            } else {
                self.log.log("no mesh anchors → skipping mesh.obj")
            }

            let tier: String
            if lidar && meshStats != nil {
                tier = "lidar_mesh"
            } else if lidar {
                tier = "lidar"
            } else {
                tier = "vio"
            }

            let manifest = self.makeManifestDataNonisolated(
                tier: tier,
                frameCount: count,
                imageResolution: imageRes,
                depthResolution: depthRes,
                meshStats: meshStats
            )
            let manifestURL = dir.appendingPathComponent("manifest.json")
            try? manifest.write(to: manifestURL)

            DispatchQueue.main.async {
                self.frameCount = count
                self.sessionBytes = Self.directorySize(dir)
                self.status = "Captured \(count) frames · \(formatBytes(self.sessionBytes))"
                self.phase = .stopped(directory: dir, frameCount: count)
                self.log.log(
                    "STOP done; tier=\(tier), \(count) frames, "
                    + "\(formatBytes(self.sessionBytes)) on disk"
                )
            }
        }
    }

    private nonisolated func lidarSupportedSnapshot() -> Bool {
        ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    fileprivate func handleARFrame(_ frame: ARFrame) {
        // Throttle decision — atomic.
        var shouldCapture = false
        var index = 0
        var dir: URL? = nil

        stateLock.lock()
        if recording, let d = captureDirectory {
            let interval = 1.0 / captureFps
            if frame.timestamp - lastCaptureTimestamp >= interval || captureCount == 0 {
                lastCaptureTimestamp = frame.timestamp
                index = captureCount
                captureCount += 1
                dir = d
                shouldCapture = true
                if imageResolution == .zero { imageResolution = frame.camera.imageResolution }
                if depthResolution == .zero, let depth = frame.sceneDepth {
                    depthResolution = CGSize(
                        width: CVPixelBufferGetWidth(depth.depthMap),
                        height: CVPixelBufferGetHeight(depth.depthMap)
                    )
                }
            }
        }
        stateLock.unlock()

        guard shouldCapture, let dir = dir else { return }

        // Snapshot data we need from the ARFrame, *synchronously*. JPEG
        // encoding here releases the CVPixelBuffer (and the ARFrame) before
        // we leave this stack frame, so the writeQueue closure below has no
        // ARKit references — no retention pile-up.
        let intrinsics = frame.camera.intrinsics
        let transform = frame.camera.transform
        let imageRes = frame.camera.imageResolution

        let pixelBuffer = frame.capturedImage
        var ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        // Downsample before JPEG-encoding.
        let extent = ciImage.extent
        let longSide = max(extent.width, extent.height)
        if longSide > jpegLongSideCap {
            let scale = jpegLongSideCap / longSide
            ciImage = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let jpegData: Data? = ciContext.jpegRepresentation(
            of: ciImage,
            colorSpace: colorSpace,
            options: [
                kCGImageDestinationLossyCompressionQuality
                    as CIImageRepresentationOption: jpegQuality
            ]
        )

        // Snapshot depth + ARConfidence synchronously while the ARFrame
        // is still alive. Same retention-safety reasoning as the JPEG
        // above — we copy the raw bytes off the CVPixelBuffer here and
        // ship plain Data into the writeQueue.
        let depthSnap: DepthSnapshot? = frame.sceneDepth.flatMap { snapshotDepth($0) }
        let depthRes: CGSize? = depthSnap.map {
            CGSize(width: $0.width, height: $0.height)
        }

        let poseData = encodePose(
            intrinsics: intrinsics,
            transform: transform,
            imageResolution: imageRes,
            depthResolution: depthRes
        )

        let frameURL = dir.appendingPathComponent("frames/\(formatIndex(index)).jpg")
        let poseURL = dir.appendingPathComponent("poses/\(formatIndex(index)).json")
        let depthBinURL = dir.appendingPathComponent("depth/\(formatIndex(index)).bin")
        let depthConfURL = dir.appendingPathComponent("depth/\(formatIndex(index)).conf")

        var depthBytes = 0
        if let snap = depthSnap {
            depthBytes = snap.depthBytes.count + snap.confidenceBytes.count
        }
        let bytesAdded = Int64(
            (jpegData?.count ?? 0) + poseData.count + depthBytes
        )

        writeQueue.async {
            if let data = jpegData {
                try? data.write(to: frameURL)
            }
            try? poseData.write(to: poseURL)
            if let snap = depthSnap {
                try? snap.depthBytes.write(to: depthBinURL)
                try? snap.confidenceBytes.write(to: depthConfURL)
            }
        }

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.frameCount = index + 1
            self.sessionBytes += bytesAdded
            // Periodic log every 10 frames so we can see capture progress
            // without flooding.
            if (index + 1) % 10 == 0 {
                self.log.log(
                    "captured \(index + 1) frames · \(formatBytes(self.sessionBytes))"
                )
            }
        }
    }

    fileprivate func handleSessionError(_ error: Error) {
        DispatchQueue.main.async {
            self.status = "AR error: \(error.localizedDescription)"
            self.log.log("ARSession error: \(error.localizedDescription)")
        }
    }

    /// ARKit reports tracking quality changes here. We surface them to the
    /// user so they understand when SLAM is initialized (and the
    /// `Skipping integration due to poor slam` Xcode log spam is over).
    fileprivate func handleTrackingChange(_ state: ARCamera.TrackingState) {
        let (text, ok): (String, Bool)
        switch state {
        case .notAvailable:
            (text, ok) = ("AR unavailable", false)
        case .normal:
            (text, ok) = ("Tracking OK", true)
        case .limited(let reason):
            switch reason {
            case .initializing:
                (text, ok) = ("Initializing — move slowly", false)
            case .relocalizing:
                (text, ok) = ("Relocalizing — return to the scanned area", false)
            case .excessiveMotion:
                (text, ok) = ("Slow down — moving too fast", false)
            case .insufficientFeatures:
                (text, ok) = ("Add light or move to a textured area", false)
            @unknown default:
                (text, ok) = ("Tracking limited", false)
            }
        }
        DispatchQueue.main.async {
            if self.trackingState != text {
                self.log.log("tracking → \(text)")
            }
            self.trackingState = text
            self.trackingOK = ok
        }
    }

    // MARK: Manifest

    private nonisolated func makeManifestDataNonisolated(
        tier: String,
        frameCount: Int,
        imageResolution: CGSize,
        depthResolution: CGSize,
        meshStats: MeshExporter.Stats?
    ) -> Data {
        var dict: [String: Any] = [
            // v3: re-introduces per-frame `depth/<idx>.bin` + `.conf`
            // sidecars for every keyframe, alongside the previously-shipped
            // mesh.obj + frames/ + poses/. The server's gsplat trainer
            // uses these as depth-supervised loss for the lidar_mesh tier.
            "version": 3,
            "tier": tier,
            "frame_count": frameCount,
            "fps": captureFpsSnapshot,
            "image_resolution": [Int(imageResolution.width), Int(imageResolution.height)],
            "captured_at": ISO8601DateFormatter().string(from: Date()),
            "device": deviceIdentifier(),
            "transform_convention": "c2w_4x4_row_major_arkit_world",
            "intrinsics_convention": "fx_fy_cx_cy_at_image_resolution",
        ]
        if depthResolution != .zero {
            dict["depth_resolution"] = [
                Int(depthResolution.width),
                Int(depthResolution.height),
            ]
            dict["depth_format"] = "float32_le_meters"
            dict["confidence_format"] = "uint8_arconfidence"
        }
        if let m = meshStats {
            dict["mesh"] = [
                "format": "obj",
                "vertex_count": m.vertexCount,
                "face_count": m.faceCount,
                "anchor_count": m.anchorCount,
                "bytes": m.bytes,
                "frame_convention": "arkit_world",
            ] as [String: Any]
        }
        return (try? JSONSerialization.data(
            withJSONObject: dict,
            options: [.prettyPrinted, .sortedKeys]
        )) ?? Data()
    }

    /// Capture FPS as a plain Double — used from the nonisolated manifest
    /// builder, which can't read the main-actor `captureFps` directly.
    private nonisolated var captureFpsSnapshot: Double { 2.0 }

    private nonisolated func deviceIdentifier() -> String {
        var sysinfo = utsname()
        uname(&sysinfo)
        let mirror = Mirror(reflecting: sysinfo.machine)
        let chars = mirror.children.compactMap { ($0.value as? Int8).map { UInt8(bitPattern: $0) } }
        return String(bytes: chars.filter { $0 != 0 }, encoding: .utf8) ?? "unknown"
    }

    /// Recursively sum file sizes under `dir`. Used to report bundle size
    /// after capture is finished. Cheap enough at our typical scale (~hundreds
    /// of small files); we call it once on stop, not per-frame. Nonisolated
    /// so the Bundler can call it from a background queue.
    nonisolated static func directorySize(_ dir: URL) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: dir,
            includingPropertiesForKeys: [.totalFileAllocatedSizeKey, .isRegularFileKey]
        ) else { return 0 }
        var total: Int64 = 0
        for case let url as URL in enumerator {
            let values = try? url.resourceValues(forKeys: [.totalFileAllocatedSizeKey, .isRegularFileKey])
            if values?.isRegularFile == true {
                total += Int64(values?.totalFileAllocatedSize ?? 0)
            }
        }
        return total
    }
}

// MARK: - ARSession delegate proxy

private final class SessionDelegateProxy: NSObject, ARSessionDelegate {
    weak var owner: CaptureRecorder?

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        MainActor.assumeIsolated {
            owner?.handleARFrame(frame)
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        MainActor.assumeIsolated {
            owner?.handleSessionError(error)
        }
    }

    func session(
        _ session: ARSession,
        cameraDidChangeTrackingState camera: ARCamera
    ) {
        MainActor.assumeIsolated {
            owner?.handleTrackingChange(camera.trackingState)
        }
    }
}

// MARK: - Frame helpers

private struct DepthSnapshot {
    let width: Int
    let height: Int
    let depthBytes: Data
    let confidenceBytes: Data
}

private func snapshotDepth(_ scene: ARDepthData) -> DepthSnapshot? {
    let depthMap = scene.depthMap
    CVPixelBufferLockBaseAddress(depthMap, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
    let w = CVPixelBufferGetWidth(depthMap)
    let h = CVPixelBufferGetHeight(depthMap)
    let bpr = CVPixelBufferGetBytesPerRow(depthMap)
    guard let base = CVPixelBufferGetBaseAddress(depthMap) else { return nil }

    var depthData = Data(count: w * h * MemoryLayout<Float32>.size)
    depthData.withUnsafeMutableBytes { dest in
        guard let dst = dest.bindMemory(to: UInt8.self).baseAddress else { return }
        for row in 0..<h {
            memcpy(
                dst.advanced(by: row * w * 4),
                base.advanced(by: row * bpr),
                w * 4
            )
        }
    }

    var confData = Data(count: w * h)
    if let conf = scene.confidenceMap {
        CVPixelBufferLockBaseAddress(conf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(conf, .readOnly) }
        let cw = CVPixelBufferGetWidth(conf)
        let ch = CVPixelBufferGetHeight(conf)
        let cbpr = CVPixelBufferGetBytesPerRow(conf)
        guard let cBase = CVPixelBufferGetBaseAddress(conf) else { return nil }
        let copyW = min(cw, w)
        let copyH = min(ch, h)
        confData.withUnsafeMutableBytes { dest in
            guard let dst = dest.bindMemory(to: UInt8.self).baseAddress else { return }
            for row in 0..<copyH {
                memcpy(
                    dst.advanced(by: row * w),
                    cBase.advanced(by: row * cbpr),
                    copyW
                )
            }
        }
    }

    return DepthSnapshot(width: w, height: h, depthBytes: depthData, confidenceBytes: confData)
}

private func encodePose(
    intrinsics: simd_float3x3,
    transform: simd_float4x4,
    imageResolution: CGSize,
    depthResolution: CGSize?
) -> Data {
    func mat3(_ m: simd_float3x3) -> [[Float]] {
        // simd is column-major; emit row-major arrays for human-readable JSON.
        (0..<3).map { row in (0..<3).map { col in m[col][row] } }
    }
    func mat4(_ m: simd_float4x4) -> [[Float]] {
        (0..<4).map { row in (0..<4).map { col in m[col][row] } }
    }
    var dict: [String: Any] = [
        "intrinsics": mat3(intrinsics),
        "transform_c2w": mat4(transform),
        "image_resolution": [Int(imageResolution.width), Int(imageResolution.height)],
    ]
    if let depthRes = depthResolution {
        dict["depth_resolution"] = [Int(depthRes.width), Int(depthRes.height)]
    }
    return (try? JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys])) ?? Data()
}

private func formatIndex(_ index: Int) -> String {
    String(format: "%06d", index)
}
