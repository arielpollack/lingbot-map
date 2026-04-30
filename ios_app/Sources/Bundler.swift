import Foundation
import ZIPFoundation

enum Bundler {
    /// Zip the capture session directory into `<directory>.zip` and return
    /// (zipURL, inputBytes, zipBytes).
    ///
    /// Logs start/finish + final size so we can see whether zip is the slow
    /// phase (it usually is for big LiDAR captures: 100 frames at LiDAR
    /// resolution + JPEG ≈ 60 MB and zip+deflate of binary depth is the
    /// bottleneck).
    static func zipSession(at directory: URL, log: LogStream) throws -> (
        url: URL, inputBytes: Int64, zipBytes: Int64
    ) {
        let parent = directory.deletingLastPathComponent()
        let zipURL = parent.appendingPathComponent("\(directory.lastPathComponent).zip")
        try? FileManager.default.removeItem(at: zipURL)

        let inputSize = CaptureRecorder.directorySize(directory)
        log.log("zip start; \(formatBytes(inputSize)) on disk")
        let start = Date()

        try FileManager.default.zipItem(
            at: directory,
            to: zipURL,
            shouldKeepParent: false,
            compressionMethod: .deflate
        )

        let elapsed = Date().timeIntervalSince(start)
        let zipSize =
            ((try? FileManager.default.attributesOfItem(atPath: zipURL.path))?[.size]
                as? NSNumber)?.int64Value ?? 0
        log.log(
            String(
                format: "zip done in %.1fs; %@ → %@ (%.0f%% ratio)",
                elapsed,
                formatBytes(inputSize),
                formatBytes(zipSize),
                Double(zipSize) / max(Double(inputSize), 1) * 100
            ))
        return (zipURL, inputSize, zipSize)
    }
}
